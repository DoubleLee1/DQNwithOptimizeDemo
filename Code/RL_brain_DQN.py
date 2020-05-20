import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Cropping2D, Lambda, Dropout, \
    ZeroPadding2D,AveragePooling2D
from keras.optimizers import RMSprop, SGD, adam
from keras.utils import np_utils

np.random.seed(1)

# 测试控制台
batch_con = False        # if use continuity data for batch


class DQN:
    def __init__(
            self,
            n_actions,
            observation_shape,
            retrain=False,
            net_path=None,
            net_name=None,
            learning_rate=0.0001,  # 0.0001 for pre_train best
            reward_decay=0.9,
            epsilon_max=0.9,
            replace_target_iter=300,
            memory_size=10000,
            batch_size=128,
            good_memory_size=5000,
            bad_memory_size=5000,
            e_greedy_increment=0.001,
    ):
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_max
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.good_memory_size = good_memory_size
        self.bad_memory_size = bad_memory_size
        self.batch_size = batch_size
        # self.epsilon_increment = None if retrain is True else e_greedy_increment
        self.epsilon_increment = e_greedy_increment

        self.learn_step_counter = 0
        self.memoryList = []
        self.net_path = net_path
        self.net_name = net_name
        self.expert_memory = None
        self.good_memoryList = []
        self.bad_memoryList = []
        self.score_memory = []
        self.dropout = 0.5


        # consist of [target_net, evaluate_net]
        if retrain:
            self.build_net_retrain()
            self.epsilon = 0.9
        else:
            self._build_net()
            # e_greedy_increment 通过神经网络选择的概率慢慢增加
            self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max

        # 记录cost然后画出来
        self.cost_his = []
        self.reward = []

    def _build_net(self):

        # Build eval net
        self.model_eval = Sequential()
        self.model_eval.add(Lambda(lambda x: (x / 255.0) - 0.5, batch_input_shape=(None,self.observation_shape[0],
                                                                                 self.observation_shape[1],
                                                                                 self.observation_shape[2])))
        self.model_eval.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
        self.model_eval.add(AveragePooling2D((4, 4), strides=(2, 2)))
        self.model_eval.add(ZeroPadding2D((2, 2)))

        self.model_eval.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
        self.model_eval.add(Dropout(self.dropout))

        self.model_eval.add(Flatten())
        self.model_eval.add(Dense(1024, activation="relu"))
        self.model_eval.add(Dropout(self.dropout))
        self.model_eval.add(Dense(512, activation="relu"))
        self.model_eval.add(Dropout(self.dropout))
        self.model_eval.add(Dense(self.n_actions, activation="softmax"))

        rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
        sgd = SGD(lr=self.lr, decay=1e-6, momentum=0.7, nesterov=True)
        adamprop = adam(lr=self.lr)
        self.model_eval.compile(loss='mse', optimizer=rmsprop, metrics=['accuracy'])

        # Build target net
        self.model_target = Sequential()
        self.model_target.add(Lambda(lambda x: (x / 255.0) - 0.5, batch_input_shape=(None, self.observation_shape[0],
                                                                                   self.observation_shape[1],
                                                                                   self.observation_shape[2])))
        self.model_target.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
        self.model_target.add(AveragePooling2D((4, 4), strides=(2, 2)))
        self.model_target.add(ZeroPadding2D((2, 2)))

        self.model_target.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
        self.model_target.add(Dropout(self.dropout))

        self.model_target.add(Flatten())
        self.model_target.add(Dense(1024, activation="relu"))
        self.model_target.add(Dropout(self.dropout))
        self.model_target.add(Dense(512, activation="relu"))
        self.model_target.add(Dropout(self.dropout))
        self.model_target.add(Dense(self.n_actions, activation="softmax"))
    # def _build_net(self):
    #
    #     # Build eval net
    #     self.model_eval = Sequential()
    #     self.model_eval.add(Lambda(lambda x: (x / 255.0) - 0.5, batch_input_shape=(None,self.observation_shape[0],
    #                                                                              self.observation_shape[1],
    #                                                                              self.observation_shape[2])))
    #     self.model_eval.add(Convolution2D(10, (6, 6), activation='relu'))
    #     self.model_eval.add(MaxPooling2D((6, 6)))
    #
    #     self.model_eval.add(Convolution2D(20, (5, 5), activation='relu'))
    #     self.model_eval.add(Convolution2D(40, (3, 3), activation='relu'))
    #     self.model_eval.add(Dropout(self.dropout))
    #
    #     self.model_eval.add(Flatten())
    #     self.model_eval.add(Dense(1024, activation="relu"))
    #     self.model_eval.add(Dropout(self.dropout))
    #     self.model_eval.add(Dense(512, activation="relu"))
    #     self.model_eval.add(Dropout(self.dropout))
    #     self.model_eval.add(Dense(self.n_actions, activation="softmax"))
    #
    #     rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
    #     # sgd = SGD(lr=self.lr, decay=1e-6, momentum=0.7, nesterov=True)
    #     adamprop = adam(lr=self.lr)
    #     self.model_eval.compile(loss='mse', optimizer=adamprop, metrics=['accuracy'])
    #
    #     # Build target net
    #     self.model_target = Sequential()
    #     self.model_target.add(Lambda(lambda x: (x / 255.0) - 0.5, batch_input_shape=(None,self.observation_shape[0],
    #                                                                              self.observation_shape[1],
    #                                                                              self.observation_shape[2])))
    #     self.model_target.add(Convolution2D(10, (6, 6), activation='relu'))
    #     self.model_target.add(MaxPooling2D((6, 6)))
    #
    #     self.model_target.add(Convolution2D(20, (4, 4), activation='relu'))
    #     self.model_target.add(Dropout(self.dropout))
    #
    #     self.model_target.add(Flatten())
    #     self.model_target.add(Dense(1024, activation="relu"))
    #     self.model_target.add(Dropout(self.dropout))
    #     self.model_target.add(Dense(512, activation="relu"))
    #     self.model_target.add(Dropout(self.dropout))
    #     self.model_target.add(Dense(self.n_actions, activation="softmax"))

    # def _build_net(self):
    #
    #     # Build eval net
    #     self.model_eval = Sequential()
    #     self.model_eval.add(Lambda(lambda x: (x / 255.0) - 0.5, batch_input_shape=(None,self.observation_shape[0],
    #                                                                              self.observation_shape[1],
    #                                                                              self.observation_shape[2])))
    #     self.model_eval.add(Convolution2D(64, (11, 11), strides=(4, 4), activation='relu'))
    #     self.model_eval.add(MaxPooling2D((3, 3), strides=(2, 2)))
    #     self.model_eval.add(ZeroPadding2D((2, 2)))
    #
    #     self.model_eval.add(Convolution2D(192, (5, 5), activation='relu'))
    #     self.model_eval.add(MaxPooling2D((3, 3), strides=(2, 2)))
    #     self.model_eval.add(ZeroPadding2D((1, 1)))
    #     self.model_eval.add(Convolution2D(384, (3, 3), activation='relu'))
    #     self.model_eval.add(ZeroPadding2D((1, 1)))
    #     self.model_eval.add(Convolution2D(256, (3, 3), activation='relu'))
    #     self.model_eval.add(MaxPooling2D((3, 3), strides=(2, 2)))
    #     self.model_eval.add(Flatten())
    #     self.model_eval.add(Dense(4096, activation='relu'))
    #     self.model_eval.add(Dropout(self.dropout))
    #     self.model_eval.add(Dense(2048, activation='relu'))
    #     self.model_eval.add(Dropout(self.dropout))
    #     self.model_eval.add(Dense(self.n_actions, activation='softmax'))
    #
    #     rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
    #     # sgd = SGD(lr=self.lr, decay=1e-6, momentum=0.7, nesterov=True)
    #     adamprop = adam(lr=self.lr)
    #     self.model_eval.compile(loss='mse', optimizer=adamprop, metrics=['accuracy'])
    #
    #     # Build target net
    #     self.model_target = Sequential()
    #     self.model_target.add(Lambda(lambda x: (x / 255.0) - 0.5, batch_input_shape=(None, self.observation_shape[0],
    #                                                                                self.observation_shape[1],
    #                                                                                self.observation_shape[2])))
    #     self.model_target.add(Convolution2D(64, (11, 11), strides=(4, 4), activation='relu'))
    #     self.model_target.add(MaxPooling2D((3, 3), strides=(2, 2)))
    #     self.model_target.add(ZeroPadding2D((2, 2)))
    #
    #     self.model_target.add(Convolution2D(192, (5, 5), activation='relu'))
    #     self.model_target.add(MaxPooling2D((3, 3), strides=(2, 2)))
    #     self.model_target.add(ZeroPadding2D((1, 1)))
    #     self.model_target.add(Convolution2D(384, (3, 3), activation='relu'))
    #     self.model_target.add(ZeroPadding2D((1, 1)))
    #     self.model_target.add(Convolution2D(256, (3, 3), activation='relu'))
    #     self.model_target.add(MaxPooling2D((3, 3), strides=(2, 2)))
    #     self.model_target.add(Flatten())
    #     self.model_target.add(Dense(4096, activation='relu'))
    #     self.model_target.add(Dropout(self.dropout))
    #     self.model_target.add(Dense(2048, activation='relu'))
    #     self.model_target.add(Dropout(self.dropout))
    #     self.model_target.add(Dense(self.n_actions, activation='softmax'))

    def build_net_retrain(self):

        self.model_eval = load_model(self.net_path+'/'+self.net_name+'_eval_net.h5')

        adamprop = adam(lr=self.lr)
        rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
        sgd = SGD(lr=self.lr, decay=1e-6, momentum=0.7, nesterov=True)
        self.model_eval.compile(loss='mse', optimizer=rmsprop, metrics=['accuracy'])

        self.model_target = load_model(self.net_path+'/'+self.net_name+'_target_net.h5')
        print('Load Net successful *_*')

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        a = np.array(a, dtype='float32')
        r = np.array(r, dtype='float32')
        # 先进先出存储池
        if self.memory_counter > self.memory_size:
            self.memoryList.insert(0, [s, a, r, s_])
            del self.memoryList[-1]
        else:
            self.memoryList.insert(0, [s, a, r, s_])

        self.memory_counter += 1

    def store_good_transition(self, s, a, r, s_):
        if not hasattr(self, 'good_memory_counter'):
            self.good_memory_counter = 0

        a = np.array(a, dtype='float32')
        r = np.array(r, dtype='float32')
        # 先进先出存储池
        if self.good_memory_counter > self.good_memory_size:
            self.good_memoryList.insert(0, [s, a, r, s_])
            del self.good_memoryList[-1]
        else:
            self.good_memoryList.insert(0, [s, a, r, s_])

        self.good_memory_counter += 1

    def store_bad_transition(self, s, a, r, s_):
        if not hasattr(self, 'bad_memory_counter'):
            self.bad_memory_counter = 0

        a = np.array(a, dtype='float32')
        r = np.array(r, dtype='float32')
        # 先进先出存储池
        if self.bad_memory_counter > self.bad_memory_size:
            self.bad_memoryList.insert(0, [s, a, r, s_])
            del self.bad_memoryList[-1]
        else:
            self.bad_memoryList.insert(0, [s, a, r, s_])

        self.bad_memory_counter += 1

    def choose_action(self, observation):

        if np.random.uniform() < self.epsilon:
            observation = observation[np.newaxis, :]
            actions_value = self.model_eval.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def choose_memory(self):

        if batch_con:
            # 随机取出记忆(有序)
            if self.memory_counter > self.memory_size:
                sample_index = np.random.randint(0, self.memory_size-self.batch_size)
            else:
                sample_index = np.random.  randint(0, self.memory_counter - self.batch_size)
            batch_memory = self.memoryList[sample_index:sample_index+self.batch_size]
        else:
            # 随机取出记忆(无序,有重复)
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)
            batch_memory = [self.memoryList[i] for i in sample_index]

        # choose memory from expert demonstration and self with ration k (无序)
        # k1, k2 = 0.2, 0.4
        # memory_expert = int(self.batch_size*k1)
        # memory_good = int(self.batch_size*k2)
        # if memory_good > len(self.good_memoryList):
        #     memory_expert += memory_good  # good memory不够expert凑
        #     memory_good = 0
        # memory_self = self.batch_size-memory_expert-memory_good
        #
        # # choose self memory
        # if self.memory_counter > self.memory_size:
        #     sample_index = np.random.choice(self.memory_size, size=memory_self, replace=False)
        # else:
        #     sample_index = np.random.choice(self.memory_counter, size=memory_self, replace=False)
        # batch_memory = [self.memoryList[i] for i in sample_index]
        # # choose expert memory
        # sample_index = np.random.choice(len(self.expert_memory), size=memory_expert, replace=False)
        # batch_memory_expert = [self.expert_memory[i] for i in sample_index]
        #
        # batch_memory = np.append(batch_memory, batch_memory_expert, axis=0)
        #
        # # choose good memory
        # if memory_good != 0:
        #     if self.good_memory_counter > self.good_memory_size:
        #         sample_index = np.random.choice(self.good_memory_size, size=memory_good, replace=False)
        #     else:
        #         sample_index = np.random.choice(self.good_memory_counter, size=memory_good, replace=False)
        #     batch_memory_good = [self.good_memoryList[i] for i in sample_index]
        #
        #     batch_memory = np.append(batch_memory, batch_memory_good, axis=0)

        return batch_memory

    def learn(self):
        # 经过一定的步数来做参数替换
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.model_target.set_weights(self.model_eval.get_weights())
            # print('\ntarget_params_replaced\n')

        batch_memory = self.choose_memory()

        batch_s = np.array([i[0] for i in batch_memory])
        batch_a = np.array([i[1] for i in batch_memory], dtype=int)
        batch_r = np.array([i[2] for i in batch_memory])
        batch_s_ = np.array([i[3] for i in batch_memory])

        # 这里需要得到估计值加上奖励 成为训练中损失函数的期望值
        # q_next是目标神经网络的q值，q_eval是估计神经网络的q值
        # q_next是用现在状态得到的q值 q_eval是用这一步之前状态得到的q值

        q_next = self.model_target.predict(batch_s_, batch_size=self.batch_size)
        q_eval = self.model_eval.predict(batch_s, batch_size=self.batch_size)

        # calculate the two net error
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_a
        reward = batch_r

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        self.cost = self.model_eval.train_on_batch(batch_s, q_target)

        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        cost_plt = []
        for i in np.arange(0, len(self.cost_his), 3):  # every 3 steps to plot
            cost_plt.append(self.cost_his[i])
        plt.plot(np.arange(len(cost_plt)), cost_plt)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def save_net(self):
        self.model_eval.save(self.net_path+'/'+self.net_name+'_eval_net.h5')
        self.model_target.save(self.net_path+'/'+self.net_name+'_target_net.h5')
        print('save Net successful ^_^')

    def save_net_BC(self):
        self.model_eval.save(self.net_path + '/' + self.net_name + '_eval_net.h5')
        self.model_eval.save(self.net_path + '/' + self.net_name + '_target_net.h5')
        print('save Net successful ^_^')

    def pre_train_BC(self):
        if batch_con:
            # 随机取出记忆(有序)
            sample_index = np.random.randint(0, self.memory_size-self.batch_size)
            batch_memory = self.memoryList[sample_index:sample_index+self.batch_size]
        else:
            # 随机取出记忆(无序,有重复)
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
            batch_memory = [self.memoryList[i] for i in sample_index]

        batch_s = np.array([i[0] for i in batch_memory])
        batch_a = np.array([i[1] for i in batch_memory], dtype=int)

        # one-hot encoding
        label = np_utils.to_categorical(batch_a, num_classes=self.n_actions)

        self.cost = self.model_eval.train_on_batch(batch_s, label)

        self.cost_his.append(self.cost)

    def pre_train(self):
        # 经过一定的步数来做参数替换
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.model_target.set_weights(self.model_eval.get_weights())

        if batch_con:
            # 随机取出记忆(有序)
            sample_index = np.random.randint(0, self.memory_size - self.batch_size)
            batch_memory = self.memoryList[sample_index:sample_index + self.batch_size]
        else:
            # 随机取出记忆(无序,有重复)
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
            batch_memory = [self.memoryList[i] for i in sample_index]

        batch_s = np.array([i[0] for i in batch_memory])
        batch_a = np.array([i[1] for i in batch_memory], dtype=int)
        batch_r = np.array([i[2] for i in batch_memory])
        batch_s_ = np.array([i[3] for i in batch_memory])

        # 这里需要得到估计值加上奖励 成为训练中损失函数的期望值
        # q_next是目标神经网络的q值，q_eval是估计神经网络的q值
        # q_next是用现在状态得到的q值 q_eval是用这一步之前状态得到的q值

        q_next = self.model_target.predict(batch_s_, batch_size=self.batch_size)
        q_eval = self.model_eval.predict(batch_s, batch_size=self.batch_size)

        # calculate the two net error
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_a
        reward = batch_r

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        self.cost = self.model_eval.train_on_batch(batch_s, q_target)

        self.cost_his.append(self.cost)

        self.learn_step_counter += 1

    def pre_train_test(self):

        # 选取一部分的训练集数据，计算准确率
        sample_index = np.random.choice(self.memory_size, size=int(self.memory_size/50), replace=False)
        batch_memory = [self.memoryList[i] for i in sample_index]

        batch_s = np.array([i[0] for i in batch_memory])
        batch_a = np.array([i[1] for i in batch_memory], dtype=int)

        error = 0
        for i in range(len(sample_index)):
            state = np.expand_dims(batch_s[i], axis=0)
            predict = np.argmax(self.model_eval.predict(state))
            if batch_a[i] != predict:
                error += 1
        accuracy = 1. - (error / len(sample_index))

        return accuracy

    def early_stop(self, score, set_score):
        label = False
        # 先进先出
        if len(self.score_memory) > 8:
            self.score_memory.insert(0, score)
            del self.score_memory[-1]
            # 连续10次平均分大于设定分数
            if np.mean(self.score_memory) > (set_score-30):
                label = True
        else:
            self.score_memory.insert(0, score)

        return label




