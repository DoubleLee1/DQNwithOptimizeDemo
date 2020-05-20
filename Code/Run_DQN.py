import numpy as np
import My_tools
from pd_controller import PID_controller
from RL_brain_DQN import DQN

# record data or pre-train
record = False

# Steer Throttle Brake
Steer_index = 0.3
Throttle_index = 0.5
brake_index = 0.1
set_speed = 60. # np.random.randint(1, 30)
set_score = 840

# pre train index
n_action = 9  # [0~0.5left 0.5mid 0.5~1right]
observation_shape = [96, 96, 1]
expert_buffer = 'F:/DQfD/Data/expert_data/Ed_1_none.p'
net_path, net_name = 'F:/DQfD/Net', 'DQN_DQN'
data_cost_his = 'F:/DQfD/Data/cost_his.p'
data_reward_his = 'F:/DQfD/Data/reward_his.p'

data_vcc_his = 'F:/DQfD/Data/vcc_his.p'
video_add = 'F:/DQfD/video/1'
# expert_buffer = 'D:/DoubleLee_Use/Data/expert_data/Ed_1.p'
# net_path, net_name = 'D:/DoubleLee_Use/Net', 'Pure_DQN'
# data_cost_his = 'D:/DoubleLee_Use/Data/cost_his.p'
# data_reward_his = 'D:/DoubleLee_Use/Data/reward_his.p'
# data_vcc_his = 'D:/DoubleLee_Use/Data/vcc_his.p'
retrain = False

import Car_env

def run_Record():
    from pyglet.window import key
    env = Car_env.CarRacing()
    env.render()
    memoryList = []
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -Steer_index  # -1.0
        if k == key.RIGHT: a[0] = +Steer_index  # +1.0
        if k == key.UP:    a[1] = +Throttle_index  # +1.0
        if k == key.DOWN:  a[2] = +brake_index  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -Steer_index: a[0] = 0
        if k == key.RIGHT and a[0] == +Steer_index: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    isopen = True
    while isopen:
        state = env.reset()
        restart = False
        steps = 0

        for i in range(60):
            state_, r, done, info = env.step(a)
            state = state_
        state = My_tools.processing(state)

        while True:
            # fresh env
            isopen = env.render()

            # RL take action and get next observation and reward
            state_, reward, done, _ = env.step(a)
            action = My_tools.a2n(list(a))

            # restart = My_tools.detector(state_)  # detect if out the track
            if restart: reward = -100.  # if out the track give a punish
            state_ = My_tools.processing(state_)  # grey, cut, convert_shape

            a_n = np.array(action, dtype='float32')
            r = np.array(reward, dtype='float32')

            if steps % 2 == 0 or steps < 50:  # 10FPS for record
                memoryList.append([state, a_n, r, state_])
            else:
                pass
            # swap observation
            state = state_
            steps += 1

            # break while loop when end of this episode
            if done or restart or isopen == False:
                break
    env.close()
    if record:
        My_tools.save_data(expert_buffer, memoryList)
        print('The memory have', len(memoryList), 'steps')

def run_Record_AUTO():
    from pyglet.window import key
    env = Car_env.CarRacing()
    env.render()
    memoryList = []
    epi_r = []
    a = np.array([0.0, 0.0, 0.0])

    def get_speed():
        true_speed = np.sqrt(np.square(env.car.hull.linearVelocity[0]) + np.square(env.car.hull.linearVelocity[1]))
        return true_speed

    # Controller
    # input: picture
    # output: action

    def dis_controller(picture):
        action = np.array([0.0, 0.0, 0.0])
        x1, x2 = 0, 0
        STATE_W, STATE_H = observation_shape[0], observation_shape[1]

        for x in range(int(STATE_W / 2)):
            if picture[int(STATE_H * 0.7 - 5), int(STATE_W / 2) - x][1] > 200:
                x1 = x
                break
            else:
                x1 = int(STATE_W / 2)
        for x in range(int(STATE_W / 2)):
            if picture[int(STATE_H * 0.7 - 5), int(STATE_W / 2) + x][1] > 200:
                x2 = x
                break
            else:
                x2 = int(STATE_W / 2)
        # mark the border
        # picture[int(STATE_H * 0.7)-3][int(STATE_W / 2) - x1][1] = 255
        # picture[int(STATE_H * 0.7)-3][int(STATE_W / 2) + x2][1] = 255
        error = x2 - x1
        # Steer control
        if error < -3:
            action[0] = -Steer_index
        elif error > 3:
            action[0] = Steer_index
        else:
            action[0] = 0.
        # Speed control
        speed = get_speed()
        if (set_speed-1) <= speed < set_speed:
            # 随机踩油门
            if( np.random.random(1)[0] > 0.98):
                action[1] = Throttle_index
        elif speed < (set_speed-1):
            action[1] = Throttle_index
        elif speed > set_speed:
            action[2] = brake_index

        return action

    # pd_steer = PID_controller(p=0.04, d=0.0001, i=0.001)
    # pd_speed = PID_controller(p=0.05, d=0, i=0.05)
    # def con_controller(picture):
    #     action = np.array([0.0, 0.0, 0.0])  # steer[-1,1] throttle[0,1] brake[0,1]
    #
    #     # get the error
    #     x1, x2 = 0, 0
    #     STATE_W, STATE_H = observation_shape[0], observation_shape[1]
    #     for x in range(int(STATE_W / 2)):
    #         if picture[int(STATE_H * 0.7 - 5), int(STATE_W / 2) - x][1] > 200:
    #             x1 = x
    #             break
    #         else:
    #             x1 = int(STATE_W / 2)
    #     for x in range(int(STATE_W / 2)):
    #         if picture[int(STATE_H * 0.7 - 5), int(STATE_W / 2) + x][1] > 200:
    #             x2 = x
    #             break
    #         else:
    #             x2 = int(STATE_W / 2)
    #     # mark the border
    #     # picture[int(STATE_H * 0.7)-3][int(STATE_W / 2) - x1][1] = 255
    #     # picture[int(STATE_H * 0.7)-3][int(STATE_W / 2) + x2][1] = 255
    #     error = x2 - x1
    #     if abs(error)<2: error = 0
    #
    #     # Steer control
    #     steer = pd_steer.update(error)
    #     action[0] = steer
    #
    #     # Speed control
    #     speed = get_speed()
    #     value = pd_speed.update(set_speed-speed)
    #     if value > 0:
    #         action[1] = value
    #     else:
    #         action[2] = value
    #
    #     return action

    isopen = True
    memory_size = 0

    while isopen:
        state = env.reset()
        restart = False
        steps = 0
        total_reward = 0

        for i in range(60):
            state_, r, done, info = env.step(a)
            state = state_
        state = My_tools.processing(state)

        while True:
            # fresh env
            # isopen = env.render()

            # take action and get next observation and reward
            state_, reward, done, _ = env.step(a)
            action = My_tools.a2n(list(a))
            a = dis_controller(state_)

            restart = My_tools.detector(state_)  # detect if out the track
            if restart: reward = -100.  # if out the track give a punish
            state_ = My_tools.processing(state_)  # grey, cut, convert_shape

            a_n = np.array(action, dtype='float32')
            r = np.array(reward, dtype='float32')

            #优化数据
            if steps % 4 == 0 or restart:  # speed:40 20Fps  2m/fps
                if a_n != 8 or np.random.random(1)[0] > 0.9 or restart:  # 对空动作，随机存储
                    memoryList.append([state, a_n, r, state_])
                    memory_size += 1

            # if steps % 4 == 0:  # speed:40 20Fps  2m/fps
            #     memoryList.append([state, a_n, r, state_])
            #     memory_size += 1
            # else:
            #     pass

            # 加强加速记忆
            if steps < 50:
                memoryList.append([state, a_n, r, state_])
                memory_size += 1

            # swap observation
            state = state_
            steps += 1
            total_reward += reward

            # break while loop when end of this episode
            if done or restart or isopen == False:
                break

        print("set_speed:", set_speed, "total_reward:%.2f" % total_reward, "size:",memory_size)
        epi_r.append(total_reward)

        if memory_size > 10000:
            isopen = False

    env.close()
    if record:
        My_tools.save_data(expert_buffer, memoryList)
        print('The memory have', len(memoryList), 'steps')
        print('total_epi', len(epi_r), 'mean:%.2f' % np.mean(epi_r), 'std:%.2f ' % np.std(epi_r))

    My_tools.save_data(data_reward_his,epi_r)
    print('total_epi', len(epi_r), 'mean:%.2f' % np.mean(epi_r), 'std:%.2f ' % np.std(epi_r))

def run_Pretrain_BC(epi):

    steps_p_e = 20  # how many learns steps per episode
    env = Car_env.CarRacing()
    env.render()
    expert = My_tools.load_data(expert_buffer)
    print('The memory have', len(expert), 'steps')
    RL = DQN(n_action, observation_shape, retrain=retrain, net_path=net_path, net_name=net_name,
             memory_size=len(expert))
    # Load the expert experience to memory
    RL.memoryList = expert
    steps = 0

    if retrain: r_his = My_tools.load_data(data_reward_his)
    else: r_his =[]
    best = 0.
    vali_his = []

    for episode in range(epi):
        # initial observation
        state = env.reset()
        total_reward = 0.0

        # learn from demonstration   pre-train
        RL.dropout = 0.5
        for i in range(steps_p_e):
            RL.pre_train_BC()
            steps += 1

        # Show the effect
        RL.dropout = 1.
        RL.epsilon = 1.
        # Show the effect - train acc
        accuracy = RL.pre_train_test()
        print('epi:', episode, ' acc:%.3f' % accuracy)
        r_his.append(accuracy)

        # Show the effect - validation acc
        if( episode % 5 == 0):
            for i in range(60):
                state_, r, done, info = env.step(np.array([0.0, 0.0, 0.0]))
                state = state_
            state = My_tools.processing(state)

            while True:
                # fresh env
                # env.render()

                # RL choose action based on observation
                action_n = RL.choose_action(state)

                # RL take action and get next observation and reward
                action = My_tools.n2a(action_n)
                state_, reward, done, _ = env.step(action)

                restart = My_tools.detector(state_)  # detect if out the track
                if restart: reward = -100.  # if out the track give a punish
                state_ = My_tools.processing(state_)     # grey, cut, convert_shape

                # swap observation
                state = state_

                # break while loop when end of this episode
                if done or restart or total_reward < -50.:
                    break
                total_reward += reward

            vali_his.append(total_reward)
            print('epi:', episode, 'Validation_r:%.3f' % total_reward)
            if total_reward > best or total_reward > 500:
                RL.save_net_BC()
                best = total_reward

            # early_stop
            if RL.early_stop(total_reward, set_score) or steps > 20000:
                break

    # end of game
    env.close()
    print('game over! total steps:', steps)
    RL.plot_cost()
    # My_tools.plot(r_his)
    # My_tools.plot(vali_his)
    # store the important data
    My_tools.save_data(data_cost_his, r_his)   # acc
    My_tools.save_data(data_reward_his, vali_his)  # vali_r

def run_Pretrain(epi):

    steps_p_e = 20  # how many learns steps per episode
    env = Car_env.CarRacing()
    env.render()
    expert = My_tools.load_data(expert_buffer)
    print('The memory have', len(expert), 'steps')
    RL = DQN(n_action, observation_shape, retrain=retrain, net_path=net_path, net_name=net_name,
             memory_size=len(expert))
    # Load the expert experience to memory
    RL.memoryList = expert
    steps = 0

    if retrain:
        r_his = My_tools.load_data(data_reward_his)
        vcc_his = My_tools.load_data(data_vcc_his)
    else:
        r_his, vcc_his =[], []
    best = 0.

    for episode in range(epi):
        # initial observation
        state = env.reset()
        total_reward = 0.0

        # learn from demonstration   pre-train
        RL.dropout = 0.5
        for i in range(steps_p_e):
            RL.pre_train()
            steps += 1

        # Show the effect
        RL.dropout = 1.
        accuracy = RL.pre_train_test()
        print('epi:', episode, ' acc:%.3f' % accuracy)
        vcc_his.append(accuracy)

        # Show the effect - validation acc
        RL.epsilon = 1.
        if (episode % 5 == 0):
            for i in range(60):
                state_, r, done, info = env.step(np.array([0.0, 0.0, 0.0]))
                state = state_
            state = My_tools.processing(state)

            while True:
                # fresh env
                # env.render()

                # RL choose action based on observation
                action_n = RL.choose_action(state)

                # RL take action and get next observation and reward
                action = My_tools.n2a(action_n)
                state_, reward, done, _ = env.step(action)

                restart = My_tools.detector(state_)  # detect if out the track
                if restart: reward = -100.  # if out the track give a punish
                state_ = My_tools.processing(state_)  # grey, cut, convert_shape

                # swap observation
                state = state_

                # break while loop when end of this episode
                if done or restart or total_reward < -50.:
                    break
                total_reward += reward

            r_his.append(total_reward)
            print('epi:', episode, 'Validation_r:%.3f' % total_reward, "lr_steps:", steps)
            if total_reward > best or total_reward > set_score:
                RL.save_net()
                best = total_reward

            # early_stop
            if RL.early_stop(total_reward, set_score) or steps > 20000:
                break

    # end of game
    env.close()
    RL.save_net()
    print('game over! total steps:', steps)
    RL.plot_cost()
    My_tools.plot_save(r_his, 'you_'+'r_his'+'s'+str(int(set_score)))
    My_tools.plot_save(vcc_his, 'you_'+'vcc_his'+'s'+str(int(set_score)))
    # store the important data
    My_tools.save_data(data_vcc_his, vcc_his)
    My_tools.save_data(data_reward_his, r_his)

def run_train_fl(epi):  # full learn

    env = Car_env.CarRacing()
    env.render()
    RL = DQN(n_action, observation_shape, retrain=False, net_path=net_path, net_name=net_name)
    # RL.expert_memory = My_tools.load_data(expert_buffer)
    steps = 0

    if retrain: r_his, RL.cost_his = My_tools.load_data(data_reward_his), My_tools.load_data(data_cost_his)
    else: r_his =[]
    best = 500.
    good = set_score
    steps_p_e = 20

    for episode in range(epi):
        # initial observation
        state = env.reset()
        total_reward = 0.0
        epi_memory = []

        for i in range(60):
            state_, r, done, info = env.step(np.array([0.0, 0.0, 0.0]))
            state = state_
        state = My_tools.processing(state)

        while True:
            # fresh env
            # env.render()

            # RL choose action based on observation
            action_n = RL.choose_action(state)

            # RL take action and get next observation and reward
            action = My_tools.n2a(action_n)
            state_, reward, done, _ = env.step(action)
            action_n = My_tools.a2n(action)

            restart = My_tools.detector(state_)  # detect if out the track
            if restart: reward = -100.  # if out the track give a punish
            state_ = My_tools.processing(state_)     # grey, cut, convert_shape

            RL.store_transition(state, action_n, reward, state_)
            epi_memory.append([state, action_n, reward, state_])

            # if memory full, off-learning
            if (RL.memory_counter > RL.memory_size):
                print("learning...")
                # learn from full memory
                RL.dropout = 0.5
                for k in range(100):  # 100*steps_p_e = 2000
                    for i in range(steps_p_e):
                        RL.pre_train()
                    # Show the effect
                    RL.dropout = 1.
                    accuracy = RL.pre_train_test()
                    print('lr_epi:', k, ' acc:%.3f' % accuracy)
                RL.memory_counter = 0
                RL.memoryList = []

            # swap observation
            state = state_
            steps += 1

            # break while loop when end of this episode
            if done or restart:
                break
            total_reward += reward

        # store epi bad memory
        # lenth = len(epi_memory)
        # for i in range(len(epi_memory)):
        #     RL.store_good_transition(epi_memory[i][0], epi_memory[i][1], epi_memory[i][2], epi_memory[i][3])

        # judge and store the good epi memory
        # if total_reward > good*0.985:
        #     good = total_reward
        #     for i in range(len(epi_memory)):
        #         RL.store_good_transition(epi_memory[i][0], epi_memory[i][1], epi_memory[i][2], epi_memory[i][3])
        #     print("save good")

        print('epi:', episode, '/total_r:%.3f' % total_reward, '/learn:', RL.learn_step_counter, 'steps/ memory', RL.memory_counter)
        r_his.append(total_reward)
        if total_reward > best:
            RL.save_net()
            best = total_reward

        # early_stop
        if RL.early_stop(total_reward, set_score) or RL.learn_step_counter > 60000:
            break

    # end of game
    env.close()
    RL.save_net()
    print('game over!, steps:', steps)
    RL.plot_cost()
    My_tools.plot(r_his)
    # store the important data
    My_tools.save_data(data_cost_his, RL.cost_his)
    My_tools.save_data(data_reward_his, r_his)

def run_train(epi):

    env = Car_env.CarRacing()
    env.render()
    RL = DQN(n_action, observation_shape, retrain=True, net_path=net_path, net_name=net_name)
    RL.expert_memory = My_tools.load_data(expert_buffer)
    steps = 0

    if retrain: r_his, RL.cost_his = My_tools.load_data(data_reward_his), My_tools.load_data(data_cost_his)
    else: r_his =[]
    best = 500.
    good = 850.

    for episode in range(epi):
        # initial observation
        state = env.reset()
        total_reward = 0.0
        epi_memory = []
        epi_steps = 0

        for i in range(60):
            state_, r, done, info = env.step(np.array([0.0, 0.0, 0.0]))
            state = state_
        state = My_tools.processing(state)

        while True:
            # fresh env
            # env.render()

            # RL choose action based on observation
            action_n = RL.choose_action(state)

            # RL take action and get next observation and reward
            action = My_tools.n2a(action_n)
            state_, reward, done, _ = env.step(action)
            action_n = My_tools.a2n(action)

            restart = My_tools.detector(state_)  # detect if out the track
            if restart: reward = -100.  # if out the track give a punish
            state_ = My_tools.processing(state_)     # grey, cut, convert_shape

            if epi_steps % 4 == 0 or restart:  # speed:40 20Fps  2m/fps
                if action_n != 8 or np.random.random(1)[0] > 0.6 or restart:  # 对空动作，随机存储
                    RL.store_transition(state, action_n, reward, state_)
            # epi_memory.append([state, action_n, reward, state_])

            RL.dropout = 0.5
            if (RL.memory_counter > RL.batch_size) and (steps % 4 == 0):
                RL.learn()
            RL.dropout = 1.

            # swap observation
            state = state_
            epi_steps +=1
            steps += 1

            # break while loop when end of this episode
            if done or restart or total_reward < -50.:
                break
            total_reward += reward

        # store epi bad memory
        # lenth = len(epi_memory)
        # for i in range(len(epi_memory)):
        #     RL.store_good_transition(epi_memory[i][0], epi_memory[i][1], epi_memory[i][2], epi_memory[i][3])

        # judge and store the good epi memory
        # if total_reward > good*0.985:
        #     good = total_reward
        #     for i in range(len(epi_memory)):
        #         RL.store_good_transition(epi_memory[i][0], epi_memory[i][1], epi_memory[i][2], epi_memory[i][3])
        #     print("save good")

        print('epi:', episode, 'total_r:%.3f' % total_reward, 'learn:', RL.learn_step_counter, 'steps')
        r_his.append(total_reward)
        if total_reward > best:
            RL.save_net()
            best = total_reward

        # early_stop
        if RL.early_stop(total_reward, set_score) or RL.learn_step_counter > 60000:
            break

    # end of game
    env.close()
    RL.save_net()
    print('game over!, steps:', steps)
    RL.plot_cost()
    My_tools.plot(r_his)
    # store the important data
    My_tools.save_data(data_cost_his, RL.cost_his)
    My_tools.save_data(data_reward_his, r_his)

def run_pure_DQN(epi):

    env = Car_env.CarRacing()
    env.render()
    RL = DQN(n_action, observation_shape, retrain=retrain, net_path=net_path, net_name=net_name)
    steps = 0

    if retrain: r_his, RL.cost_his = My_tools.load_data(data_reward_his), My_tools.load_data(data_cost_his)
    else: r_his =[]

    for episode in range(epi):
        # initial observation
        state = env.reset()
        total_reward = 0.0

        for i in range(60):
            state_, r, done, info = env.step(np.array([0.0, 0.0, 0.0]))
            state = state_
        state = My_tools.processing(state)

        while True:
            # fresh env
            # env.render()

            # RL choose action based on observation
            action_n = RL.choose_action(state)

            # RL take action and get next observation and reward
            action = My_tools.n2a(action_n)
            state_, reward, done, _ = env.step(action)
            action_n = My_tools.a2n(action)

            restart = My_tools.detector(state_)  # detect if out the track
            if restart: reward = -10.  # if out the track give a punish
            state_ = My_tools.processing(state_)     # grey, cut, convert_shape

            RL.store_transition(state, action_n, reward, state_)

            if (steps > 200) and (steps % 2 == 0):
                RL.learn()

            # swap observation
            state = state_

            # break while loop when end of this episode
            if done or restart:
                break
            total_reward += reward
            steps += 1
        print('epi:', episode, 'total_r:%.3f' % total_reward, 'learn:', RL.learn_step_counter, 'steps', "e:%.4f"% RL.epsilon)
        r_his.append(total_reward)
        if episode % 10 == 0:
            # store the important data
            RL.save_net()
            My_tools.save_data(data_cost_his, RL.cost_his)
            My_tools.save_data(data_reward_his, r_his)
    # end of game
    env.close()
    print('game over!, interact steps:', steps)
    RL.plot_cost()
    My_tools.plot(r_his)

def browse_record():
    expert = My_tools.load_data(expert_buffer)
    import matplotlib.pyplot as plt
    import cv2
    print('The memory have', len(expert), 'steps')
    for i in range(0, len(expert)):
        img = expert[i][0]
        img = np.squeeze(img)
        # r = expert[i][2]
        # print(r)
        cv2.imshow('1', img)
        cv2.imwrite("F:/1/"+str(i)+'.jpg', img)
        cv2.waitKey(50)

def show_agent(epi, show=False):

    env = Car_env.CarRacing()

    env.render()
    RL = DQN(n_action, observation_shape, retrain=True, net_path=net_path, net_name=net_name)
    RL.dropout = 1.
    RL.epsilon = 1.
    # Load the expert experience to memory
    steps = 0
    r_his = []
    for episode in range(epi):
        # initial observation
        state = env.reset()
        total_reward = 0.0

        # Show the effect
        for i in range(60):
            state_, r, done, info = env.step(np.array([0.0, 0.0, 0.0]))
            state = state_
        state = My_tools.processing(state)

        while True:
            # fresh env
            if show: env.render()

            # RL choose action based on observation
            action_n = RL.choose_action(state)

            # RL take action and get next observation and reward
            action = My_tools.n2a(action_n)
            state_, reward, done, _ = env.step(action)

            restart = My_tools.detector(state_)  # detect if out the track
            state_ = My_tools.processing(state_)     # grey, cut, convert_shape

            # swap observation
            state = state_

            # break while loop when end of this episode
            if done or restart or total_reward < -50.:
                break
            total_reward += reward
            steps += 1
        print('epi:', episode, 'total_r:%.3f' % total_reward)
        r_his.append(total_reward)

    # end of game
    env.close()
    print('game over!')
    print('The mean score is: %.1f ' % np.mean(r_his), 'std is:%.2f' % np.std(r_his, ddof=1))
    # My_tools.plot(r_his)
    My_tools.save_data(data_reward_his, r_his)
    My_tools.plot_save(r_his, 's'+str(int(set_speed))+'m'+str(int(np.mean(r_his)))+'std'+str(int(np.std(r_his, ddof=1))))


if __name__ == '__main__':

    # run_Record()
    # run_Record_AUTO()
    # browse_record()
    # run_Pretrain_BC(1000)
    # run_Pretrain(10000)
    show_agent(30, show=False)
    # run_train(1000)
    # run_train_fl(1000)

    # Calculate the action ratio
    # data = My_tools.load_data(expert_buffer)
    # My_tools.action_counter(data)

    # run_pure_DQN(1000)