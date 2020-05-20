expert_buffer = 'F:/DQfD/Data/expert_data/Ed_1_you.p'
data_bs_acc = 'F:/DQfD/Data/reward_his.p'
data_reward_you = 'F:/DQfD/Data/reward_1.p'
data_reward_none = 'F:/DQfD/Data/reward_2.p'
data_reward_bc = 'F:/DQfD/Data/reward_3.p'

import matplotlib.pyplot as plt
import numpy as np
import My_tools

def plot_dqn_reward(data, name):
    plt.plot(np.arange(len(data)), data)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(str(name)+'.jpeg', bbox_inches='tight')
    plt.close()

def plot_bc_acc(data, name):
    plt.plot(np.arange(0, len(data)*20, 20), data)
    plt.ylabel('Accuracy')
    plt.xlabel('Learn-steps')
    plt.savefig(str(name)+'.jpeg', bbox_inches='tight')
    plt.close()

def plot_all_reward(data_you, data_none, data_bc,  name):
    plt.plot(np.arange(0, 30, 1), data_you, label="optimize-DQN", color='red')
    plt.plot(np.arange(0, 30, 1), data_none, label="DQN", color='orange')
    plt.plot(np.arange(0, 30, 1), data_bc, label="BehaviorClone", color='blue')
    plt.plot(np.arange(0, 30, 1), [850]*30, label="Demonstrator", color='gray', linestyle='-')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.legend(loc="center right")
    plt.savefig(str(name)+'.jpeg', bbox_inches='tight')
    plt.close()

data_you = My_tools.load_data(data_reward_you)
data_none = My_tools.load_data(data_reward_none)
# data_none = [0]*30
data_bc = My_tools.load_data(data_reward_bc)
# print(len(data_you))

plot_all_reward(data_you, data_none, data_bc, 'V30')
# data = My_tools.load_data(data_bs_acc)
# plot_bc_acc(data, 'BC_train_acc')
