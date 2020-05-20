from cv2 import cvtColor, COLOR_BGR2GRAY
import numpy as np
from Car_env import STATE_W, STATE_H
from Run_DQN import Steer_index, Throttle_index, brake_index


def detector(pho):
    label= False
    p = [[int(STATE_H*0.7-2), int(STATE_W/2)],
         [int(STATE_H*0.7-2), int(STATE_W/2-2)],
         [int(STATE_H*0.7-2), int(STATE_W/2)+1]]

    for i in p:
        if pho[i[0]][i[1]][1] > 200:
            label = True
            break
    # for i in p: pho[i[0]][i[1]] = [255, 255, 255]
    # plt.imshow(pho)
    # plt.savefig('F:/20.jpg')
    return label


def cut_state(pho):
    return pho[:-12]  # cut the button black line


def processing(s):

    pho = cvtColor(s, COLOR_BGR2GRAY)
    # pho = cut_state(pho)
    pho = pho[:, :, np.newaxis]  # (84,96) -> (84,96,1)
    return pho

# action[ , , ] -> number0~8
convert_dict = {'0': [-Steer_index,0.,0.],
                '1': [-Steer_index,0.,brake_index],
                '2': [-Steer_index,Throttle_index,0.],
                '3': [Steer_index,0.,0.],
                '4': [Steer_index,0.,brake_index],
                '5': [Steer_index,Throttle_index,0.],
                '6': [0.,0.,brake_index],
                '7': [0.,Throttle_index,0.],
                '8': [0.,0.,0.]}


def a2n(action):
    for key, value in convert_dict.items():
        if value == action:
            return int(key)
    return int(8)


def n2a(n):
    action = convert_dict[str(n)]
    return action


def plot(data):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(data)), data)
    plt.ylabel('reward')
    plt.xlabel('episodes')
    plt.show()

def plot_save(data,name):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(data)), data)
    plt.ylabel('reward')
    plt.xlabel('episodes')
    plt.savefig(name)
    plt.close()

import pickle

def save_data(file_add, data):
    with open(file_add, 'wb') as f:
        pickle.dump(data, f)
    print('Save ok!')

def load_data(file_add):
    with open(file_add,'rb') as f:
        data = pickle.load(f)
    return data

def action_counter(data):
    from collections import Counter
    action = np.array([i[1] for i in data], dtype=int)
    count = Counter(action)
    print(count)
    # action_total = 0
    # for i in range(len(data)):
    #     if data[i][1] != 8.:
    #         count += 1
    # print("total:",len(data),"action:",count,"ratio:%.1f" % float((count/len(data))*100),'%')

def speed_label(speed, set_speed, var=5):
    # 速度达标亮灯 为达标不亮
    # if speed >= set_speed:
    #     return 1.
    # else:
    #     return 0.

    # 速度指示函数，在set_speed附近变化明显
    # output: 0.~ 1.

    if speed >= set_speed:
        return 1.
    elif (set_speed - var) <= speed < set_speed:
        return ((speed-(set_speed-var))/float(var))*0.7+0.3
    else:
        return (speed/(set_speed-var))*0.3

def topspeed_label(speed, set_speed):

    if speed >= set_speed:
        return 1.
    else:
        return 0.




