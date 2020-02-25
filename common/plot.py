import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def plot(data):
    plt.figure(figsize=(14,4))

    for i, data in enumerate(data):
        plt.subplot(131)
        plt.title('score')
        plt.plot(data[0][3], label=data[2], color=data[1])
        plt.xlabel('Episode')
        plt.ylabel('score')
        plt.legend(loc='best', facecolor='none', edgecolor='none')

        plt.subplot(132)
        plt.title('平均奖赏值')
        mean_reward, std = get_mean_std(data[0][1], data[0][3])
        plt.plot(mean_reward, label=data[2], color=data[1])

        
       # plt.fill_between(range(len(mean_reward)), mean_reward - std, mean_reward + std, alpha=0.5, edgecolor='none', facecolor=data[1])
        plt.xlabel('Episode')
        plt.ylabel('平均奖赏值')
        plt.legend(loc='best', facecolor='none', edgecolor='none')

        plt.subplot(133)
        plt.title('loss')
        plt.plot(data[0][4], label=data[2], color=data[1])
        plt.xlabel('timesteps')
        plt.ylabel('loss')
        plt.legend(loc='best', facecolor='none', edgecolor='none')



    plt.tight_layout()
    plt.savefig("save/log/pic.png", format='png', bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.show()

def plot_anim(score, mean_score, loss, eps):
    plt.figure(2, figsize=(12,3))
    plt.clf()
    plt.subplot(131)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('score')
    plt.plot(score)
    plt.plot(mean_score)
        
    plt.subplot(132)
    plt.xlabel('timestep')
    plt.ylabel('loss')
    plt.plot(loss)

    plt.subplot(133)
    plt.xlabel('Episode')
    plt.ylabel('epsilon')
    plt.plot(eps)

    plt.pause(0.001)
    plt.tight_layout()

def get_mean_std(episodes, scores):
    mean_rewards = np.zeros(episodes)
    std = np.zeros(episodes)
    for i in range(episodes):
        mean_rewards[i] = np.mean(scores[max(0, i-30):(i+1)])
        std[i] = (scores[i] - mean_rewards[i]) / 5.0
    return mean_rewards, std