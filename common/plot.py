import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Dict, List, Tuple

def plot(data):
    """Plot the training progresses."""
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 4))
    # get the min episode number over all algorithms
    min_episodes = get_min_episodes(data)
        
    # actually plot the data
    for i, d in enumerate(data):
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (d[0], np.mean(d[1][-10:])))
        mean_rewards = get_mean_rewards(min_episodes, d[1][:min_episodes])
        std = np.std(d[1][:min_episodes]) / 10.0
        plt.plot(range(min_episodes), mean_rewards, label=d[4], color=d[5])
        plt.fill_between(range(min_episodes), mean_rewards-std, mean_rewards+std, alpha = 0.3, facecolor = d[5])
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.legend(loc='best')
    
        plt.subplot(132)
        plt.title('loss')
        plt.plot(d[2], label=d[4], color=d[5])
        plt.xlabel('frame')
        plt.ylabel('loss')
        plt.legend(loc='best')

        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(d[3], label=d[4], color=d[5])
        plt.xlabel('frame')
        plt.ylabel('epislon')
        plt.legend(loc='best')
        
    plt.tight_layout()
    plt.savefig("logs/pic.png", format='png', bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.show()

def get_min_episodes(data):
    min = float("inf")
    for j, c in enumerate(data):
        if(min > len(c[1])):
            min = len(c[1])
    return min

def get_mean_rewards(episodes, scores):
    mean_rewards = np.zeros(episodes)
    for i in range(episodes):
        mean_rewards[i] = np.mean(scores[max(0, i-50):(i+1)])
    return mean_rewards
