import matplotlib.pyplot as plt

import os
import numpy as np
from typing import Dict, List, Tuple

def plot(data):
    """Plot the training progresses."""
    plt.style.use('ggplot')
    plt.figure(figsize=(8,7))
    # get the min episode number over all algorithms
    min_episodes = get_min_episodes(data)
        
    # actually plot the data
    for i, d in enumerate(data):
        plt.subplot(221)
        plt.title('scores')
        plt.plot(range(min_episodes), d[1][:min_episodes], label=d[4], color=d[5])
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.legend(loc='best')

        plt.subplot(222)
        plt.title('mean scores')
        mean_rewards = get_mean(min_episodes, d[1][:min_episodes])
        std_reward = np.std(d[1][:min_episodes], ddof=1) / 5
        plt.plot(range(min_episodes), mean_rewards, label=d[4], color=d[5])
        plt.fill_between(range(min_episodes), mean_rewards-std_reward, mean_rewards+std_reward, alpha=0.3, facecolor=d[5])
        plt.xlabel('episode')
        plt.ylabel('mean score')
        plt.legend(loc='best')
    
        plt.subplot(223)
        plt.title('loss')
        mean_loss = get_mean(d[0], d[2])
        std_loss = np.std(d[2], ddof=1)
        plt.plot(mean_loss, label=d[4], color=d[5], alpha=0.5)
        plt.fill_between(range(d[0]), mean_loss-std_loss, mean_loss+std_loss, alpha=0.3, facecolor=d[5])
        plt.xlabel('frame')
        plt.ylabel('loss')
        plt.legend(loc='best')

        plt.subplot(224)
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

def get_mean(episodes, scores):
    mean_rewards = np.zeros(episodes)
    for i in range(episodes):
        mean_rewards[i] = np.mean(scores[max(0, i-50):(i+1)])
    return mean_rewards
