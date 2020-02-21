import matplotlib.pyplot as plt
import numpy as np

def plot(data):
    plt.style.use('ggplot')
    plt.figure(figsize=(4,3))

    #plt.subplot(221)
    plt.title('score')
    plt.plot(data[2])
    plt.xlabel('Episode')
    plt.ylabel('score')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

mean_reward = []

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