import matplotlib.pyplot as plt
import numpy as np
def plot(data):
    plt.style.use('ggplot')
    plt.figure(figsize=(4,3))

    #plt.subplot(221)
    plt.title('score')
    plt.plot(range(data[0]), data[2])
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

mean_reward = []

def plot_anim(score, loss, eps):
    plt.figure(2, figsize=(12,3))
    plt.clf()
    plt.xlabel('Episode')
    plt.subplot(131)
    plt.title('Training...')
    plt.ylabel('score')
    plt.plot(score)


    mean = np.mean(score[max(0, len(score)-100):(len(score)+1)])
    mean_reward.append(mean)
    plt.plot(mean_reward)
        
    plt.subplot(132)
    plt.ylabel('loss')
    plt.plot(loss)

    plt.subplot(133)
    plt.ylabel('epsilon')
    plt.plot(eps)

    plt.pause(0.001)
    plt.tight_layout()