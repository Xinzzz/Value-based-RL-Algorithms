import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Dict, List, Tuple

def plot(data):
    """Plot the training progresses."""
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 4))
    for i, d in enumerate(data):
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (d[0], np.mean(d[1][-10:])))
        plt.plot(d[1], label=d[4], color=d[5])
        plt.xlabel('frame')
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