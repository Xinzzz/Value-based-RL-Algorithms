'''
To handle the file dir problem
'''

import numpy as np
import torch
import os

from common.plot import plot

save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)+'/'+'..'), 'save')

model_dir = os.path.join(save_dir, 'model')
video_dir = os.path.join(save_dir, 'video')
log_dir = os.path.join(save_dir, 'log')
ckpt_dir = os.path.join(save_dir, 'checkpoint')


def print_green(skk):
    print("\033[92m {}\033[00m" .format(skk))

def print_yellow(skk):
    print("\033[93m {}\033[00m" .format(skk))

def print_cyan(skk):
    print("\033[96m {}\033[00m" .format(skk))
    
def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_model(model, model_name):
    filename = model_name + '.pth'
    filedir = os.path.join(model_dir, filename)
    torch.save(model, filedir)

def load_model(model_name):
    filename = model_name + '.pth'
    filedir = os.path.join(model_dir, filename)
    loaded_model = torch.load(filedir)
    return loaded_model

def save_data(model_name, episode, steps, reward, loss, eps):
    filename = model_name + '_plot.pth'
    filedir = os.path.join(log_dir, filename)
    data = (episode, steps, reward, loss, eps)
    torch.save(data, filedir)

def load_data(model_name):
    filename = model_name + '_plot.pth'
    filedir = os.path.join(log_dir, filename)
    loaded_data = torch.load(filedir)
    plot(loaded_data)

def save_ckpt(model, model_name, episode, optimizer, rewards, mean_reward, losses, eps, step, replay_memory):
    filename = model_name + str(step) + '_ckpt.tar'
    filedir = os.path.join(ckpt_dir, filename)
    state = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optim_state_dict':optimizer.state_dict(),
        'total_reward': rewards,
        'mean_reward': mean_reward,
        'total_loss':losses,
        'total_eps':eps,
        'numsteps':step,
        'replay': replay_memory
    }
    torch.save(state, filedir)

def load_ckpt(model_name, step):
    filename = model_name + str(step) + '_ckpt.tar'
    filedir = os.path.join(ckpt_dir, filename)
    checkpoint = torch.load(filedir)
    return checkpoint


def log_show(step, episode, total_episodes, reward, loss, eps, memory_size):
    print_cyan('-----------------------------------------------')
    print('timestep: ', step)
    print('episode: ', episode, '/', total_episodes)
    print('reward: ', reward)
    print('loss: ', loss)
    print('epsilon: ', eps)
    print('memory: ', memory_size)
    print("")

