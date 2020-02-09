import os
from typing import Dict, List, Tuple
# import sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))

import gym
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from common.plot import plot
from common.train_test import train, test
from algos.DQN import DQNAgent
from algos.double_DQN import DoubleDQNAgent

# ------------ environment ------------ 
env_id = "CartPole-v0"
env = gym.make(env_id)

seed = 777

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)
env.seed(seed)

# ------------ parameters ------------ 
num_frames = 1000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 1 / 2000

# ------------ train ------------ 
# dqn_agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
# frame_dqn, score_dqn, loss_dqn, eps_dqn = train(dqn_agent, env, num_frames)

# double_dqn_agent = DoubleDQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
# frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn = train(double_dqn_agent, env, num_frames)

# ------------ save model ------------ 
# with open('logs/dqn.pickle', 'wb') as f:
#     pickle.dump((frame_dqn, score_dqn, loss_dqn, eps_dqn), f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('logs/double_dqn.pickle', 'wb') as f:
#     pickle.dump((frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn), f, protocol=pickle.HIGHEST_PROTOCOL)

# ------------ load model ------------ 
with open('logs/dqn.pickle', 'rb') as f:
    frame_dqn, score_dqn, loss_dqn, eps_dqn = pickle.load(f)
with open('logs/double_dqn.pickle', 'rb') as f:
    frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn = pickle.load(f)

data = [(frame_dqn, score_dqn, loss_dqn, eps_dqn, "dqn"),
        (frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn, "double dqn")]

plot(data)
