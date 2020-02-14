import os
from typing import Dict, List, Tuple


import gym
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from common.plot import plot
from common.train_test import train, test
from common.arguments import get_args
from common.data import save_model
from common.data import load_data
from common.data import load_model
from algos.DQN import DQNAgent


# ------------ environment ------------ 
env_id = "CartPole-v0"
env = gym.make(env_id)
config = get_args()

seed = config.seed

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)
env.seed(seed)

# ------------ train ------------ 
data_to_store = {}
training = False
if training:
    dqn_agent = DQNAgent(env, config, double=True, dueling=True, PER=True, opt='Adam', noisy=True)
    frame_dqn, score_dqn, loss_dqn, eps_dqn = dqn_agent.train(config)
    data_to_store['dqn'] = (frame_dqn, score_dqn, loss_dqn, eps_dqn)
    save_model('dqn', data_to_store['dqn'])
    save_model('dqn_model', dqn_agent)

    # dqn_agent = DQNAgent(env, config, double=True, dueling=True, PER=True, opt='Adam', noisy=True)
    # frame_dqn, score_dqn, loss_dqn, eps_dqn = train(dqn_agent, env, config.num_frames)
    # data_to_store['dqn noisy'] = (frame_dqn, score_dqn, loss_dqn, eps_dqn)
    # save_model('dqn noisy', data_to_store['dqn noisy'])

# ------------ load model ------------ 
loading = True
loaded_data = []
if loading:
    loaded_data.append(load_data('dqn', 'darkgrey'))
    #loaded_data.append(load_model('dqn noisy', 'salmon'))
    #loaded_data.append(load_model('dqn per', 'orange'))
    #loaded_data.append(load_model('double dqn per', 'teal'))
    plot(loaded_data)

testing = False
if testing:
    agent = load_model('dqn_model')
    for i in range(10):
        agent.test()
