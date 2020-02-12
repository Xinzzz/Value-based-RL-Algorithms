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
from common.data import load_model
from algos.DQN import DQNAgent


# ------------ environment ------------ 
env_id = "CartPole-v1"
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
    dqn_agent = DQNAgent(env, config, double=False, dueling=False, PER=False)
    frame_dqn, score_dqn, loss_dqn, eps_dqn = train(dqn_agent, env, config.num_frames)
    data_to_store['dqn'] = (frame_dqn, score_dqn, loss_dqn, eps_dqn)

    double_dqn_agent = DQNAgent(env, config, double=True, dueling=False, PER=False)
    frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn = train(double_dqn_agent, env, config.num_frames)
    data_to_store['double dqn'] = (frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn)

    dqn_per_agent = DQNAgent(env, config, double=False, dueling=False, PER=True)
    frame_dqn_per, score_dqn_per, loss_dqn_per, eps_dqn_per = train(dqn_per_agent, env, config.num_frames, PER = True)
    data_to_store['dqn per'] = (frame_dqn_per, score_dqn_per, loss_dqn_per, eps_dqn_per)

    double_dqn_per_agent = DQNAgent(env, config, double=True, dueling=False, PER=True)
    frame_double_dqn_per, score_double_dqn_per, loss_double_dqn_per, eps_double_dqn_per = train(double_dqn_per_agent, env, config.num_frames, PER = True)
    data_to_store['double dqn per'] = (frame_double_dqn_per, score_double_dqn_per, loss_double_dqn_per, eps_double_dqn_per)

    # dqn_dueling_agent = DQNAgent(env, config, dueling=True)
    # frame_dqn_dueling, score_dqn_dueling, loss_dqn_dueling, eps_dqn_dueling = train(dqn_dueling_agent, env, config.num_frames)
    # data_to_store['dqn dueling'] = (frame_dqn_dueling, score_dqn_dueling, loss_dqn_dueling, eps_dqn_dueling)

# ------------ save model ------------ 
saving = False
if saving:
    save_model('dqn', data_to_store['dqn'])
    save_model('double dqn', data_to_store['double dqn'])
    save_model('dqn per', data_to_store['dqn per'])
    save_model('double dqn per', data_to_store['double dqn per'])
    # save_model('dqn dueling', data_to_store['dqn dueling'])

# ------------ load model ------------ 
loading = True
loaded_data = []
if loading:
    loaded_data.append(load_model('dqn', 'darkgrey'))
    loaded_data.append(load_model('double dqn', 'salmon'))
    loaded_data.append(load_model('dqn per', 'orange'))
    loaded_data.append(load_model('double dqn per', 'teal'))
    plot(loaded_data)
