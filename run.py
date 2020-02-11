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
from algos.DQN import DQNAgent
from algos.double_DQN import DoubleDQNAgent
from algos.DQN_PER import DQNAgentPER


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
# dqn_agent = DQNAgent(env, config)
# frame_dqn, score_dqn, loss_dqn, eps_dqn = train(dqn_agent, env, config.num_frames)

# double_dqn_agent = DoubleDQNAgent(env, config)
# frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn = train(double_dqn_agent, env, config.num_frames)

# dqn_per_agent = DQNAgentPER(env, config)
# frame_dqn_per, score_dqn_per, loss_dqn_per, eps_dqn_per = train(dqn_per_agent, env, config.num_frames, PER = True)
# # ------------ save model ------------ 
# with open('logs/dqn.pickle', 'wb') as f:
#     pickle.dump((frame_dqn, score_dqn, loss_dqn, eps_dqn), f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('logs/double_dqn.pickle', 'wb') as f:
#     pickle.dump((frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn), f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('logs/dqn_per.pickle', 'wb') as f:
#     pickle.dump((frame_dqn_per, score_dqn_per, loss_dqn_per, eps_dqn_per), f, protocol=pickle.HIGHEST_PROTOCOL)

# ------------ load model ------------ 
with open('logs/dqn.pickle', 'rb') as f:
    frame_dqn, score_dqn, loss_dqn, eps_dqn = pickle.load(f)
with open('logs/double_dqn.pickle', 'rb') as f:
    frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn = pickle.load(f)
with open('logs/dqn_per.pickle', 'rb') as f:
    frame_dqn_per, score_dqn_per, loss_dqn_per, eps_dqn_per = pickle.load(f)

data = []
data.append((frame_dqn, score_dqn, loss_dqn, eps_dqn, "dqn", "darkgrey"))
# data.append((frame_double_dqn, score_double_dqn, loss_double_dqn, eps_double_dqn, "double dqn", "coral"))
data.append((frame_dqn_per, score_dqn_per, loss_dqn_per, eps_dqn_per, "dqn per", "blue"))
plot(data)
