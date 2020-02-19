import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))
import matplotlib.pyplot as plt
import gym
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from tqdm import tqdm
from common.memory import ReplayBuffer
from common.network import *
from common.parameters import parse_argument
from common.plot import *
from common.wrappers import *

config = parse_argument()


class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        self.episodes = config.episodes

        self.memory_size = config.memory_size
        self.warmup_memory_size = config.warmup_memory_size
        self.batch_size = config.batch_size
        self.replay_memory = ReplayBuffer(self.memory_size)

        self.gamma = config.gamma

        self.eps_start = config.eps_start
        self.eps = 0
        self.eps_end = config.eps_end
        self.eps_decay = config.eps_decay 
        self.step_count = 0
        self.learn_step_count = 0

        self.target_update_freq = config.target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(self.device)
        if(len(self.env.observation_space.shape) >= 3):
            # atari
            self.dqn = cnn_DQN(self.state_dim, self.action_dim).to(self.device)
            self.dqn_target = cnn_DQN(self.state_dim, self.action_dim).to(self.device)
        else:
            self.dqn = nn_DQN(self.state_dim, self.action_dim).to(self.device)
            self.dqn_target = nn_DQN(self.state_dim, self.action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval() # for BN and Dropout

        self.optimizer = optim.Adam(self.dqn.parameters())
        self.loss_func = nn.SmoothL1Loss().to(self.device)

    def select_action(self, state) -> torch.Tensor:
        sample = random.random()
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.step_count / self.eps_decay)

        self.step_count += 1

        if sample > self.eps:
            with torch.no_grad():
                if len(self.state_dim) >= 3:
                    state = np.array(state) / 255.0
                state = np.array(state)
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                return self.dqn(state).max(1)[1]
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def update_model(self) -> torch.Tensor:
        # not enough memory to train
        if len(self.replay_memory) < self.batch_size:
            return 
        if self.learn_step_count % self.target_update_freq == 0:
            self._target_hard_update()
            print("")
            print("update target network...")
        self.learn_step_count += 1
        # return a dict
        experiences = self.replay_memory.sample(self.batch_size)
        loss = self._compute_loss(experiences)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_loss(self, experience_samples) -> torch.Tensor:
        device = self.device

        state = torch.FloatTensor(experience_samples["state"]).to(device)
        next_state = torch.FloatTensor(experience_samples["next_state"]).to(device)
        action = torch.LongTensor(experience_samples["action"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(experience_samples["reward"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(experience_samples["done"].reshape(-1, 1)).to(device)

        if len(self.state_dim) >= 3:
            state /= 255.0
            next_state /= 255.0

        cur_qval = self.dqn(state).gather(1, action)
        next_qval = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()

        mask = 1 - done

        target = (reward + next_qval * self.gamma * mask).to(device)

        loss = self.loss_func(cur_qval, target)

        return loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def train(self):
        state = self.env.reset()
        episodes_reward = []
        losses = []
        eps = []
        rewards = 0
        learning_flag = True
        print("Start!")
        plt.ion()
        for episode in tqdm(range(1, self.episodes + 1)):

            while True:
                action = self.select_action(state).item()

                next_state, reward, done, _ = self.env.step(action)
                self.env.render()
                self.replay_memory.store(state, action, reward, next_state, done)

                state = next_state
                rewards += reward

                # if episode ends
                if done:
                    state = self.env.reset()
                    episodes_reward.append(rewards)
                    rewards = 0    
                    plot_anim(episodes_reward, losses, eps)      
                    if self.step_count % 1000 == 0:
                        print("current timestep : ", self.step_count)       
                    break

                # its about to train
                if len(self.replay_memory) >= self.warmup_memory_size:
                    if learning_flag:
                        print("")
                        print("Start Training...")
                        learning_flag = False
                    loss = self.update_model()
                    losses.append(loss)
                    eps.append(self.eps)


        self.env.close()
        plt.show()
        plt.ioff()
        return episode, self.step_count, episodes_reward, losses, eps



seed = config.seed
env = gym.make(config.env)
env = wrap_env(env)




def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_model(model, data):
    path = 'logs/' + model + '.pickle'
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

seed_torch(seed)
env.seed(seed)

data_to_store = {}
agent = DQNAgent(env, config)
episode_dqn, step_dqn, score_dqn, loss_dqn, eps_dqn = agent.train()
data_to_store['dqn'] = (episode_dqn, step_dqn, score_dqn, loss_dqn, eps_dqn)
data = data_to_store['dqn'] 
save_model('dqn', data)

plot(data)


                






