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
import time

from common.memory import *
from common.network import *
from common.plot import *
from common.wrappers import *
from common.logger import *


class DQNAgent:
    def __init__(self, env, params, hidden_dim, hidden_dim_2, double=False, dueling=False, noisy=False, mod=False, target=False):
        self.env = env
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        self.episodes = params['episode']
        self.env_name = params['run_name']
        self.model_name = '_dqn'

        self.replay_size = params['replay_size']
        self.replay_warmup = params['replay_warmup']
        self.batch_size = params['batch_size']
        self.replay_memory = ReplayBuffer(self.replay_size)

        self.gamma = params['gamma']

        self.epsilon_start = params['epsilon_start']
        self.eps = self.epsilon_start
        self.epsilon_final = params['epsilon_final']
        self.epsilon_decay = params['epsilon_decay']
        self.decay_step = 0
        self.step_count = 0
        self.loss = 0
        self.starting_learning = False
        self.learn_step_count = 0
        self.is_test = False
        self.start_making_noise = False
        '''
        DQN extension 
        '''
        self.double = double
        if self.double:
            self.model_name += '_double'

        self.dueling = dueling
        if self.dueling:
            self.model_name += '_dueling' 

        self.noisy = noisy
        if self.noisy:
            self.model_name += '_noisy'

        self.mod = mod
        if self.mod:
            self.model_name += '_mod'

        self.target = target
        if self.target:
            self.model_name += '_target'

        self.model_name += '_hid_' + str(hidden_dim) + '_' + str(hidden_dim_2)

        self.target_update_freq = params['target_update_freq']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if(len(self.env.observation_space.shape) >= 3):
            # atari
            self.dqn = cnn_DQN(self.state_dim, self.action_dim).to(self.device)
            self.dqn_target = cnn_DQN(self.state_dim, self.action_dim).to(self.device)
        else:
            if not self.dueling:
                self.dqn = nn_DQN(self.state_dim, self.action_dim, noisy=self.noisy, 
                hidden_dim=hidden_dim, hidden_dim_2=hidden_dim_2).to(self.device)
                self.dqn_target = nn_DQN(self.state_dim, self.action_dim, noisy=self.noisy, 
                hidden_dim=hidden_dim, hidden_dim_2=hidden_dim_2).to(self.device)
            elif self.dueling:
                self.dqn = nn_Dueling(self.state_dim, self.action_dim, noisy=self.noisy, 
                hidden_dim=hidden_dim, hidden_dim_2=hidden_dim_2).to(self.device)
                self.dqn_target = nn_Dueling(self.state_dim, self.action_dim, noisy=self.noisy, 
                hidden_dim=hidden_dim, hidden_dim_2=hidden_dim_2).to(self.device)

        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval() # for BN and Dropout

        self.learning_rate = params['learning_rate']
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.loss_func = nn.SmoothL1Loss().to(self.device)

    def select_action(self, state) -> torch.Tensor:
        sample = random.random()
        if self.starting_learning:
            if not self.noisy and not self.mod:
                self.eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                    math.exp(-1. * self.decay_step / self.epsilon_decay)
            elif self.noisy and not self.mod:
                self.eps = 0
            elif self.noisy and self.mod:
                if not self.start_making_noise:
                    self.eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                        math.exp(-1. * self.decay_step / self.epsilon_decay)
                    if(self.decay_step / self.epsilon_decay >=1.8): #1.8
                        self.start_making_noise = True
                elif self.start_making_noise:
                    self.eps = 0    

            self.decay_step += 1
            
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
            # print("")
            # print_green("update target network...")
        self.learn_step_count += 1
        # return a dict
        experiences = self.replay_memory.sample(self.batch_size)
        loss = self._compute_loss(experiences)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.noisy:
            if not self.mod:
                self.dqn.reset_noise()
                self.dqn_target.reset_noise()
            elif self.mod:
                if self.start_making_noise:
                    self.dqn.reset_noise()
                    self.dqn_target.reset_noise()

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
        if not self.double:
            next_qval = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        elif self.double:
            next_qval = self.dqn_target(next_state).gather(1, self.dqn(next_state).argmax(dim=1, keepdim=True)).detach()

        mask = 1 - done

        target = (reward + next_qval * self.gamma * mask).to(device)

        loss = self.loss_func(cur_qval, target)

        return loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def load_saved_model(self):
        self.dqn.load_state_dict(load_model(self.env_name + self.model_name))

    def train(self):
        start_time = time.time()
        loading = False
        saving = True
        self.is_test = False
        state = self.env.reset()
        episodes_reward = []
        mean_reward = []
        losses = []
        eps = []
        rewards = 0
        cur_episode = 1
        plt.ion()
        if loading:
            loading = True
            print("")
            print_yellow('loading checkpoint..')
            loaded_checkpoint = load_ckpt(self.env_name + self.model_name, 100000)
            self.dqn.load_state_dict(loaded_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(loaded_checkpoint['optim_state_dict'])
            cur_episode = loaded_checkpoint['episode']
            episodes_reward = loaded_checkpoint['total_reward']
            mean_reward = loaded_checkpoint['mean_reward']
            losses = loaded_checkpoint['total_loss']
            eps = loaded_checkpoint['total_eps']
            self.eps = loaded_checkpoint['cur_eps']
            self.step_count = loaded_checkpoint['numsteps']
            self.replay_memory = loaded_checkpoint['replay']
            self.decay_step = self.step_count - self.replay_warmup
            self.dqn.train()

        for episode in range(cur_episode, self.episodes + 1):
            while True:
                action = self.select_action(state).item()

                next_state, reward, done, _ = self.env.step(action)
                self.env.render()

                if not self.is_test:
                    self.replay_memory.store(state, action, reward, next_state, done)
                    
                state = next_state
                rewards += reward
                if saving:
                    if self.step_count % 10000 == 0:
                        print("")
                        print_yellow("saving checkpoint..")
                        save_ckpt(self.dqn, self.env_name + self.model_name, episode, 
                        self.optimizer, episodes_reward, mean_reward, 
                        losses, eps, self.eps, self.step_count, self.replay_memory)
                # if episode ends
                if done:
                    state = self.env.reset()
                    episodes_reward.append(rewards)
                    log_show(self.step_count, episode, self.episodes, rewards, self.loss, self.eps, len(self.replay_memory)) 
                    rewards = 0    
                    mean = np.mean(episodes_reward[max(0, len(episodes_reward)-100):(len(episodes_reward)+1)])
                    mean_reward.append(mean)
                    #plot_anim(episodes_reward, mean_reward, losses, eps)     
                    break

                # its about to train
                if len(self.replay_memory) >= self.replay_warmup:
                    if not self.starting_learning:
                        print("")
                        print_green("Start Training...")
                        self.starting_learning = True
                    self.loss = self.update_model()
                    losses.append(self.loss)
                    eps.append(self.eps)


        self.env.close()
        plt.show()
        plt.ioff()
        end_time = time.time()
        print_cyan(f'Training cost {round(((end_time - start_time) / 60), 2)} mins')
        save_data(self.env_name + self.model_name, self.model_name, episode, self.step_count, episodes_reward, losses, eps)
        save_model(self.dqn, self.env_name + self.model_name)

    def test(self) -> None:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        while not done:
            self.env.render()
            action = self.select_action(state).item()
            next_state, reward, done, _ = self.env.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()


                






