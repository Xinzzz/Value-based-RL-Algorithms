import os
from typing import Dict, List, Tuple
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common.PER import PrioritizedReplayBuffer
from common.experience_replay import ReplayBuffer
from common.network import LinearNetwork
from common.network import LinearDuelingNetwork
from common.arguments import get_args

config = get_args()

class DQNAgent(object):

    def __init__(
        self, 
        env: gym.Env, 
        config, 
        double: bool = False, 
        PER: bool = False, 
        dueling: bool = False,
        opt: str = 'Adam',
        ):
        """Initialization"""
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n  
        self.env = env
        self.batch_size = config.batch_size
        self.epsilon = config.max_epsilon
        self.epsilon_decay = config.epsilon_decay
        self.max_epsilon = config.max_epsilon
        self.min_epsilon = config.min_epsilon
        self.target_update = config.target_update
        self.gamma = config.gamma

        self.PER = PER
        self.double = double
        self.dueling = dueling
        # PER
        # In DQN, We used "ReplayBuffer(obs_dim, memory_size, batch_size)"
        self.beta = config.beta
        self.prior_eps = config.prior_eps
        if not self.PER:
            self.memory = ReplayBuffer(self.obs_dim, config.memory_size, config.batch_size)
        else:
            self.memory = PrioritizedReplayBuffer(self.obs_dim, config.memory_size, config.batch_size, config.alpha)

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        if self.double:
            print("Double")
        if self.dueling:
            print("Dueling")
        if self.PER:
            print("PER")
        print("DQN")


        # networks: dqn, dqn_target
        if not self.dueling:
            self.dqn = LinearNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.dqn_target = LinearNetwork(self.obs_dim, self.action_dim).to(self.device)
        else:
            self.dqn = LinearDuelingNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.dqn_target = LinearDuelingNetwork(self.obs_dim, self.action_dim).to(self.device)

        self.dqn_target.load_state_dict(self.dqn.state_dict())
        # dqn_target not for training, without change in dropout and BN
        self.dqn_target.eval()
        
        # optimizer
        if opt == 'RMSprop':
            self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=2e-4, momentum=5e-2)
        elif opt == 'Adam':
            self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        # transition (list): transition information including state, action, reward, next_state, done
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            # get data from tensor
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            # *list for unwarpping parameters
            # store the transition into replay memory
            self.memory.store(*self.transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        if not self.PER:
            samples = self.memory.sample_batch()
            loss = self._compute_dqn_loss(samples)
        else:
            samples = self.memory.sample_batch(self.beta)
            weights = torch.FloatTensor(
                samples["weights"].reshape(-1, 1)
            ).to(self.device)

            indices = samples["indices"]
            # PER: importance sampling before average
            elementwise_loss = self._compute_dqn_loss(samples)
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not self.PER:
            pass
        else:
            # PER: update priorities
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indices, new_priorities)

        return loss.item()
        
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.dqn(state).gather(1, action)

        if not self.double:
            # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
            #       = r                       otherwise
            next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        else:           
            # Double DQN
            next_q_value = self.dqn_target(next_state).gather(1, self.dqn(next_state).argmax(dim=1, keepdim=True)).detach()

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        if not self.PER:
            # calculate dqn loss
            loss = F.smooth_l1_loss(curr_q_value, target)
            return loss
        else:
            # calculate element-wise dqn loss
            elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
            return elementwise_loss


    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())