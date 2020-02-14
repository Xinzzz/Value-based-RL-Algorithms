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
from common.network import NoisyLinearNetwork
from common.network import NoisyLinearDuelingNetwork
from common.arguments import get_args
from tqdm import tqdm

config = get_args()

class DQNAgent(object):

    def __init__(
        self, 
        env: gym.Env, 
        config, 
        double: bool = False, 
        PER: bool = False, 
        dueling: bool = False,
        noisy: bool = False,
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
        '''Components parameters'''
        self.PER = PER
        self.double = double
        self.dueling = dueling
        self.noisy = noisy
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
        if self.PER:
            print("PER")
        print("DQN")


        # networks: dqn, dqn_target
        if not self.dueling and not self.noisy:
            self.dqn = LinearNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.dqn_target = LinearNetwork(self.obs_dim, self.action_dim).to(self.device)

        elif self.dueling and not self.noisy:
            print("Dueling")
            self.dqn = LinearDuelingNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.dqn_target = LinearDuelingNetwork(self.obs_dim, self.action_dim).to(self.device)

        elif not self.dueling and self.noisy:
            print("Noisy")
            self.dqn = NoisyLinearNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.dqn_target = NoisyLinearNetwork(self.obs_dim, self.action_dim).to(self.device)

        elif self.dueling and self.noisy:
            print("Dueling")
            print("Noisy")
            self.dqn = NoisyLinearDuelingNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.dqn_target = NoisyLinearDuelingNetwork(self.obs_dim, self.action_dim).to(self.device)


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
        if self.epsilon > np.random.random() and not self.noisy:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            # detach data from tensor
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

        # PER: update priorities
        if self.PER:
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        if self.noisy:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

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

    def train(self, config):
        """Train the agent."""
        self.is_test = False
        print(self)
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in tqdm(range(1, config.num_frames + 1)):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
            
            # PER: increase beta
            if self.PER:
                fraction = min(frame_idx / config.num_frames, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
            
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
                
        self.env.close()
        return frame_idx, scores, losses, epsilons

    def test(self) -> None:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        while not done:
            self.env.render()
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()