import numpy as np

import os
import random
from typing import Dict, List, Tuple
from common.segment_tree import *
class ReplayBuffer():
    def __init__(self, max_size):
        self.memory = []
        self.max_size = max_size
        self.cur_index = 0

    def __len__(self):
        return len(self.memory)

    def store(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.memory) < self.max_size:
            self.memory.append(experience)
        else:
            self.memory[self.cur_index] = experience

        self.cur_index = (self.cur_index + 1) % self.max_size

    def sample(self, batch_size):
        batch_buffer = np.random.choice(len(self.memory), batch_size, replace=False)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

        for index in batch_buffer:
            experience = self.memory[index]
            state, action, reward, next_state, done = experience
            state_batch.append(np.array(state, copy=False))
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(np.array(next_state, copy=False))
            done_batch.append(done)
        result = {
            'state': np.array(state_batch),
            'action': np.array(action_batch),
            'reward': np.array(reward_batch),
            'next_state': np.array(next_state_batch),
            'done': np.array(done_batch)
        }
        return result

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(self, max_size, alpha: float = 0.2):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(max_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(self, state, action, reward, next_state, done):
        """Store experience and priority."""
        super().store(state, action, reward, next_state, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, batch_size, beta: float = 0.6):
        """Sample a batch of experiences."""
        self.batch_size = batch_size
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

        for index in indices:
            experience = self.memory[index]
            state, action, reward, next_state, done = experience
            state_batch.append(np.array(state, copy=False))
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(np.array(next_state, copy=False))
            done_batch.append(done)
            weights = np.array([self._calculate_weight(i, beta) for i in indices])
        result = {
            'state': np.array(state_batch),
            'action': np.array(action_batch),
            'reward': np.array(reward_batch),
            'next_state': np.array(next_state_batch),
            'done': np.array(done_batch),
            'weights':weights,
            'indices':indices,
        }
        return result

        
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

