import numpy as np

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





