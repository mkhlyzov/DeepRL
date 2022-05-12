from collections.abc import Iterable

import random
import numpy as np


class Buffer(object):
    def __init__(self, *args, **kwargs):
        self.mem_size = kwargs['max_size']
        self.mem_cntr = 0

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __setitem__(self, idx, val):
        raise NotImplementedError()

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

    def _assert_idx_is_valid(self, idx):
        if isinstance(idx, int):
            assert idx < len(self)
        elif isinstance(idx, Iterable):
            assert max(idx) < len(self)

    def append(self, *args):
        raise NotImplementedError()

    def sample(self, batch_size):
        idx = self._sample_index(batch_size)
        return self.__getitem__(idx)

    def _sample_index(self, batch_size):
        """
        Sampling batch index w/o replacement takes too long for big replay
        buffers. Sampling WITH replacement, on the other hand, won't
        produce duplicates (with high probability) if memory buffer is large
        enough.
        k = 100  yields less than 0.50% of duplicates
        k = 200  yields less than 0.25% of duplicates
        k = 1000 yields less than 0.05% of duplicates
        """
        max_mem = len(self)
        k = 200
        if max_mem > batch_size * k:
            idx = np.random.choice(max_mem, batch_size, replace=True)
        else:
            idx = np.random.choice(max_mem, batch_size, replace=False)

        return idx

    def clear(self):
        self.mem_cntr = 0


class ReplayBuffer(Buffer):
    def __init__(self, max_size, input_shape, mod='numpy'):
        super(ReplayBuffer, self).__init__(max_size=max_size)

        assert mod == 'numpy'

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=dtype)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=dtype)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def __getitem__(self, idx):
        self._assert_idx_is_valid(idx)

        state = self.state_memory[idx]
        state_ = self.new_state_memory[idx]
        action = self.action_memory[idx]
        reward = self.reward_memory[idx]
        terminal = self.terminal_memory[idx]

        return state, action, reward, state_, terminal

    def append(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = int(done)
        self.mem_cntr += 1


class NstepReplayBuffer(Buffer):
    def __init__(self, max_size, observation_shape, n_steps=1, mod='numpy'):
        super(NstepReplayBuffer, self).__init__(max_size=max_size)
        self.n_steps = n_steps

        if mod == 'numpy':
            self.states_memory = np.zeros(
                (self.mem_size, n_steps + 1, *observation_shape), dtype=np.float32)
            self.actions_memory = np.zeros(
                (self.mem_size, n_steps + 1), dtype=np.int64)
            self.rewards_memory = np.zeros(
                (self.mem_size, n_steps + 1), dtype=np.float32)
        elif mod == 'torch':
            import torch
            self.states_memory = torch.zeros(
                (self.mem_size, n_steps + 1, *observation_shape), dtype=torch.float32)
            self.actions_memory = torch.zeros(
                (self.mem_size, n_steps + 1), dtype=torch.long)
            self.rewards_memory = torch.zeros(
                (self.mem_size, n_steps + 1), dtype=torch.float32)
        else:
            raise ValueError(f'Incorrect argument value mod={mod}')

    def __getitem__(self, idx):
        self._assert_idx_is_valid(idx)
        states = self.states_memory[idx]
        actions = self.actions_memory[idx]
        rewards = self.rewards_memory[idx]
        return states, actions, rewards

    def __setitem__(self, idx, val):
        self._assert_idx_is_valid(idx)
        states, actions, rewards = val
        self.states_memory[idx] = states
        self.actions_memory[idx] = actions
        self.rewards_memory[idx] = rewards

    def append(self, states, actions, rewards):
        idx = self.mem_cntr % self.mem_size
        self.mem_cntr += 1
        self[idx] = states, actions, rewards
