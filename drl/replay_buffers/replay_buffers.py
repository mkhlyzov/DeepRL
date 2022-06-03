from collections.abc import Iterable
from collections import deque

import numpy as np


class Buffer(object):
    def __init__(self, *args, **kwargs):
        self.capacity = kwargs['max_size']
        self.mem_cntr = 0

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __setitem__(self, idx, val):
        raise NotImplementedError()

    def __len__(self):
        return min(self.mem_cntr, self.capacity)

    def _assert_idx_is_valid(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx += self.mem_cntr % self.capacity
            assert 0 <= idx < len(self)
        elif isinstance(idx, Iterable):
            assert max(idx) < len(self)
        return idx

    def append(self, val):
        idx = self.mem_cntr % self.capacity
        self.mem_cntr += 1
        self[idx] = val

    def sample(self, batch_size=1):
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


class DequeBuffer(Buffer):
    def __init__(self, max_size, *args, **kwargs):
        self.data = deque(maxlen=max_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx]
        batch = [self.data[i] for i in idx]
        return tuple(np.array(items) for items in zip(*batch))

    def __setitem__(self, idx, val):
        self.data[idx] = val

    def append(self, val):
        self.data.append(val)

    def clear(self):
        self.data.clear()


class NumpyBuffer(Buffer):
    def __init__(self, max_size, *args, **kwargs):
        super(NumpyBuffer, self).__init__(max_size=max_size)
        self.data = np.empty(max_size, dtype=object)

    def __getitem__(self, idx):
        idx = self._assert_idx_is_valid(idx)
        batch = self.data[idx]
        if isinstance(idx, int):
            return batch
        return tuple(np.array(items) for items in zip(*batch))

    def __setitem__(self, idx, val):
        idx = self._assert_idx_is_valid(idx)
        self.data[idx] = val


class ReplayBuffer(Buffer):
    def __init__(self, max_size, observation_shape, mod='numpy'):
        super(ReplayBuffer, self).__init__(max_size=max_size)

        assert mod == 'numpy'

        self.state_memory = np.zeros(
            (self.capacity, *observation_shape), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.capacity, *observation_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.capacity, dtype=np.int64)
        self.reward_memory = np.zeros(self.capacity, dtype=np.float32)
        self.terminal_memory = np.zeros(self.capacity, dtype=np.bool_)

    def __getitem__(self, idx):
        idx = self._assert_idx_is_valid(idx)
        state = self.state_memory[idx]
        state_ = self.new_state_memory[idx]
        action = self.action_memory[idx]
        reward = self.reward_memory[idx]
        terminal = self.terminal_memory[idx]
        return state, action, reward, state_, terminal

    def __setitem__(self, idx, val):
        idx = self._assert_idx_is_valid(idx)
        state, action, reward, state_, done = val
        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = int(done)


class NstepReplayBuffer(Buffer):
    def __init__(self, max_size, observation_shape, n_steps=1, mod='numpy'):
        super(NstepReplayBuffer, self).__init__(max_size=max_size)
        self.n_steps = n_steps

        if mod == 'numpy':
            self.states_memory = np.zeros(
                (self.capacity, n_steps + 1, *observation_shape), dtype=np.float32)
            self.actions_memory = np.zeros(
                (self.capacity, n_steps + 1), dtype=np.int64)
            self.rewards_memory = np.zeros(
                (self.capacity, n_steps + 1), dtype=np.float32)
        elif mod == 'torch':
            import torch
            self.states_memory = torch.zeros(
                (self.capacity, n_steps + 1, *observation_shape), dtype=torch.float32)
            self.actions_memory = torch.zeros(
                (self.capacity, n_steps + 1), dtype=torch.long)
            self.rewards_memory = torch.zeros(
                (self.capacity, n_steps + 1), dtype=torch.float32)
        else:
            raise ValueError(f'Incorrect argument value mod={mod}')

    def __getitem__(self, idx):
        idx = self._assert_idx_is_valid(idx)
        states = self.states_memory[idx]
        actions = self.actions_memory[idx]
        rewards = self.rewards_memory[idx]
        return states, actions, rewards

    def __setitem__(self, idx, val):
        idx = self._assert_idx_is_valid(idx)
        states, actions, rewards = val
        self.states_memory[idx] = states
        self.actions_memory[idx] = actions
        self.rewards_memory[idx] = rewards
