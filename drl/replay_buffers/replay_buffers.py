from collections.abc import Iterable

import random
import numpy as np

from drl.replay_buffers.segment_tree import (
    MinSegmentTree,
    SumSegmentTree
)


class Buffer(object):
    def __init__(self, *args, **kwargs):
        self.mem_size = kwargs['max_size']
        self.mem_cntr = 0

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

    def store_transition(self, *args):
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
        if isinstance(idx, int):
            assert idx < len(self)
        elif isinstance(idx, Iterable):
            assert max(idx) < len(self)

        state = self.state_memory[idx]
        state_ = self.new_state_memory[idx]
        action = self.action_memory[idx]
        reward = self.reward_memory[idx]
        terminal = self.terminal_memory[idx]

        return state, action, reward, state_, terminal

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = int(done)
        self.mem_cntr += 1


class NstepReplayBuffer(Buffer):
    def __init__(self, max_size, input_shape, hist_len=2, mod='numpy'):
        super(NstepReplayBuffer, self).__init__(max_size=max_size)

        if mod == 'numpy':
            self.states_memory = np.zeros(
                (self.mem_size, hist_len, *input_shape), dtype=np.float32)
            self.actions_memory = np.zeros(
                (self.mem_size, hist_len), dtype=np.int32)
            self.rewards_memory = np.zeros(
                (self.mem_size, hist_len), dtype=np.float32)
        elif mod == 'torch':
            import torch
            self.states_memory = torch.zeros(
                (self.mem_size, hist_len, *input_shape), dtype=torch.float32)
            self.actions_memory = torch.zeros(
                (self.mem_size, hist_len), dtype=torch.int32)
            self.rewards_memory = torch.zeros(
                (self.mem_size, hist_len), dtype=torch.float32)
        else:
            raise ValueError(f'Incorrect argument value mod={mod}')

    def __getitem__(self, idx):
        if isinstance(idx, int):
            assert idx < len(self)
        elif isinstance(idx, Iterable):
            assert max(idx) < len(self)

        states = self.states_memory[idx]
        actions = self.actions_memory[idx]
        rewards = self.rewards_memory[idx]

        return states, actions, rewards

    def store_transition(self, states, actions, rewards):
        idx = self.mem_cntr % self.mem_size
        self.states_memory[idx] = states
        self.actions_memory[idx] = actions
        self.rewards_memory[idx] = rewards
        self.mem_cntr += 1


class Prioritized(object):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(self, BufferInstance: Buffer, alpha=0.6):
        assert alpha >= 0

        self.mem_buffer = BufferInstance
        self.mem_buffer.clear()

        self.alpha = alpha
        self.max_priority = 1.
        self.tree_ptr = 0
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.mem_buffer.mem_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        # temporary variables for maintenance
        self.last_indices = None  # used for updating priorities
        self.awaiting_priority_update = False

    def __getitem__(self, idx):
        return (*self.mem_buffer[idx], self._calculate_weight(idx))

    def __len__(self):
        return len(self.mem_buffer)

    def store_transition(self, *args, **kwargs):
        self.mem_buffer.store_transition(*args, **kwargs)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        # self.tree_ptr = (self.tree_ptr + 1) % self.mem_buffer.mem_size
        self.tree_ptr = self.mem_buffer.mem_cntr % self.mem_buffer.mem_size

    def sample(self, batch_size, beta=None):
        """Sample a batch of experiences."""
        # assert not self.awaiting_priority_update
        assert len(self) >= batch_size
        if beta is None:
            beta = self.alpha * 0.67
        assert beta >= 0

        idx = self._sample_index(batch_size)
        data = self.mem_buffer[idx]
        weights = self._calculate_weight(idx, beta)

        self.last_indices = idx
        self.awaiting_priority_update = True

        return (*data, weights)

    def _sample_index(self, batch_size):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum()
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        assert isinstance(idx, (int, Iterable))
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        if isinstance(idx, int):
            p_sample = self.sum_tree[idx]
        elif isinstance(idx, Iterable):
            p_sample = np.array([self.sum_tree[i] for i in idx])
        p_sample /= self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

    def update_last_priorities(self, priorities):
        """Update priorities of sampled transitions."""
        # assert self.awaiting_priority_update
        assert len(self.last_indices) == len(priorities)

        for idx, priority in zip(self.last_indices, priorities):
            assert priority > 0

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

        self.last_indices = None
        self.awaiting_priority_update = False
