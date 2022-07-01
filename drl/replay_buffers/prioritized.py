from collections.abc import Iterable

import random
import numpy as np

from drl.replay_buffers.segment_tree import (
    MinSegmentTree,
    SumSegmentTree
)
from drl.replay_buffers import Buffer


class Prioritized(object):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
        self,
        BufferInstance: Buffer,
        *,
        alpha=0.6,
        max_priority=1.,
    ):
        assert alpha >= 0
        assert max_priority > 0

        self.mem_buffer = BufferInstance
        self.mem_buffer.clear()

        self.alpha = alpha
        self.max_priority = max_priority
        self.tree_ptr = 0
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.mem_buffer.capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        # temporary variables for maintenance
        self.last_indices = None  # used for updating priorities
        self.awaiting_priority_update = False

    def __setitem__(self, idx, val):
        self.mem_buffer[idx] = val
        self.sum_tree[idx] = self.max_priority ** self.alpha
        self.min_tree[idx] = self.max_priority ** self.alpha

    def __getitem__(self, idx, beta=None):
        return (*self.mem_buffer[idx], self._calculate_weight(idx, beta))

    def __len__(self):
        return len(self.mem_buffer)

    def append(self, *args, **kwargs):
        self.mem_buffer.append(*args, **kwargs)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = self.mem_buffer.mem_cntr % self.mem_buffer.capacity

    def sample(self, batch_size=1, beta=None):
        """Sample a batch of experiences."""
        # assert not self.awaiting_priority_update
        assert len(self) >= batch_size

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

    def _calculate_weight(self, idx, beta=None):
        """Calculate the weight of the experience at idx."""
        assert isinstance(idx, (int, Iterable))
        if beta is None:
            beta = self.alpha * 0.67
        assert beta >= 0
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
