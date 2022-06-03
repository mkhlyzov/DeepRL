from collections.abc import Iterable
import pathlib
import sys
import unittest

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from drl.replay_buffers import (
    NumpyBuffer,
    DequeBuffer,
    ReplayBuffer,
    NstepReplayBuffer,
    Prioritized,
)
from drl.replay_buffers.segment_tree import (
    SegmentTree,
    SumSegmentTree,
    MinSegmentTree
)


def items_equal(item1, item2):
    if isinstance(item1, Iterable) and isinstance(item2, Iterable):
        is_equal = True
        for i1, i2 in zip(item1, item2):
            is_equal *= items_equal(i1, i2)
        return bool(is_equal)
    if not isinstance(item1, Iterable) and not isinstance(item2, Iterable):
        return np.isclose(item1, item2)
    return False


class BufferTestCase(object):
    class BufferTest(unittest.TestCase):

        def setUp(self):
            self.max_size = 100_000
            self.observation_shape = (8,)

        def test_should_be_created_empty(self):
            self.assertEqual(len(self.buffer), 0)

        def test_after_append_should_not_be_empty(self):
            self.buffer.append(self.fake_item)
            self.assertGreater(len(self.buffer), 0)

        def test_after_deleting_all_items_should_be_empty(self):
            self.buffer.append(self.fake_item)
            self.buffer.append(self.fake_item)
            self.buffer.clear()
            self.assertEqual(len(self.buffer), 0)

        def test_last_item_should_be_the_one_we_appended_earlier(self):
            self.buffer.append(self.fake_item)
            item = self.buffer[-1]
            self.assertTrue(items_equal(item, self.fake_item))

        def test_should_not_overflow_max_capacity(self):
            for _ in range(self.max_size * 2):
                self.buffer.append(self.fake_item)
            self.assertEqual(len(self.buffer), self.max_size)

        def test_sampled_batch_should_inherit_dimentionality(self):
            batch_size = 64
            for _ in range(self.max_size):
                self.buffer.append(self.fake_item)
            batch = self.buffer.sample(batch_size)

            # batch is a sarsa-like tuple of np.ndarrays
            self.assertEqual(len(batch), len(self.fake_item))
            for batch_i, item_i in zip(batch, self.fake_item):
                self.assertEqual(batch_i.shape[0], batch_size)
                if isinstance(item_i, Iterable):
                    self.assertEqual(batch_i.shape[1:], item_i.shape)
                else:
                    self.assertFalse(isinstance(batch_i[0], Iterable))


class ReplayBufferTest(BufferTestCase.BufferTest):

    def setUp(self):
        super().setUp()
        self.buffer = ReplayBuffer(self.max_size, self.observation_shape)
        self.fake_item = (
            np.ones(self.observation_shape, dtype=np.float32),
            1,
            1.5,
            np.ones(self.observation_shape, dtype=np.float32) * 2,
            False
        )


class NstepReplayBufferTest(BufferTestCase.BufferTest):

    def setUp(self):
        super().setUp()
        self.buffer = NstepReplayBuffer(
            self.max_size, self.observation_shape, n_steps=2)
        self.fake_item = (
            np.ones((3, *self.observation_shape), dtype=np.float32),
            np.array([0, 1, 2]),
            np.array([-0.33, 1.5, float('inf')])
        )


class NumpyBufferTest_sarsa(BufferTestCase.BufferTest):

    def setUp(self):
        super().setUp()
        self.buffer = NumpyBuffer(self.max_size)
        self.fake_item = (
            np.ones(self.observation_shape, dtype=np.float32),
            1,
            1.5,
            np.ones(self.observation_shape, dtype=np.float32) * 2,
            False
        )


class NumpyBufferTest_nstep(BufferTestCase.BufferTest):

    def setUp(self):
        super().setUp()
        self.buffer = NumpyBuffer(self.max_size)
        self.fake_item = (
            np.ones((3, *self.observation_shape), dtype=np.float32),
            np.array([0, 1, 2]),
            np.array([-0.33, 1.5, float('inf')])
        )


class DequeBufferTest(BufferTestCase.BufferTest):

    def setUp(self):
        super().setUp()
        self.buffer = DequeBuffer(self.max_size)
        self.fake_item = (
            np.ones((3, *self.observation_shape), dtype=np.float32),
            np.array([0, 1, 2]),
            np.array([-0.33, 1.5, float('inf')])
        )


if __name__ == '__main__':
    unittest.main()
