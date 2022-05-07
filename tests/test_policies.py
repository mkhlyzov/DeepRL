import pathlib
import sys
import unittest

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from drl.policies import (
    Policy,
    GreedyPolicy,
    EpsilonGreedyPolicy
)


class PolicySamplingTests(unittest.TestCase):
    """

    Intended use:

    obs = env.reset()       (both standart gym.env and vectorized)
    q = q_function(obs)
    probs = policy.distribution(q)
    action = policy.sample(probs)
    _ = env.step(action)

                        shape
    obs:    np.ndarray() or  tuple(np.ndarray())
    q:      (n,)         or      (batch, n)
    probs   (n,)         or      (batch, n)

    policy.sample(probs):
            single value or ndarray
            int        or (batch, )

    sampled action dtype is always int, never tesor

    """

    def test_should_sample_from_batched_numpy_probs(self):
        policy = Policy()
        probs = np.ones((64, 8)) / 8.
        policy.sample(probs=probs)

    def test_should_sample_from_batched_pytorch_probs(self):
        policy = Policy()
        probs = torch.ones((64, 8)) / 8.
        policy.sample(probs=probs)

    def test_should_sample_from_1d_numpy_probs(self):
        policy = Policy()
        probs = np.ones(8) / 8.
        policy.sample(probs=probs)

    def test_should_sample_from_1d_pytorch_probs(self):
        policy = Policy()
        probs = torch.ones(8) / 8.
        policy.sample(probs=probs)

    def test_sample_raises_error_when_probs_shape_is_grater_than_2(self):
        policy = Policy()
        with self.assertRaises(Exception):
            probs = np.ones((5, 5, 5))
            policy.sample(probs=probs)
        with self.assertRaises(Exception):
            probs = np.ones((5, 5, 5, 5))
            policy.sample(probs=probs)

    def test_samples_shape_should_mimic_batch_dimention(self):
        policy = Policy()
        probs = np.ones(8) / 8
        action = policy.sample(probs)
        self.assertIsInstance(action, int)

        probs = np.ones((64, 8)) / 8
        action = policy.sample(probs)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.dtype, int)
        self.assertTupleEqual(action.shape, (64,))

        probs = torch.ones(8) / 8
        action = policy.sample(probs)
        self.assertIsInstance(action, int)

        probs = torch.ones((64, 8)) / 8
        action = policy.sample(probs)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.dtype, int)
        self.assertTupleEqual(action.shape, (64,))

    def test_samples_should_be_categorical(self):
        policy = Policy()
        probs = np.ones((64, 8)) / 8.
        action = policy.sample(probs)
        self.assertEqual(action.dtype, int)
        self.assertTrue(
            ((action >= 0) * (action < 8)).all()
        )

        probs = np.ones(8) / 8.
        action = policy.sample(probs)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 8)

        probs = torch.ones((64, 8)) / 8.
        action = policy.sample(probs)
        self.assertEqual(action.dtype, int)
        self.assertTrue(
            ((action >= 0) * (action < 8)).all()
        )

        probs = torch.ones(8) / 8.
        action = policy.sample(probs)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 8)

    def test_uniform_sampling_imperical_check(self):
        policy = Policy()
        probs = np.ones((500, 5)) / 5.
        samples = policy.sample(probs=probs)
        for count in np.bincount(samples):
            self.assertGreater(count, 50)

        probs = torch.ones((500, 5)) / 5.
        samples = policy.sample(probs=probs)
        for count in np.bincount(samples):
            self.assertGreater(count, 50)


class PolicyDistributionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if cls is PolicyDistributionTests:
            raise unittest.SkipTest("DerivedPolicyTests")
        super(PolicyDistributionTests, cls).setUpClass()

    def test_generated_numpy_distribution_inherits_input_shape(self):
        q = np.array([
            [14, 17, 31, 18, 22],
            [20, 86, 62, 52, 19],
            [61, 73, 44, 27, 60],
            [80, 69, 58, 32, 17],
            [53, 16, 94, 90, 34],
            [27, 13, 55, 41, 93]
        ])
        probs = self.policy.distribution(q)
        self.assertTupleEqual(q.shape, probs.shape)

        q = np.array([14, 17, 31, 18, 22])
        probs = self.policy.distribution(q)
        self.assertTupleEqual(q.shape, probs.shape)

    def test_generated_pytorch_distribution_inherits_input_shape(self):
        q = torch.tensor(np.array([
            [14, 17, 31, 18, 22],
            [20, 86, 62, 52, 19],
            [61, 73, 44, 27, 60],
            [80, 69, 58, 32, 17],
            [53, 16, 94, 90, 34],
            [27, 13, 55, 41, 93]
        ]))
        probs = self.policy.distribution(q)
        self.assertTupleEqual(q.shape, probs.shape)

        q = torch.tensor(np.array([14, 17, 31, 18, 22]))
        probs = self.policy.distribution(q)
        self.assertTupleEqual(q.shape, probs.shape)

    def test_numpy_distribution_should_inherit_input_dtype_for_floats(self):
        q = np.array([
            [14, 17, 31, 18, 22],
            [20, 86, 62, 52, 19]
        ], dtype=np.float16)
        probs = self.policy.distribution(q)
        self.assertEqual(q.dtype, probs.dtype)

    def test_pytorch_distribution_should_inherit_input_dtype_for_floats(self):
        q = torch.tensor(np.array([
            [14, 17, 31, 18, 22],
            [20, 86, 62, 52, 19]
        ], dtype=np.float16))
        probs = self.policy.distribution(q)
        self.assertEqual(q.dtype, probs.dtype)

    def test_numpy_distribution_should_fail_to_inherit_integer_dtype(self):
        q = np.array([
            [14, 17, 31, 18, 22],
            [20, 86, 62, 52, 19]
        ], dtype=np.int32)
        probs = self.policy.distribution(q)
        self.assertNotEqual(q.dtype, probs.dtype)

    def test_pytorch_distribution_should_fail_to_inherit_integer_dtype(self):
        q = torch.tensor(np.array([
            [14, 17, 31, 18, 22],
            [20, 86, 62, 52, 19]
        ], dtype=np.int32))
        probs = self.policy.distribution(q)
        self.assertNotEqual(q.dtype, probs.dtype)


class GreedyPolicyTests(PolicyDistributionTests):

    def setUp(self):
        self.policy = GreedyPolicy()

    def test_generated_numpy_1d_distribution_is_correct(self):
        q = np.array(
            [14, 17, 31, 18, 22])
        desired_probs = np.array(
            [0., 0., 1., 0., 0.])
        probs = self.policy.distribution(q)
        self.assertTrue(np.allclose(probs, desired_probs))

    def test_generated_pytorch_1d_distribution_is_correct(self):
        q = torch.tensor(np.array(
            [14, 17, 31, 18, 22], dtype=np.float32))
        desired_probs = torch.tensor(np.array(
            [0., 0., 1., 0., 0.], dtype=np.float32))
        probs = self.policy.distribution(q)
        self.assertTrue(torch.allclose(probs, desired_probs))

    def test_generated_numpy_batch_distribution_is_correct(self):
        policy = GreedyPolicy()
        q = np.array([
            [14, 17, 31, 18, 22],
            [20, 86, 62, 52, 19],
            [61, 73, 44, 27, 60],
            [80, 69, 58, 32, 17],
            [53, 16, 94, 90, 34],
            [27, 13, 55, 41, 93]
        ])
        desired_probs = np.array([
            [0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1.]
        ])
        probs = policy.distribution(q)
        self.assertTrue(np.allclose(probs, desired_probs))

    def test_generated_pytorch_batch_distribution_is_correct(self):
        policy = GreedyPolicy()
        q = torch.tensor(np.array([
            [14, 17, 31, 18, 22],
            [20, 86, 62, 52, 19],
            [61, 73, 44, 27, 60],
            [80, 69, 58, 32, 17],
            [53, 16, 94, 90, 34],
            [27, 13, 55, 41, 93]
        ], dtype=np.float32))
        desired_probs = torch.tensor(np.array([
            [0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1.]
        ], dtype=np.float32))
        probs = policy.distribution(q)
        self.assertTrue(torch.allclose(probs, desired_probs))

# class EpsilonGreedyPolicyTests(GreedyPolicyTests):

#     @unittest.skip('.')
#     def test_generated_numpy_distribution_inherits_input_shape(self):
#         policy = EpsilonGreedyPolicy()
#         q = np.array([
#             [14, 17, 31, 18, 22],
#             [20, 86, 62, 52, 19],
#             [61, 73, 44, 27, 60],
#             [80, 69, 58, 32, 17],
#             [53, 16, 94, 90, 34],
#             [27, 13, 55, 41, 93]
#         ])
#         probs = policy.distribution(q)
#         self.assertTupleEqual(q.shape, probs.shape)

#         q = np.array([14, 17, 31, 18, 22])
#         probs = policy.distribution(q)
#         self.assertTupleEqual(q.shape, probs.shape)

    # @unittest.skip('.')
    # def test_generated_pytorch_distribution_inherits_input_shape(self):
    #     policy = EpsilonGreedyPolicy()
    #     q = torch.tensor(np.array([
    #         [14, 17, 31, 18, 22],
    #         [20, 86, 62, 52, 19],
    #         [61, 73, 44, 27, 60],
    #         [80, 69, 58, 32, 17],
    #         [53, 16, 94, 90, 34],
    #         [27, 13, 55, 41, 93]
    #     ]))
    #     probs = policy.distribution(q)
    #     self.assertTupleEqual(q.shape, probs.shape)

    #     q = torch.tensor(np.array([14, 17, 31, 18, 22]))
    #     probs = policy.distribution(q)
    #     self.assertTupleEqual(q.shape, probs.shape)


if __name__ == '__main__':
    unittest.main()
