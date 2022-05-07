import pathlib
import sys
import unittest

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from drl.policies import (
    Policy,
    GreedyPolicy,
    EpsilonGreedyPolicy,
    BoltzmannPolicy,
    MixedPolicy
)


class TorchSamplingTests(unittest.TestCase):

    def setUp(self):
        self.probs1d = torch.ones(8, dtype=torch.float32) / 8.
        self.probs2d = torch.ones((64, 8), dtype=torch.float32) / 8.
        self.policy = Policy()

    def test_should_sample_from_1d_probs(self):
        self.policy.sample(probs=self.probs1d)

    def test_should_sample_from_2d_probs(self):
        self.policy.sample(probs=self.probs2d)

    def test_sample_raises_error_when_probs_shape_is_grater_than_2(self):
        with self.assertRaises(Exception):
            probs = torch.ones((5, 5, 5) / 5.)
            self.policy.sample(probs=probs)
        with self.assertRaises(Exception):
            probs = torch.ones((5, 5, 5, 5) / 5.)
            self.policy.sample(probs=probs)

    def test_samples_shape_should_mimic_batch_dimention(self):
        action = self.policy.sample(self.probs1d)
        self.assertTupleEqual(action.shape, self.probs1d.shape[:-1])

        action = self.policy.sample(self.probs2d)
        self.assertTupleEqual(action.shape, self.probs2d.shape[:-1])

    def test_samples_should_be_categorical(self):
        action = self.policy.sample(self.probs1d)
        self.assertFalse(action.dtype.is_floating_point)
        self.assertTrue(
            (0 <= action < self.probs1d.shape[-1]).all().item()
        )

        action = self.policy.sample(self.probs2d)
        self.assertFalse(action.dtype.is_floating_point)
        self.assertTrue(
            ((action >= 0) * (action < self.probs2d.shape[-1])).all()
        )

    def test_uniform_sampling_imperical_check(self):
        uniform_probs = torch.ones((500, 5)) / 5.
        samples = self.policy.sample(probs=uniform_probs)
        for count in torch.bincount(samples):
            self.assertGreater(count, 50)


class TorchProbsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if cls is TorchProbsTests:
            raise unittest.SkipTest("DerivedPolicyTests")
        super(TorchProbsTests, cls).setUpClass()

    def setUp(self):
        self.q1d = torch.tensor(np.array(
            [14, 17, 31, 18, 22])
        ).float()
        self.q2d = torch.tensor(np.array([
            [14, 17, 31, 18, 22],
            [20, 86, 62, 52, 19],
            [61, 73, 44, 27, 60],
            [80, 69, 58, 32, 17],
            [53, 16, 94, 90, 34],
            [27, 13, 55, 41, 93]
        ])).float()

    def test_generated_probs_inherits_input_shape(self):
        probs = self.policy.probs(self.q1d)
        self.assertTupleEqual(self.q1d.shape, probs.shape)

        probs = self.policy.probs(self.q2d)
        self.assertTupleEqual(self.q2d.shape, probs.shape)

    def test_generated_probs_should_inherit_input_dtype_for_floats(self):
        dtype = torch.float16
        probs = self.policy.probs(self.q2d.to(dtype))
        self.assertEqual(probs.dtype, dtype)

        dtype = torch.float64
        probs = self.policy.probs(self.q2d.to(dtype))
        self.assertEqual(probs.dtype, dtype)

    @unittest.skip('Undefined behaviour')
    def test_generated_probs_should_fail_to_inherit_integer_dtype(self):
        dtype = torch.int16
        probs = self.policy.probs(self.q2d.to(dtype))
        self.assertNotEqual(probs.dtype, dtype)

        dtype = torch.int64
        probs = self.policy.probs(self.q2d.to(dtype))
        self.assertNotEqual(probs.dtype, dtype)

    @unittest.skipUnless(torch.cuda.is_available(), 'cuda test')
    def test_generated_probs_should_inherit_input_device(self):
        device = torch.device('cuda')
        probs = self.policy.probs(self.q1d.to(device))
        self.assertEqual(probs.device, device)

        probs = self.policy.probs(self.q2d.to(device))
        self.assertEqual(probs.device, device)


class GreedyPolicyTests(TorchProbsTests):

    def setUp(self):
        super().setUp()
        self.policy = GreedyPolicy()

    def test_generated_1d_probs_is_correct(self):
        q = self.q1d.to(torch.float32)
        desired_probs = torch.tensor(np.array(
            [0., 0., 1., 0., 0.], dtype=np.float32))
        probs = self.policy.probs(q)
        self.assertTrue(torch.allclose(probs, desired_probs))

    def test_generated_2d_probs_is_correct(self):
        q = self.q2d.to(torch.float32)
        desired_probs = torch.tensor(np.array([
            [0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1.]
        ], dtype=np.float32))
        probs = self.policy.probs(q)
        self.assertTrue(torch.allclose(probs, desired_probs))


class EpsilonGreedyPolicyTests(TorchProbsTests):

    def setUp(self):
        super().setUp()
        self.policy = EpsilonGreedyPolicy(epsilon=0.1)

    def test_generated_1d_probs_is_correct(self):
        q = self.q1d.to(torch.float32)
        desired_probs = torch.tensor(np.array(
            [0.02, 0.02, 0.92, 0.02, 0.02], dtype=np.float32))
        probs = self.policy.probs(q)
        self.assertTrue(torch.allclose(probs, desired_probs))

    def test_generated_2d_probs_is_correct(self):
        q = self.q2d.to(torch.float32)
        desired_probs = torch.tensor(np.array([
            [0.02, 0.02, 0.92, 0.02, 0.02],
            [0.02, 0.92, 0.02, 0.02, 0.02],
            [0.02, 0.92, 0.02, 0.02, 0.02],
            [0.92, 0.02, 0.02, 0.02, 0.02],
            [0.02, 0.02, 0.92, 0.02, 0.02],
            [0.02, 0.02, 0.02, 0.02, 0.92]
        ], dtype=np.float32))
        probs = self.policy.probs(q)
        self.assertTrue(torch.allclose(probs, desired_probs))


class BoltzmannPolicyTests(TorchProbsTests):

    def setUp(self):
        super().setUp()
        self.policy = BoltzmannPolicy(temperature=1)


class MixedPolicyTests(TorchProbsTests):

    def setUp(self):
        super().setUp()
        p1 = GreedyPolicy()
        p2 = BoltzmannPolicy(temperature=1)
        self.policy = MixedPolicy([p1, p2], [0.5, 1])


if __name__ == '__main__':
    unittest.main()
