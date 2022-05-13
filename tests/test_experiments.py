import logging
import pathlib
import sys
import unittest

import gym
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from drl.agents import DQAgent
from drl.envs import MultiprocessVectorEnv
from drl.experiments import (
    Trainer,
    Trainer_old,
    evaluate_agent,
)


class EvaluationTest(unittest.TestCase):

    def setUp(self):
        self.env_fn = lambda: gym.make('CartPole-v1')
        self.agent = DQAgent(env_fn=self.env_fn, device='cpu')

    def test_raises_error_when_called_with_bad_arguments(self):
        with self.assertRaises(ValueError):
            evaluate_agent(self.agent, self.env_fn(), num_envs=4, num_steps=1000)

    def test_evaluates_agent_in_regular_environment(self):
        scores = evaluate_agent(self.agent, self.env_fn, num_steps=1000)
        self.assertIsNotNone(scores)

    def test_evaluates_agent_in_vectorized_environment(self):
        scores = evaluate_agent(
            self.agent, self.env_fn, num_steps=1000, num_envs=4)
        self.assertIsNotNone(scores)

    def test_evaluates_agent_in_pseudo_vectorized_environment(self):
        scores = evaluate_agent(
            self.agent, self.env_fn, num_steps=1000, num_envs=1)
        self.assertIsNotNone(scores)

    def test_evaluates_agen_in_multiprocess_vec_environment(self):
        env = MultiprocessVectorEnv(self.env_fn, 4)
        scores = evaluate_agent(self.agent, env, num_steps=1000)
        self.assertIsNotNone(scores)
        env.close()


@unittest.skip('.')
class TrainigTest(unittest.TestCase):

    def setUp(self):

        self.env_fn = lambda: gym.make('CartPole-v1')
        self.agent = DQAgent(
            env_fn=self.env_fn, device='cpu', batch_size=64, min_history=100
        )
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s: %(message)s',
            datefmt='%H:%M:%S',
        )

    def test_trains_agent_in_regular_environment(self):
        trainer = Trainer_old(self.agent, self.env_fn, samples_per_update=1)
        trainer.train(
            num_steps=1_000, eval_freq=1_000, eval_steps=1_000,
            plot=False, to_csv=False)

    def test_trains_agent_in_vectorized_environment(self):
        trainer = Trainer(
            self.agent, self.env_fn, samples_per_update=1, num_envs=8)
        trainer.train(
            num_steps=1_000, eval_freq=1_000, eval_steps=1_000,
            plot=False, to_csv=False)


if __name__ == '__main__':
    unittest.main()
