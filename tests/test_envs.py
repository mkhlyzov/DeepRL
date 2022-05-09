import pathlib
import sys
import unittest

import gym
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from drl.envs import VectorEnv


class VectorEnvTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'env_fn': lambda: gym.make('CartPole-v1'),
            'num_envs': 16
        }

    def test_vectorenv_creation(self):
        VectorEnv(**self.cfg)

    def test_shouldnt_create_negative_or_zero_amount_of_envs(self):
        self.cfg['num_envs'] = 0
        with self.assertRaises(ValueError):
            VectorEnv(**self.cfg)

        self.cfg['num_envs'] = -20
        with self.assertRaises(ValueError):
            VectorEnv(**self.cfg)

    def test_should_not_create_when_passed_invalid_env_fn(self):
        self.cfg['env_fn'] = self.cfg['env_fn']()
        with self.assertRaises(ValueError):
            VectorEnv(**self.cfg)
        self.cfg['env_fn'] = lambda: 42
        with self.assertRaises(ValueError):
            VectorEnv(**self.cfg)

    def test_should_correctly_return_its_size(self):
        self.cfg['num_envs'] = 3
        vec_env = VectorEnv(**self.cfg)
        self.assertEqual(vec_env.num_envs, 3)

        self.cfg['num_envs'] = 7
        vec_env = VectorEnv(**self.cfg)
        self.assertEqual(vec_env.num_envs, 7)

    def test_should_inherit_specs(self):
        vec_env = VectorEnv(**self.cfg)
        self.assertEqual(
            vec_env.observation_space,
            self.cfg['env_fn']().observation_space
        )
        self.assertEqual(
            vec_env.action_space,
            self.cfg['env_fn']().action_space
        )

    def test_reset_method_returns_observations(self):
        vec_env = VectorEnv(**self.cfg)
        vec_observation = vec_env.reset()
        self.assertEqual(len(vec_observation), self.cfg['num_envs'])
        for obs in vec_observation:
            self.assertIn(obs, vec_env.observation_space)

    def test_should_not_step_before_initialization(self):
        vec_env = VectorEnv(**self.cfg)
        random_vec_action = [
            vec_env.action_space.sample()
            for _ in range(vec_env.num_envs)
        ]
        with self.assertRaises(Exception):
            vec_env.step(random_vec_action)

    def test_step_method_should_work_with_vectorized_action(self):
        vec_env = VectorEnv(**self.cfg)
        vec_env.reset()
        random_vec_action = [
            vec_env.action_space.sample()
            for _ in range(vec_env.num_envs)
        ]
        vec_env.step(random_vec_action)
        vec_env.step(np.array(random_vec_action))

    def test_step_method_should_not_work_with_non_iterable_action(self):
        vec_env = VectorEnv(**self.cfg)
        vec_env.reset()
        random_action = vec_env.action_space.sample()
        with self.assertRaises(Exception):
            vec_env.step(random_action)
        with self.assertRaises(Exception):
            vec_env.step(np.array(random_action))

    def test_step_method_should_not_work_with_wrong_length_action(self):
        vec_env = VectorEnv(**self.cfg)
        vec_env.reset()
        random_vec_action = [
            vec_env.action_space.sample()
            for _ in range(vec_env.num_envs * 5)
        ]
        with self.assertRaises(Exception):
            vec_env.step(random_vec_action)
        with self.assertRaises(Exception):
            vec_env.step(np.array(random_vec_action))

    def test_step_method_should_returns_obs_rews_dones_infos(self):
        vec_env = VectorEnv(**self.cfg)
        vec_env.reset()
        random_vec_action = [
            vec_env.action_space.sample()
            for _ in range(vec_env.num_envs)
        ]
        new_obs, rewards, dones, infos = vec_env.step(random_vec_action)
        self.assertEqual(len(new_obs), self.cfg['num_envs'])
        self.assertEqual(len(rewards), self.cfg['num_envs'])
        self.assertEqual(len(dones), self.cfg['num_envs'])
        self.assertEqual(len(infos), self.cfg['num_envs'])
        for obs_ in new_obs:
            self.assertIn(obs_, vec_env.observation_space)

    def test_step_method_can_autoreset_episodes_when_they_are_done(self):
        vec_env = VectorEnv(**self.cfg)
        max_episode_len = 1_000
        dones_old = [False] * vec_env.num_envs
        vec_env.reset()

        for i in range(max_episode_len):
            random_vec_action = [
                vec_env.action_space.sample()
                for _ in range(vec_env.num_envs)
            ]
            _1, _2, dones_new, _3 = vec_env.step(
                random_vec_action, auto_reset=True
            )
            for d_old, d_new in zip(dones_old, dones_new):
                if d_old:
                    self.assertNotEqual(d_new, d_old)
                    return
            dones_old = dones_new
        self.assertNotEqual(i, max_episode_len - 1)

    def test_step_method_default_shouldnt_autoreset_episodes(self):
        vec_env = VectorEnv(**self.cfg)
        max_episode_len = 1_000
        dones_old = [False] * vec_env.num_envs
        vec_env.reset()

        for i in range(max_episode_len):
            random_vec_action = [
                vec_env.action_space.sample()
                for _ in range(vec_env.num_envs)
            ]
            _1, _2, dones_new, _3 = vec_env.step(
                random_vec_action
            )
            for d_old, d_new in zip(dones_old, dones_new):
                if d_old:
                    self.assertEqual(d_new, d_old)
                    return
            dones_old = dones_new
        self.assertNotEqual(i, max_episode_len - 1)

    def test_episodes_should_end_in_1000_steps_or_tests_are_incomplete(self):
        vec_env = VectorEnv(**self.cfg)
        max_episode_len = 1_000
        vec_env.reset()

        for i in range(max_episode_len):
            random_vec_action = [
                vec_env.action_space.sample()
                for _ in range(vec_env.num_envs)
            ]
            _1, _2, dones, _3 = vec_env.step(random_vec_action)
            if any(dones):
                break
        self.assertNotEqual(i, max_episode_len - 1)


if __name__ == '__main__':
    unittest.main()
