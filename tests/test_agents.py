import pathlib
import sys
import unittest

import gym
import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from drl.agents import DQAgent
from drl.envs import VectorEnv
from drl.estimators import DuelingDeepQNetwork
import drl.policies as policies
import drl.replay_buffers as replay_buffers


class DQAgentTest(unittest.TestCase):

    def setUp(self):
        test_env_name = 'CartPole-v1'
        self.env_fn = lambda: gym.make(test_env_name)
        self.env = self.env_fn()

        n_steps = 4
        memory = replay_buffers.NstepReplayBuffer(
            1000, self.env.observation_space.shape, 1)
        self.cfg = {
            'observation_space': self.env.observation_space,
            'action_space': self.env.action_space,
            # 'estimator': None,
            'replay_buffer': memory,
            'min_history': 64,
            'n_steps': n_steps,
            'gamma': 0.99,
            'batch_size': 64,
            'lr': 3e-4,
            'replace_target': 100,
            'noisy': True,
            'noisy_use_factorized': False,
            'parametrize': False,
            'device': 'cpu',
            'behaviour_policy': policies.EpsilonGreedyPolicy(0.1)
        }

    def test_should_create_normally(self):
        cfg = dict(self.cfg)
        del cfg['observation_space']
        del cfg['action_space']
        cfg['env'] = self.env
        DQAgent(**cfg)

        cfg = dict(self.cfg)
        del cfg['observation_space']
        del cfg['action_space']
        cfg['env_fn'] = self.env_fn
        DQAgent(**cfg)

        cfg = dict(self.cfg)
        DQAgent(**cfg)

    def test_raises_error_when_env_argument_is_garbage(self):
        cfg = dict(self.cfg)
        del cfg['observation_space']
        del cfg['action_space']
        with self.assertRaises(AttributeError):
            cfg['env'] = None
            DQAgent(**cfg)
        with self.assertRaises(ValueError):
            cfg['env'] = 666
            DQAgent(**cfg)

    def test_raises_error_when_envfn_argument_is_garbage(self):
        cfg = dict(self.cfg)
        del cfg['observation_space']
        del cfg['action_space']
        with self.assertRaises(AttributeError):
            cfg['env_fn'] = None
            DQAgent(**cfg)
        with self.assertRaises(ValueError):
            cfg['env_fn'] = lambda: 777
            DQAgent(**cfg)

    def test_raises_error_when_specs_argument_is_garbage(self):
        cfg = dict(self.cfg)
        with self.assertRaises(AttributeError):
            cfg['observation_space'] = None
            DQAgent(**cfg)
        with self.assertRaises(ValueError):
            cfg['observation_space'] = 234
            DQAgent(**cfg)

        cfg = dict(self.cfg)
        with self.assertRaises(AttributeError):
            cfg['action_space'] = None
            DQAgent(**cfg)
        with self.assertRaises(ValueError):
            cfg['action_space'] = 234
            DQAgent(**cfg)

    def test_shouldnt_create_if_env_and_envfn_both_passed(self):
        cfg = dict(self.cfg)
        del cfg['observation_space']
        del cfg['action_space']
        cfg['env'] = self.env
        cfg['env_fn'] = self.env_fn
        with self.assertRaises(ValueError):
            DQAgent(**cfg)

    def test_shouldnt_create_if_specs_and_env_or_envfn_passed(self):
        cfg = dict(self.cfg)
        cfg['env'] = self.env
        with self.assertRaises(ValueError):
            DQAgent(**cfg)

        cfg = dict(self.cfg)
        cfg['env_fn'] = self.env_fn
        with self.assertRaises(ValueError):
            DQAgent(**cfg)

    def test_should_act_on_observation(self):
        agent = DQAgent(**self.cfg)
        observation = self.env.reset()
        action = agent.action(observation)
        self.assertIn(action, self.env.action_space)
        self.env.step(action)

    @unittest.skipUnless(torch.cuda.is_available(), 'cuda test')
    def test_should_act_on_observation_when_using_cuda(self):
        self.cfg['device'] = 'cuda'
        agent = DQAgent(**self.cfg)
        observation = self.env.reset()
        action = agent.action(observation)
        self.assertIn(action, self.env.action_space)
        self.env.step(action)

    def test_should_act_on_vectorized_observation(self):
        vecenv_cfg = {
            'env_fn': self.env_fn,
            'num_envs': 16
        }
        vectorized_env = VectorEnv(**vecenv_cfg)
        agent = DQAgent(**self.cfg)
        vectorized_observation = vectorized_env.reset()
        vectorized_action = agent.action(vectorized_observation)
        vectorized_action = agent.action(np.array(vectorized_observation))
        vectorized_env.step(vectorized_action)

    def test_should_act_on_pseudo_vectorized_observation(self):
        vecenv_cfg = {
            'env_fn': self.env_fn,
            'num_envs': 1
        }
        vectorized_env = VectorEnv(**vecenv_cfg)
        agent = DQAgent(**self.cfg)
        vectorized_observation = vectorized_env.reset()
        vectorized_action = agent.action(vectorized_observation)
        vectorized_action = agent.action(np.array(vectorized_observation))
        vectorized_env.step(vectorized_action)

    def test_should_be_able_to_play_a_game_for_a_bit(self):
        max_steps = 1_000
        vecenv_cfg = {
            'env_fn': self.env_fn,
            'num_envs': 8
        }
        env = VectorEnv(**vecenv_cfg)
        agent = DQAgent(**self.cfg)
        observation = env.reset()
        for step in range(max_steps):
            action = agent.action(observation)
            obs_, reward, done, debug = env.step(action)
            if any(done):
                obs_ = env.reset(done)
            observation = obs_

    def test_should_store_played_trajectories_to_memory(self):
        steps = 200
        obs = [self.cfg['observation_space'].sample() for i in range(steps)]
        actions = [self.cfg['action_space'].sample() for i in range(steps)]
        rewards = [np.random.randn() for i in range(steps)]
        dones = [False] * (steps - 1) + [True]

        agent = DQAgent(**self.cfg)
        self.assertEqual(len(agent.replay_buffer), 0)
        agent.process_trajectory(obs, actions, rewards, dones)
        self.assertGreater(len(agent.replay_buffer), 0)
        self.assertEqual(len(agent.replay_buffer), steps)

    def test_agent_should_learn(self):
        steps = 200
        obs = [self.cfg['observation_space'].sample() for i in range(steps)]
        actions = [self.cfg['action_space'].sample() for i in range(steps)]
        rewards = [np.random.randn() for i in range(steps)]
        dones = [False] * (steps - 1) + [True]

        agent = DQAgent(**self.cfg)
        agent.process_trajectory(obs, actions, rewards, dones)
        q_params_before = agent.q_eval.state_dict()
        agent.learn()
        q_params_after = agent.q_eval.state_dict()
        for key in q_params_before:
            p1, p2 = q_params_before[key], q_params_after[key]
            self.assertFalse(torch.equal(p1, p2))


if __name__ == '__main__':
    unittest.main()
