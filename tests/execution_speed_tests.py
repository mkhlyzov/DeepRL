import functools
import pathlib
import sys
import time

import gym
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from drl.agents import DQAgent
from drl.envs import *
from drl.estimators import DuelingDeepQNetwork
from drl.estimators.nn import NoisyLinear
from drl.experiments import Trainer, evaluate_agent
from drl.utils import (
    GET_TIME, highlight_output, slow_down
)
from drl.replay_buffers import (
    DequeBuffer, NumpyBuffer, ReplayBuffer, NstepReplayBuffer, Prioritized
)


class ConfigGenerator():
    def __call__(self, cfg_dict, _unroll_from=0):
        """
        _unroll_from hack is used so that generator doesn't unroll \
        list of lists more than once.
        """
        for i, key in enumerate(
                list(cfg_dict.keys())
        ):
            if i < _unroll_from:
                continue
            if isinstance(cfg_dict[key], list):
                for value in cfg_dict[key]:
                    cfg_dict_ = dict(cfg_dict)
                    cfg_dict_[key] = value
                    yield from self.__call__(cfg_dict_, _unroll_from=i + 1)
                return
        yield cfg_dict


@highlight_output
def test_noisy_linear(n_iters=100_000):
    """
    Benchmark:
            - torch.nn.Linear (cpu/cuda)
            - NoisyLinear frozen (cpu/cuda)
            - NoisyLinear with noise reset (cpu/cuda)
    """
    config_dict = {
        'in_features': [128, 256],
        'batch_size': [64, 256],
        'device': ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'],
    }

    def run(cfg, n_iters):
        data = torch.randn(
            cfg['batch_size'], cfg['in_features']).to(cfg['device'])
        linear_layer = torch.nn.Linear(
            cfg['in_features'], cfg['in_features']).to(cfg['device'])
        noisy_layer = NoisyLinear(
            cfg['in_features'], cfg['in_features']).to(cfg['device'])

        print(f'cfg={cfg}, n_iters={n_iters}')

        t0 = GET_TIME()
        for i in range(n_iters):
            _ = linear_layer(data)
        time_linear = GET_TIME() - t0
        print(f'time linear:            {time_linear:.2f}s')

        t0 = GET_TIME()
        for i in range(n_iters):
            _ = noisy_layer(data)
        time_noisy_frozen = GET_TIME() - t0
        print(f'time noisy frozen:      {time_noisy_frozen:.2f}s')

        t0 = GET_TIME()
        for i in range(n_iters):
            noisy_layer.reset_noise()
            _ = noisy_layer(data)
        time_noisy_reset = GET_TIME() - t0
        print(f'time noisy reset:       {time_noisy_reset:.2f}s')

        t0 = GET_TIME()
        noisy_layer.use_factorised = True
        for i in range(n_iters):
            noisy_layer.reset_noise()
            _ = noisy_layer(data)
        time_noisy_reset = GET_TIME() - t0
        print(f'time factorised reset:  {time_noisy_reset:.2f}s')

    print('test_noisy_linear: Starting tests.\n')
    for cfg in ConfigGenerator()(config_dict):
        run(cfg, n_iters)
        print('')
    print('test_noisy_linear: Tests performed.')


@highlight_output
def test_deep_q_network(n_iters=50_000):
    """
    Benchmark:
            - DuelingDeepQNetwork regular forward pass (cpu/cuda)
                    * with parametrizations
                    * w/o  parametrizations
                    * with noise factorization
                    * w/o  noise factorization
    """
    config_dict = {
        'batch_size': [64],
        'input_dims': [8],
        'n_actions': [4],
        'neurons': [[128, 128]],
        # 'factorised': [True, False],
        'parametrize': [True, False],
        'device': ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'],
    }

    def run(cfg, n_iters):
        q_network = DuelingDeepQNetwork(
            input_dims=cfg['input_dims'],
            n_actions=cfg['n_actions'],
            neurons=cfg['neurons'],
            noisy=True,
            factorised=False,
            parametrize=cfg['parametrize']
        ).to(cfg['device'])
        data = torch.randn(
            cfg['batch_size'], cfg['input_dims']).to(cfg['device'])

        print(f'cfg={cfg}, n_iters={n_iters}')

        t0 = GET_TIME()
        for i in range(n_iters):
            _ = q_network(data)
        time_frozen = GET_TIME() - t0
        print(f'time frozen:     {time_frozen:.2f}s')

        t0 = GET_TIME()
        q_network.factorised_noise(False)
        for i in range(n_iters):
            q_network.reset_noise()
            _ = q_network(data)
        time_reset = GET_TIME() - t0
        print(f'time reset:      {time_reset:.2f}s')

        t0 = GET_TIME()
        q_network.factorised_noise(True)
        for i in range(n_iters):
            q_network.reset_noise()
            _ = q_network(data)
        time_reset = GET_TIME() - t0
        print(f'time factorised: {time_reset:.2f}s')

    print('test_deep_q_network: Starting tests.\n')
    for cfg in ConfigGenerator()(config_dict):
        run(cfg, n_iters)
        print('')
    print('test_deep_q_network: Tests performed.')


@highlight_output
def test_evaluation(n_iters=50_000):
    """
    Benchmark:
            - idle environment stepping (no neural nets)
            - action making using agent.action(observation) (cpu/cuda)
    """
    config_dict = {
        'device': ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'],
        'no_ops': [float('inf'), 0],
        'num_envs': [1, 8, 32]
    }

    """
    noops stepping
    agent stepping (cpu / cuda)

    non-vectorized env
    sync_vecenv (1, 8, 32)
    async_vecenv (1, 8, 32)
    """

    def run(cfg, n_iters):
        def env_fn(): return gym.make('LunarLander-v2')
        env_fns = [env_fn for _ in range(cfg['num_envs'])]
        env = env_fn()
        agent = DQAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=cfg['device']
        )

        print(f'cfg={cfg}, n_iters={n_iters}')

        if cfg['num_envs'] == 1:
            env = env_fn()
            t0 = GET_TIME()
            _ = evaluate_agent(agent, env, num_steps=n_iters, no_ops=cfg['no_ops'])
            time_eval = GET_TIME() - t0
            print(f'1-env time eval: {time_eval:.2f}s')

        env = gym.vector.SyncVectorEnv(env_fns)
        t0 = GET_TIME()
        _ = evaluate_agent(agent, env, num_steps=n_iters, no_ops=cfg['no_ops'])
        time_eval = GET_TIME() - t0
        print(f'Sync time eval : {time_eval:.2f}s')

        env = gym.vector.AsyncVectorEnv(env_fns)
        t0 = GET_TIME()
        _ = evaluate_agent(agent, env, num_steps=n_iters, no_ops=cfg['no_ops'])
        time_eval = GET_TIME() - t0
        print(f'Async time eval: {time_eval:.2f}s')

    print('test_evaluation: Starting tests.\n')
    for cfg in ConfigGenerator()(config_dict):
        if cfg['no_ops'] and ('cuda' in cfg.keys()) and cfg['cuda']:
            continue
        run(cfg, n_iters)
        print('')
    print('test_evaluation: Tests performed.')


@highlight_output
def test_training(n_iters=None):
    """
    Benchmark:
            - agent.learn() speed
    """
    config_dict = {
    }

    def run(cfg, n_iters):
        pass

    print('test_training: Starting tests.\n')
    for cfg in ConfigGenerator()(config_dict):
        run(cfg, n_iters)
        print('')
    print('test_training: Tests performed.')


@highlight_output
def test_replay_buffers(n_iters=None):
    """
    Benchmark:
            - DequeBuffer       (data appending / batch sampling)
            - NumpyBuffer       (data appending / batch sampling)
            - ReplayBuffer      (data appending / batch sampling)
            - Prioritized       (data appending / batch sampling)
    """
    config_dict = {
        'buffer_constructor': [ReplayBuffer, NumpyBuffer, DequeBuffer],
        'batch_size': [64]
    }

    def run(cfg, n_iters):
        max_size = 1_000_000
        observation_shape = (8,)
        buffer = cfg['buffer_constructor'](max_size, observation_shape)
        fake_item = (
            np.zeros(shape=observation_shape, dtype=np.float32),
            0,
            0.,
            np.zeros(shape=observation_shape, dtype=np.float32),
            0
        )

        print(f'cfg={cfg}, n_iters={n_iters}')

        t0 = GET_TIME()
        for _ in range(n_iters):
            buffer.append(fake_item)
        append_time = GET_TIME() - t0
        print(f'Append time:    {append_time:.2f}s')

        t0 = GET_TIME()
        for _ in range(n_iters):
            buffer.sample(cfg['batch_size'])
        sample_time = GET_TIME() - t0
        print(f'Sample time:    {sample_time:.2f}s')

    print('test_replay_buffers: Starting tests.\n')
    for cfg in ConfigGenerator()(config_dict):
        run(cfg, n_iters)
        print('')
    print('test_replay_buffers: Tests performed.')


if __name__ == '__main__':
    # test_noisy_linear(100_000)
    # test_deep_q_network(50_000)
    # test_evaluation(20_000)
    test_replay_buffers(1_000_000)
