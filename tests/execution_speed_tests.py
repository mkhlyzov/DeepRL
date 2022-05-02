import time

import torch

from drl.agents import DQAgent
from drl.envs import *
from drl.estimators import DuelingDeepQNetwork
from drl.estimators.nn import NoisyLinear
from drl.experiments
from drl.replay_buffers import NstepReplayBuffer, Prioritized


def test_noisy_linear():
	"""
	Benchmark:
		- torch.nn.Linear
		- NoisyLinear frozen
		- NoisyLinear with noise reset
	"""
	pass

def test_deep_q_network():
	"""
	Benchmark:
		- DuelingDeepQNetwork regular forward pass
			* with parametrizations
			* w/o  parametrizations
	"""
	pass

def test_evaluation():
	"""
	Benchmark:
		- idle environment stepping (no neural nets)
		- action making using agent.action(observation)
	"""
	pass
