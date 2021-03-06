import gc
import importlib
import json
import logging
import pathlib
import sys
import time

import gym
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output

ROOT = pathlib.Path.cwd()
# ROOT = '/Users/a1/Dev/DeepRL'
sys.path.append(ROOT)

import drl.agents as agents
import drl.experiments as experiments
import drl.policies as policies
import drl.utils as utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    datefmt='%H:%M:%S',
)
torch.set_num_threads(1)
#####################################################################
#%%
env_fn = lambda: gym.make('LunarLander-v2')
env = env_fn()

agent = agents.DQAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,

    noisy=False, noisy_use_factorized=False, parametrize=False,
    behaviour_policy=policies.BoltzmannPolicy(0.01),
    target_policy=policies.BoltzmannPolicy(0.01),

    mem_size=500_000, min_history=1_000, batch_size=64,
    lr=3e-4, gamma=0.99, n_steps=1, replace_target=100,

    device='cpu',
    fname='DQAgent_model.pt',
)
trainer = experiments.Trainer(
    agent, env_fn,
    samples_per_update=1,
    metrics='all',
    log_dir=pathlib.Path(ROOT).joinpath('logs/LunarLander_v2/!config_0'),
    num_envs=16,
    multiprocessing=False
)
#%% training agent
trainer.train(
    num_steps=2_000_000, eval_freq=20_000, report_freq=200_000,
    eval_steps=30_000, plot=True, to_csv=True,
    sleep_after_evaluation=15
)
"""
100k eval steps with noise reset: ~90s -> 120s???
100k eval steps w/o  noise reset: ~75s -> 103s???
100k train steps w/o prioritized 1spa: 90->120s (if parameterized)
100k train steps w   prioritized 1spa: ->175s
"""
"""PaperSpace:
100k train steps w/o prioritized 1spa: 145s 260s (cpu vs cuda)
(150?) 180 - 220 - 250+ (s) cpu training (best seen, average, worst)
"""
#%% vec_env evaluation
t0 = time.perf_counter()
eval_scores = experiments.evaluate_agent(
    agent, env_fn, num_steps=100_000, no_ops=0, num_envs=16)
print('time taken (s):', time.perf_counter() - t0)
print('avg_eval_score =', np.mean(eval_scores))
print('eval_score_std =', np.std(eval_scores))
print(f'episods finished: {len(eval_scores)}')

# Fresh agent (random, 1400e) ~ 10s
# Trained (1.2m steps, score=-200 diverged, hot cpu, 320e) ~ 64s
# Trained (20k steps, score=-140, 216e) ~ 73s
# Trained (40k steps, score=-60, 99e) ~ 110s
# trained (280k steps, score=163, 203e) ~ 50s
# Trained (420k steps, score=124, 234e) ~ 44s
# Trained (600k steps, score=240, 160e) ~ 50s
# Trained (880k steps, score=184, 404e) ~ 30s
# Trained (1.5m steps, score=282, 485e) ~ 24s
#%% testing agent.action time
fake_state = [env.observation_space.sample() for j in range(32)]
# fake_state = np.array(fake_state)
t0 = time.perf_counter()
for i in range(100_000 // 32):
    agent.action(fake_state)
print('agent.action 100k/32, time taken (s):', time.perf_counter() - t0)
# Trained 600k steps //32 *32, time ~ 6.5s
# Trained 880k steps //32 *32, time ~ 6.1s
# Trained 1.5m steps //32 *32, time ~ 6.2s
#%% testing env.step time
t0 = time.perf_counter()
done = True
for i in range(100_000):
    if done:
        env.reset()
    s_, r, done, _ = env.step(env.action_space.sample())
print('env.step 100k, time taken (s):', time.perf_counter() - t0)
# time ~ 12.5s
#%% testing env.reset time
t0 = time.perf_counter()
for i in range(100_000):
    env.reset()
print('env.reset 100k, time taken (s):', time.perf_counter() - t0)
# time ~ 30s
#%% Playing for 1 episode
env = gym.make('LunarLander-v2')
# env = gym.wrappers.RecordVideo(env, ROOT / 'logs/videos')
done = False
score = 0
observation = env.reset()
t0 = time.perf_counter()
num_steps = 0

while not done:
    env.render(mode='human')
    action = agent.action(observation)

    observation_, reward, done, _ = env.step(action)
    score += reward
    observation = observation_
    # agent.learn()
    num_steps += 1
print(f'{score=:.1f}   frames={num_steps}')
print(time.perf_counter() - t0)
env.close()
env = gym.make('LunarLander-v2')
#%%
batch_size = 64
input_shape = env.observation_space.shape
n_actions = env.action_space.n
s1 = np.random.random((1, *input_shape))
s64 = np.random.random((batch_size, *input_shape))

s1 = torch.randn((1, *input_shape))
s64 = torch.randn((batch_size, *input_shape))
s = np.random.random(8)

counter = 0
t0 = time.perf_counter()
# with torch.autograd.profiler.profile(profile_memory=True) as prof:
for i in range(20_000):
    pass
    # agent.q_eval.reset_noise()
    # agent.action(s64)
    agent.learn()
    # agent.q_eval(s64)
    # agent._update_network_parameters()
    # agent.memory.sample(64)
    # agent.memory.mem_buffer.sample_buffer(64)
    # counter += set(np.random.choice(64*100, 64, replace=False)).__len__()
print(time.perf_counter() - t0)
#%% probabilistic distrib over actions
n_samples = 20
tmp_dict = [
    {i: 0 for i in range(4)} for _ in range(n_samples)
]
s = agent.replay_buffer.sample(n_samples)[0][:, 0, :]
# s = torch.tensor(np.array([env.observation_space.sample() for _ in range(n_samples)]))
t0 = time.perf_counter()
for i in range(100_000):
    a = agent.action(s)
    for j in range(n_samples):
        tmp_dict[j][a[j]] += 1
    # agent.learn()
print(time.perf_counter() - t0)
for j in range(n_samples):
    print(tmp_dict[j])
#%%
import timeit
runtimes = []
threads = [t for t in range(1, 17)]
for t in threads:
    torch.set_num_threads(t)
    r = timeit.timeit(
        setup = "import torch; x = torch.randn(2048, 2048); \
            y = torch.randn(2048, 2048)", stmt="torch.mm(x, y)",
        number=100)
    runtimes.append(r)
    
plt.style.use('seaborn')
plt.plot(threads, runtimes)