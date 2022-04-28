import sys
import gc
import importlib
import json
import os
import time

# ROOT = os.path.dirname(os.path.abspath(os.getcwd()))
# sys.path.append(os.path.join(ROOT, 'libs'))
sys.path.append('/Users/a1/Dev/DeepRL')
import gym
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output

import drl.agents as agents

#####################################################################
#%%
# importlib.reload(drl)
# writer = SummaryWriter('./runs/RAdam')
env = gym.make('LunarLander-v2')

# agent = agents.ExpectedAgent(#kappa=1,
#     input_dims=(8,), n_actions=4, gamma=0.99, n_steps=1,
#     epsilon_start=0., epsilon_end=0., lr=5e-4,
#     batch_size=64, mem_size=500_000, min_history=1_000, replace_target=100
# )
agent = agents.DQAgent(
    input_dims=(8,), n_actions=4, gamma=0.99, n_steps=1,
    epsilon=0., lr=5e-4,
    batch_size=64, mem_size=500_000, min_history=1_000, replace_target=100
)

n_games = 1000
scores = []
time_history = []
grads_history = []
frames_history = []
total_frames_observed = 0
#%%
for i in range(len(scores), n_games * 50):
    done = False
    score = 0
    observation = env.reset()
    t0 = time.time()
    
    observation_hist = []
    action_hist = []
    reward_hist = []
    done_hist = []

    while not done:
        # action = np.random.choice(env.action_space.n)
        action = agent.choose_action(observation)
            
        observation_, reward, done, _ = env.step(action)
        score += reward
        
        observation_hist.append(observation)
        action_hist.append(action)
        reward_hist.append(reward)
        done_hist.append(done)
    
        observation = observation_
        agent.learn()
    ##########
    agent.process_episode(observation_hist, action_hist, reward_hist, done_hist)
    ##########
    total_frames_observed += len(done_hist)
    frames_history.append(total_frames_observed)
    scores.append(score)
    time_history.append(time.time() - t0)
    grads_history.append(agent.grad_magnitude)
    
    avg_score = np.mean(scores[-100:])
    avg_time = np.mean(time_history[-10:])
    clear_output(wait=True)
    try:
        pass
        if i % 100 == 99:
            # pd.DataFrame(scores).rolling(100).mean().dropna().plot(grid=True)
            # plt.show()
            plt.close('all')
            fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True, figsize=(10, 10))
            pd.DataFrame(scores).rolling(100).mean().dropna().plot(ax=ax0, grid=True, title='Average score', label='score')
            pd.DataFrame(grads_history).rolling(100).mean().dropna().plot(ax=ax1, grid=True, title='Gradients magnitude', label='grads')
            plt.show()
    except:
        pass
    print(f'episode {i}  score={score:.1f}  avg_Score={avg_score:.1f}  ' + \
          f'frames={len(done_hist)}  avg_time={avg_time:.1f}  ' +\
          f'grads={agent.grad_magnitude:.1f}  ')# +\
          # f'dot={agent.features_alignment.item():.1f})'
    
    # if i % 10 == 9:
    #     df = pd.DataFrame({
    #         'frames_observed': frames_history,
    #         'scores': scores,
    #         'grads': grads_history
    #     })
    #     df.to_csv('../logs/test2.csv', index=False)
    #     with open('../logs/test2.txt', 'w') as f:
    #         json.dump(scores, f)
    
    # if (i + 1) % 1_000 == 0:
    #     agent.save_model("./dqn_model_{}.h5".format(len(scores)))
    
    # steps = agent.optimization_steps
    # if steps < 1000:
    #     continue
    # elif steps > 14_000_000:  # SGD(1e-4) baseline was run for this long
    #     break
    # writer.add_scalar('average score', avg_score, steps)
    # writer.add_scalar('loss', agent.avg_loss, steps)
    # writer.add_scalar('Q value', agent.avg_q_value, steps)
    # writer.add_scalar('Q update alignment', agent.q_update_alignment, steps)
    # writer.add_scalar('features alignment', agent.features_alignment, steps)
    # writer.add_scalar('gradient norm', agent.grad_magnitude, steps)
    # writer.add_scalar('weight norm', agent.weight_norm, steps)
    # writer.add_scalars('relative weight update', agent.relative_weight_updates, steps)
    # writer.add_scalars('per-layer gradient norm', agent.per_layer_grad_norm, steps)


 #%%
agent.batch_size = 1
agent.learn(debug=True)
agent.batch_size = 64
#%%
# agent.q_eval.save_weights(os.path.join(ROOT, 'models/ckpt.tf'))
#%%
with open('./test.txt', 'w') as f:
    json.dump(scores, f)
#%%

p = policies.EpsilonGreedyPolicy(0.1)
q = np.array(
    [
         [112.3, 111.0, 105.8, 113.0],
         [28, 25.1, 26.0, 18.9]
    ]
)
q = np.array([[112.3, 112.99, 105.8, 113.0]])
print(q)
print(p.distribution(q))
print(p.action(q))
# a = []
# t0 = time.time()
# for i in range(100000):
#     a.append(p.action(q))
# print(time.time() - t0)
#%%
for layer in agent.q_eval.layers:
    if 'dense' not in layer.name:
        continue
    layer.reset_noise()
    weights = layer.get_weights()
    w = weights[1]
    b = weights[3]
    print(
        '<|w|> = {:.3f} ± {:.3f};   <|b|> = {:.3f} ± {:.3f};'.format(
            np.mean(np.abs(w)), np.std(np.abs(w)), np.mean(np.abs(b)), np.std(np.abs(b))
        )
    )
#%%
# test run through environment
env = gym.make('LunarLander-v2')
done = False
score = 0
observation = env.reset()
t0 = time.time()
num_steps = 0

while not done:
    env.render(mode='human')
    action = agent.choose_action(observation)

    observation_, reward, done, _ = env.step(action)
    score += reward
    observation = observation_
    # agent.learn()
    num_steps += 1
print(score)
env.close()
env = gym.make('LunarLander-v2')
#%%
t0 = time.time()
for i in range(10_000):
    # agent.learn()
    # agent.q_eval(tf.constant(s), training=False)
    sub_model(tf.constant(s))
    # sub_model.predict_on_batch(s)
print(time.time() - t0)
#%%
batch_size = 64
input_shape = (8,)
n_actions = 4
s1 = np.random.random((1, *input_shape))
s64 = np.random.random((batch_size, *input_shape))

tf_model = drl2.TFModel(input_shape, n_actions, [128, 128])
torch_model = drl2.TorchModel(input_shape, n_actions, [128, 128])
#%%
# s1 = torch.randn((1, *input_shape))
# s64 = torch.randn((batch_size, *input_shape))
# s = np.random.random(8)

counter = 0
t0 = time.time()
# with torch.autograd.profiler.profile(profile_memory=True) as prof:
for i in range(10000):
    # agent.q_eval.reset_noise()
    # agent.choose_action(s)
    # agent.learn()
    # agent.q_eval(s64)
    # agent._update_network_parameters()
    agent.memory.sample_buffer(64)
    # agent.memory.mem_buffer.sample_buffer(64)
    # counter += set(np.random.choice(64*100, 64, replace=False)).__len__()
print(time.time() - t0)
#%%
tmp_dict = {i: 0 for i in range(4)}
s = agent.memory.sample_buffer(1)[0][:, 0, :]
t0 = time.time()
for i in range(100_000):
    # a = agent.choose_action(s)
    # tmp_dict[a] += 1
    agent.learn()
print(time.time() - t0)
print(tmp_dict)
    