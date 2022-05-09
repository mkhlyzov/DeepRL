"""Deep Q Learning on steroids

implementation using pytorch
"""

import copy

from gym.spaces import Space
import numpy as np
import torch

from drl.estimators import DuelingDeepQNetwork
from drl.replay_buffers import (
    ReplayBuffer,
    NstepReplayBuffer,
    Prioritized
)
from drl.policies import (
    GreedyPolicy,
    EpsilonGreedyPolicy,
    BoltzmannPolicy
)


class DQAgent(object):
    """Deep Q Agent class.

    This class implements deep Q-learning algorithm enhanced with following
    mechanisms:
    - Double Q learning
        uses two networks, regular one for acting and learning, and target
        network for assessing future Q values;
    - Dual Q learning
        devides Q network's flow into two: value and advantage;
    - N-step Q learning
        naive approach, unfolds Bellman equation n steps forward. Same as used
        in Rainbow algorithm. This approach only works under assumption that
        policy that collected the data is the optimized one, which is not
        always the case. Can be used with e-greedy policy if epsilon is low
        enough;
    - NoisyNetwork
        p -> µ + σ•ε, where µ and σ are learned parameters, ε ~ N(0, 1)
    """

    def __init__(
        self,
        env=None, env_fn=None, observation_space=None, action_space=None,

        estimator=None,
        noisy=True,
        noisy_use_factorized=False,
        parametrize=False,

        behaviour_policy=None,
        target_policy=None,
        epsilon=1e-2,  # For exploration. No decay due to NoisyNet

        replay_buffer=None,
        mem_size=1_000_000, min_history=10_000, batch_size=64,
        lr=3e-4, gamma=0.99, n_steps=1, replace_target=100,

        optimizer_params={},

        device=None,
        fname='DQAgent_model.h5',
    ):
        if env:
            if env_fn or observation_space or action_space:
                raise ValueError('Too many env arguments')
            self._assert_env_is_valid(env)
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        elif env_fn:
            if observation_space or action_space:
                raise ValueError('Too many env arguments')
            self._assert_envfn_is_valid(env_fn)
            env = env_fn()
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        elif observation_space and action_space:
            self._assert_specs_are_valid(observation_space, action_space)
            self.observation_space = observation_space
            self.action_space = action_space
        else:
            raise AttributeError('Have to pass info about environment')

        self.memory = replay_buffer

        self.memory = replay_buffer if replay_buffer is not None else \
            NstepReplayBuffer(mem_size, self.env.observation_space.shape, 1)

        self.min_history = min_history
        self.n_steps = n_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.model_file = fname

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.behaviour_policy = behaviour_policy if behaviour_policy else \
            EpsilonGreedyPolicy(epsilon)
        self.target_policy = target_policy if target_policy else \
            EpsilonGreedyPolicy(epsilon)

        self.q_eval = DuelingDeepQNetwork(
            input_dims=self.observation_space.shape,
            n_actions=self.action_space.n,
            neurons=[128, 128],
            noisy=noisy,
            factorised=noisy_use_factorized,
            parametrize=parametrize
        ).to(self.device)
        self.q_target = copy.deepcopy(self.q_eval)
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr)

    def _assert_env_is_valid(self, env):
        if not hasattr(env, 'reset'):
            raise ValueError('No reset method in {env}')
        if not hasattr(env, 'step'):
            raise ValueError('No step method in {env}')
        if not hasattr(env, 'close'):
            raise ValueError('No close method in {env}')
        if not hasattr(env, 'action_space'):
            raise ValueError('No action_space attribute in {env}')
        if not hasattr(env, 'observation_space'):
            raise ValueError('No action_space attribute in {env}')

    def _assert_envfn_is_valid(self, env_fn):
        assert callable(env_fn)
        self._assert_env_is_valid(env_fn())

    def _assert_specs_are_valid(self, observation_space, action_space):
        if not isinstance(observation_space, Space):
            raise ValueError('observation_space bad value')
        if not isinstance(action_space, Space):
            raise ValueError('action_space bad value')

    @torch.no_grad()
    def action(self, observation):
        self.q_eval.reset_noise()
        self.q_eval.eval()

        obs = torch.tensor(np.array(observation), device=self.device)
        if obs.shape == self.observation_space.shape:
            obs = obs[None, :]
            q_values = self.q_eval(obs).squeeze()
        else:
            q_values = self.q_eval(obs)
        action = self.behaviour_policy.action(q_values)
        return action.cpu().numpy()

    def process_trajectory(self, observations, actions, rewards, dones):
        """Creates trajectories and puts them into replay buffer
        """
        for idx in range(0, len(observations)):
            idx_next = idx + self.n_steps + 1

            my_states = observations[idx:idx_next]
            my_actions = actions[idx:idx_next]
            my_rewards = rewards[idx:idx_next]

            while len(my_states) < self.n_steps + 1:
                my_states += [np.zeros(my_states[0].shape)]
                my_actions += [my_actions[-1]]
                my_rewards += [-np.inf]

            n_step_reward = 0
            for i in range(self.n_steps):
                if np.isinf(my_rewards[i]):
                    break
                n_step_reward += np.power(self.gamma, i) * my_rewards[i]

            my_states = [my_states[0], my_states[-1]]
            my_actions = [my_actions[0], my_actions[-1]]
            my_rewards = [n_step_reward, my_rewards[-1]]
            # last reward is used as a proxy for "done" flag

            self.remember(my_states, my_actions, my_rewards)

    def remember(self, states, actions, rewards):
        self.memory.append(
            torch.as_tensor(np.array(states)),
            torch.as_tensor(np.array(actions)),
            torch.as_tensor(np.array(rewards))
        )

    def learn(self, debug=False):
        if len(self.memory) < self.min_history:
            return
        debug_info = self._learn(debug=debug)
        self._update_network_parameters()
        return debug_info

    def _learn(self, debug=False):
        states, actions, rewards = \
            self.memory.sample(self.batch_size)

        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device)
        # sample_weights = torch.as_tensor(
        #     sample_weights, dtype=torch.float32, device=self.device)

        self.q_eval.reset_noise()
        self.q_target.reset_noise()

        self.q_eval.eval()
        self.q_target.eval()

        state = states[:, 0, :]
        action = actions[:, 0].to(dtype=torch.long)
        reward = rewards[:, 0]
        next_state = states[:, 1, :]
        next_action = actions[:, 1].to(dtype=torch.long)
        done = torch.isinf(rewards[:, 1])
        batch_index = torch.arange(
            self.batch_size, dtype=torch.long, device=state.device)

        q_current_eval, features_current = self.q_eval.q_and_features(state)
        # self.q_eval.reset_noise()
        q_next_eval, features_next = self.q_eval.q_and_features(next_state)
        q_next_target = self.q_target(next_state)
        _, next_best_action_eval = q_next_eval.detach().max(dim=1)
        v_next = q_next_target[batch_index, next_best_action_eval]

        delta = np.power(self.gamma, self.n_steps) * v_next * ~done + \
            reward - q_current_eval[batch_index, action]

        # loss = torch.mean(delta**2) / 4 + dot_prod / 10
        loss = delta**2
        # self.memory.update_last_priorities(
        #     # (torch.abs(delta) + 1e-2).clip(0.1, 50.).detach().numpy()
        #     (torch.argsort(delta.abs()) + 1).cpu().detach().numpy()
        # )
        loss = torch.mean(loss) / 4

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 50.)
        self.optimizer.step()

        # CONSTRUCTING DEBUG INFO
        with torch.no_grad():
            debug_info = {}
            grad_norm = 0
            weight_norm = 0
            for name, p in self.q_eval.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grad_norm += p.grad.detach().data.norm(2).item()**2
                if 'weight' in name:
                    weight_norm += p.norm(2).detach().item()**2
            debug_info['weight_norm'] = weight_norm**0.5
            debug_info['grad_norm'] = grad_norm**0.5
            debug_info['loss'] = loss.detach().item()
            debug_info['features_dot'] = torch.mean(
                features_current * features_next).detach().item()
            debug_info['features_cos'] = torch.nn.CosineSimilarity(dim=1)(
                features_current, features_next).mean().detach().item()

        return debug_info

    def _update_network_parameters(self):
        tau = 1 / self.replace_target
        state_target = self.q_target.state_dict()
        state_eval = self.q_eval.state_dict()
        for key in state_target.keys():
            if 'epsilon' in key:
                continue
            state_target[key] = tau * state_eval[key] + \
                (1 - tau) * state_target[key]

        self.q_target.load_state_dict(state_target)

    def save_model(self, fname=None):
        fname = fname if fname is not None else self.model_file
        torch.save(self.q_eval.state_dict(), fname)
        print(f'Saved PyTorch Model\'s State Dictionary to {fname}')

    def load_model(self, fname=None):
        fname = fname if fname is not None else self.model_file
        self.q_eval.load_state_dict(torch.load(fname))
        self.q_target.load_state_dict(self.q_eval.state_dict())
