"""Deep Q Learning on steroids

implementation using pytorch
"""

import copy

import gym
import numpy as np
import torch

from drl.agents.base_agent import BaseAgent
from drl.estimators import DuelingDeepQNetwork
from drl.optimizers import (
    NGD,
    AdaHessian,
    KFAC,
    EKFAC
)
from drl.replay_buffers import (
    Buffer,
    ReplayBuffer,
    NstepReplayBuffer,
    Prioritized
)
from drl.policies import (
    Policy,
    GreedyPolicy,
    EpsilonGreedyPolicy,
    BoltzmannPolicy
)


class DQAgent(BaseAgent):
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
        *,
        env: gym.Env = None,
        env_fn: callable = None,
        observation_space: gym.spaces.Space = None,
        action_space: gym.spaces.Space = None,

        estimator: torch.nn.Module = None,
        noisy: bool = True,
        noisy_use_factorized: bool = True,
        parametrize: bool = True,

        behaviour_policy: Policy = None,
        target_policy: Policy = None,
        epsilon: float = 1e-2,  # For exploration. No decay due to NoisyNet

        replay_buffer: Buffer = None,
        mem_size: int = 1_000_000,
        min_history: int = 10_000,
        batch_size: int = 64,
        n_steps: int = 1,
        replace_target: int = 100,
        gamma: float = 0.99,
        lr: float = 3e-4,

        device: str = None,
        fname: str = 'DQAgent_model.h5',
    ):
        super().__init__(env=env, env_fn=env_fn,
                         observation_space=observation_space,
                         action_space=action_space)

        self.replay_buffer = replay_buffer if replay_buffer is not None else \
            NstepReplayBuffer(mem_size, self.observation_space.shape, 1)

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
            BoltzmannPolicy(epsilon)
        self.target_policy = target_policy if target_policy else \
            BoltzmannPolicy(epsilon)

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
        # self.optimizer = AdaHessian(self.q_eval.parameters(), lr)
        # self.optimizer = torch.optim.AdamW([
        #     {
        #         'params': [p for name, p in self.q_eval.named_parameters()
        #                    if 'sigma' not in name],
        #         'lr': lr,
        #         'weight_decay': 1e-2
        #     },
        #     {
        #         'params': [p for name, p in self.q_eval.named_parameters()
        #                    if 'sigma' in name],
        #         'lr': lr,
        #         'weight_decay': 0
        #     },
        # ])
        # self.preconditioner = KFAC(self.q_eval, 0.1)
        # self.preconditioner = KFAC(
        #     [p for name, p in self.q_eval.named_parameters()
        #      if 'sigma' not in name], 0.1
        # )

    @torch.no_grad()
    def action(self, observation):
        self.q_eval.reset_noise()
        self.q_eval.eval()

        obs = torch.as_tensor(
            np.array(observation, copy=False), device=self.device)
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
            for i, r in enumerate(my_rewards[:-1]):
                if np.isinf(r):
                    break
                n_step_reward += np.power(self.gamma, i) * r

            my_states = [my_states[0], my_states[-1]]
            my_actions = [my_actions[0], my_actions[-1]]
            my_rewards = [n_step_reward, my_rewards[-1]]
            # Last reward is used as a proxy for "done" flag
            # reward == -np.inf indicates that done==True

            self.remember(my_states, my_actions, my_rewards)

    def remember(self, states, actions, rewards):
        self.replay_buffer.append((
            np.array(states),
            np.array(actions),
            np.array(rewards)
        ))

    def _sample_minibatch_from_replay_buffer(self):
        samples = self.replay_buffer.sample(self.batch_size)
        keys = ['states', 'actions', 'rewards', 'weights']
        dtypes = [torch.float32, torch.long, torch.float32, torch.float32]
        batch = dict()

        for data, key, dtype in zip(samples, keys, dtypes):
            batch[key] = torch.as_tensor(
                data, dtype=dtype, device=self.device)

        return batch

    def learn(self, debug=False):
        if len(self.replay_buffer) < self.min_history:
            return
        debug_info = self._learn(debug=debug)
        self._update_network_parameters()
        return debug_info

    def _learn(self, debug=False):
        batch = self._sample_minibatch_from_replay_buffer()

        self.q_eval.reset_noise()
        self.q_target.reset_noise()

        self.q_eval.train()
        self.q_target.train()

        state = batch['states'][:, 0, :]
        action = batch['actions'][:, 0]
        reward = batch['rewards'][:, 0]
        next_state = batch['states'][:, 1, :]
        next_action = batch['actions'][:, 1]
        done = torch.isinf(batch['rewards'][:, 1])
        batch_index = torch.arange(
            self.batch_size, dtype=torch.long, device=state.device)

        q_current_eval, features_current = self.q_eval.q_and_features(state)
        # self.q_eval.reset_noise()
        # """Standart v_next with target network
        with torch.no_grad():
            q_next_eval, features_next = self.q_eval.q_and_features(next_state)
            q_next_target = self.q_target(next_state)
            v_next = (q_next_target * self.target_policy.probs(q_next_eval)).sum(1)
        # """
        # q_next_eval, features_next = self.q_eval.q_and_features(next_state)
        # v_next = (q_next_eval * self.target_policy.probs(q_next_eval)).sum(1)

        delta = np.power(self.gamma, self.n_steps) * v_next * ~done + \
            reward - q_current_eval[batch_index, action]

        # loss = torch.mean(delta**2) / 4 + dot_prod / 10
        loss = delta**2
        if 'weights' in batch:
            loss *= batch['weights']
            self.replay_buffer.update_last_priorities(
                # (torch.abs(delta) + 1e-2).clip(0.1, 50.).detach().numpy()
                (torch.argsort(delta.abs()) + 1).cpu().detach().numpy()
            )
        loss = loss.mean() / 4

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 10.)
        # self.preconditioner.step()
        self.optimizer.step()

        # CONSTRUCTING DEBUG INFO
        debug_info = None
        if debug:
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

        # for p in self.q_eval.parameters():
        #     p.grad = None
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
