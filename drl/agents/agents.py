"""Deep Reinforcement Learning Models and Agents

implementation using pytorch
"""

import copy
import sys
import os

import random
import numpy as np

import torch

from drl.estimators import DuelingDeepQNetwork
from drl.replay_buffers import (
    ReplayBuffer,
    NstepReplayBuffer,
    Prioritized
)
from drl.policies import EpsilonGreedyPolicy


class DQAgent:
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
        policy is optimal, which is not always the case. Can be used with
        e-greedy policy if epsilon is low enough;
    """

    def __init__(
        self,
        input_dims, n_actions, gamma, n_steps=1,
        epsilon=1e-2,  # no epsilon decay due to usage of NoisyNet
        lr=3e-4, batch_size=64, mem_size=1_000_000, min_history=100_000,
        replace_target=1000,
        fname='DQAgent_model.h5',
        device=None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.action_space = list(range(n_actions))
        self.n_actions = n_actions
        self.gamma = gamma              # Discount factor
        self.n_steps = n_steps

        self.epsilon = epsilon          # random action probability

        self.batch_size = batch_size
        self.min_history = min_history
        self.model_file = fname

        self.replace_target = replace_target
        self.replace_target_counter = 0
        self.target_policy = EpsilonGreedyPolicy(self.epsilon)
        self.behaviour_policy = EpsilonGreedyPolicy(self.epsilon)
        self.memory = NstepReplayBuffer(mem_size, input_dims, 2, mod='numpy')
        self.memory = Prioritized(self.memory)

        self.q_eval = DuelingDeepQNetwork(
            input_dims, n_actions, [128, 128]).to(self.device)
        self.q_target = DuelingDeepQNetwork(
            input_dims, n_actions, [128, 128]).to(self.device)
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.RAdam(self.q_eval.parameters(), lr=lr)

        self.grad_magnitude = 0

    def _reset_noise(self):
        # Resamples noise when working with NoisyDense networks
        self.q_eval.reset_noise()
        self.q_target.reset_noise()

    def _remove_noise(self):
        # Removes noise when working with NoisyDense networks
        self.q_eval.remove_noise()
        self.q_target.remove_noise()

    @torch.no_grad()
    def choose_action(self, observation):
        self.q_eval.reset_noise()
        self.q_eval.eval()
        q = np.array(self.q_eval(
            torch.as_tensor(observation[np.newaxis, :],
                            dtype=torch.float32, device=self.device)
        ).cpu().detach())[0]
        action = self.behaviour_policy.action(q)
        return action

    def process_episode(
        self, observations, actions, rewards, dones
    ):
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
        self.memory.store_transition(
            torch.as_tensor(np.array(states)),
            torch.as_tensor(np.array(actions)),
            torch.as_tensor(np.array(rewards))
        )

    def _update_network_parameters(self):
        # self.q_target.load_state_dict(self.q_eval.state_dict())

        st_target = self.q_target.state_dict()
        st_eval = self.q_eval.state_dict()
        for key in st_target.keys():
            if 'epsilon' in key:
                continue
            st_target[key] = (1 - 1 / self.replace_target) * st_target[key] + \
                (1 / self.replace_target) * st_eval[key]
        self.q_target.load_state_dict(st_target)

    def learn(self, debug=False):
        if len(self.memory) < self.min_history:
            return

        self._learn(debug=debug)

        self._update_network_parameters()
        # self.replace_target_counter += 1
        # if self.replace_target_counter % self.replace_target == 0:
        #     self._update_network_parameters()
        #     self.replace_target_counter = 0

    def _learn(self, debug=False):
        states, actions, rewards, sample_weights = \
            self.memory.sample(self.batch_size, beta=1.)

        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device)
        sample_weights = torch.as_tensor(
            sample_weights, dtype=torch.float32, device=self.device)

        self._reset_noise()
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

        # prod = self.q_eval.features(state) * self.q_eval.features(next_state)
        # dot_prod = prod.mean()

        q_current_eval = self.q_eval(state)
        q_next_eval = self.q_eval(next_state)
        q_next_target = self.q_target(next_state)
        _, next_best_action_eval = q_next_eval.detach().max(dim=1)
        v_next = q_next_target[batch_index, next_best_action_eval]

        delta = np.power(self.gamma, self.n_steps) * v_next * ~done + \
            reward - q_current_eval[batch_index, action]

        # loss = torch.mean(delta**2) / 4 + dot_prod / 10
        loss = delta**2
        self.memory.update_last_priorities(
            # (torch.abs(delta) + 1e-2).clip(0.1, 50.).detach().numpy()
            (torch.argsort(delta.abs()) + 1).cpu().detach().numpy()
        )
        loss = torch.mean(loss * sample_weights) / 4

        # print(loss)
        # print(dot_prod)
        # print("")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        grad_norm = 0
        for p in self.q_eval.parameters():
            grad_norm += p.grad.detach().data.norm(2).item()**2
        self.grad_magnitude = self.grad_magnitude * 0.95 + \
            0.05 * grad_norm**0.5
        #########################################################
        # q_current = self.q_eval(state)
        # q_next = self.q_eval(next_state)
        # v_next = np.power(self.gamma, self.n_steps) * \
        #     q_next[batch_index, next_action] * ~done

        # # v_next = self.q_target.value(next_state) * \
        # #     np.power(self.gamma, self.n_steps) * ~done

        # delta = reward + v_next - q_current[batch_index, action]
        # # Q_loss = torch.mean(delta ** 2) / 4
        # tau = 0.8
        # with torch.no_grad():
        #     c = (delta >= 0) * tau + (delta < 0) * (1 - tau)
        # Q_loss = torch.mean(c * (delta ** 2)) / c.abs().mean()

        # self.optimizer.zero_grad()
        # Q_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 35.)
        # self.optimizer.step()

        # grad_norm = 0
        # for p in self.q_eval.parameters():
        #     grad_norm += p.grad.detach().data.norm(2).item()**2
        # self.grad_magnitude = self.grad_magnitude * 0.95 + \
        #     0.05 * grad_norm**0.5

        # q_current = self.q_target(state)
        # v_current = self.q_eval.value(state)
        # delta = q_current[batch_index, action] - v_current
        # tau = 0.9
        # # (delta >= 0) * tau + (delta < 0) * (1 - tau)
        # V_loss = torch.mean(torch.abs((delta < 0) + (-tau)) * (delta ** 2))

        # self.optimizer.zero_grad()
        # V_loss.backward()
        # self.optimizer.step()

    def save_model(self, fname=None):
        if fname is None:
            fname = self.model_file
        torch.save(self.q_eval.state_dict(), fname)
        print(f'Saved PyTorch Model State to {fname}')

    def load_model(self, fname=None):
        if fname is None:
            fname = self.model_file
        self.q_eval.load_state_dict(torch.load(fname))
        self.q_target.load_state_dict(self.q_eval.state_dict())


class ExpectedAgent():
    """ ExpectedAgent class

    This class implements off-policy n-step Tree Backup Algorithm
    https://towardsdatascience.com/introduction-to-reinforcement-learning-rl-part-7-n-step-bootstrapping-6c3006a13265
    """

    def __init__(
        self,
        input_dims, n_actions, gamma, n_steps=1,
        epsilon_start=1., epsilon_dec=0.99, epsilon_end=0.01,
        lr=3e-4, batch_size=64, mem_size=1_000_000, min_history=10_000,
        replace_target=100,
        fname='dqn_model.pt',
        device=None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.action_space = list(range(n_actions))
        self.n_actions = n_actions
        self.gamma = gamma              # Discount factor
        self.n_steps = n_steps

        self.epsilon = epsilon_start    # Exploration rate
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end

        self.batch_size = batch_size
        self.min_history = min_history
        self.model_file = fname

        self.replace_target = replace_target
        self.replace_target_counter = 0
        # self.target_policy = GreedyPolicy()
        # self.behaviour_policy = GreedyPolicy()
        self.target_policy = EpsilonGreedyPolicy(self.epsilon_min)
        self.behaviour_policy = EpsilonGreedyPolicy(self.epsilon)
        self.memory = NstepReplayBuffer(mem_size, input_dims, self.n_steps + 1)

        self.q_eval = DuelingDeepQNetwork(
            input_dims, n_actions, [128, 128]).to(self.device)
        self.q_target = DuelingDeepQNetwork(
            input_dims, n_actions, [128, 128]).to(self.device)

        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.HuberLoss(delta=5.0)
        self.optimizer = torch.optim.RAdam(self.q_eval.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(self.q_eval.parameters(), lr=lr)

        # debug info
        self.optimization_steps = 0
        self.avg_loss = 0
        self.grad_magnitude = 0
        self.per_layer_grad_norm = {
            name: 0 for name, p in self.q_eval.named_parameters() if 'weight' in name}
        self.relative_weight_updates = {
            name: 0 for name, p in self.q_eval.named_parameters() if 'weight' in name}
        self.weight_norm = 0
        self.avg_q_value = 0
        self.q_update_alignment = 0
        self.features_alignment = 0

    def _reset_noise(self):
        # Resamples noise when working with NoisyDense networks
        self.q_eval.reset_noise()
        self.q_target.reset_noise()

    def _remove_noise(self):
        # Removes noise when working with NoisyDense networks
        self.q_eval.remove_noise()
        self.q_target.remove_noise()

    # @torch.no_grad()
    def choose_action(self, observation):
        with torch.no_grad():
            self.q_eval.reset_noise()
            self.q_eval.eval()
            q = np.array(self.q_eval(
                torch.as_tensor(observation[np.newaxis, :],
                                dtype=torch.float32, device=self.device)
            ).cpu().detach())[0]
        action = self.behaviour_policy.action(q)
        return action

    def process_episode(
        self, observation_history, action_history,
        reward_history, done_history
    ):
        """Creates trajectories and puts them into replay buffer
        """

        for idx in range(0, len(observation_history)):
            idx_next = idx + self.n_steps + 1

            states = observation_history[idx:idx_next]
            actions = action_history[idx:idx_next]
            rewards = reward_history[idx:idx_next]

            while len(states) < self.n_steps + 1:
                states += [np.zeros(states[0].shape)]
                actions += [actions[-1]]
                rewards += [-np.inf]

            self.remember(states, actions, rewards)

    def remember(self, states, actions, rewards):
        self.memory.store_transition(
            np.array(states), np.array(actions), np.array(rewards))

    def _update_network_parameters(self):
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def learn(self, debug=False):
        if self.memory.mem_cntr < self.min_history:
            return

        self._learn(debug=debug)

        self.replace_target_counter += 1
        if self.replace_target_counter % self.replace_target == 0:
            self._update_network_parameters()
            self.replace_target_counter = 0

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
            self.epsilon_min else self.epsilon_min
        self.behaviour_policy.update(self.epsilon)

    def _learn(self, debug=False):
        # n_step_estimation
        # rollout

        states, actions, rewards = \
            self.memory.sample(self.batch_size)
        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device)

        # Calculate Q values for whole trajectory in one pass
        with torch.no_grad():
            self._reset_noise()
            self.q_eval.eval()
            self.q_target.eval()
            concatenated_states = torch.concat(
                [states[:, i] for i in range(states.shape[1])]
            )
            concatenated_q_eval = self.q_eval(concatenated_states)
            concatenated_q_eval = [
                concatenated_q_eval[self.batch_size *
                                    i: self.batch_size * (i + 1)][:, None, :]
                for i in range(states.shape[1])
            ]
            q_evals = torch.concat(concatenated_q_eval, dim=1)

            concatenated_q_target = self.q_target(concatenated_states)
            concatenated_q_target = [
                concatenated_q_target[self.batch_size *
                                      i: self.batch_size * (i + 1)][:, None, :]
                for i in range(states.shape[1])
            ]
            q_targets = torch.concat(concatenated_q_target, dim=1)
            # Q values have been calculated
            #
            q_target = q_evals[:, 0].detach().clone()
            batch_index = torch.arange(self.batch_size, dtype=torch.long)
            q_target[batch_index, actions[:, 0]] = self._get_n_step_estimation(
                q_evals, q_targets, actions, rewards, debug=debug)

            # calculating features alignment for debug
            curr_features = self.q_eval.features(states[:, 0, :])
            next_features = self.q_eval.features(states[:, 1, :])
            prod = (curr_features * next_features).sum() / self.batch_size
            self.features_alignment = 0.95 * self.features_alignment + 0.5 * prod

        if debug:
            print('q_eval ', q_evals[:, 0])
            print('target dist    ', self.target_policy.distribution(
                q_evals[:, 0].detach().numpy()))
            # print('behaviour dist ', self.behaviour_policy.distribution(q_evals[:, 0].detach().numpy()))
            # print('boltzmann dist ', BoltzmannPolicy(1).distribution(q_evals[:, 0]))
            print(f'actions={actions[:, 0].item()}   rewards={rewards[:, 0].item():.3f}')
            print('q_target', q_target)
        # if not debug:
        _ = self._train_on_batch(x=states[:, 0], y=q_target, debug=debug)

    def _get_n_step_estimation(  # TODO: get rid of numpy dependency (Policies)
        self, q_evals, q_targets, actions, rewards, debug
    ):
        # Calculates n-step expected return
        # sigma = np.linspace(1, 0, self.n_steps + 1)[1:]
        # zeros yield full tree backup
        sigma = torch.zeros(
            self.n_steps, dtype=q_targets.dtype, device=q_targets.device)
        batch_index = torch.arange(
            self.batch_size, dtype=torch.long, device=q_targets.device)
        for i in reversed(range(q_targets.shape[1] - 1)):
            # update Q_i(a_i) = r_i + gamma * E[Q_i+1]
            # probs = self.target_policy.distribution(q_evals[:, i + 1])
            probs = torch.as_tensor(
                self.target_policy.distribution(
                    q_evals[:, i + 1].detach().numpy()),
                dtype=q_targets.dtype, device=q_targets.device
            )

            probs_sarsa = torch.zeros_like(probs)
            probs_sarsa[batch_index, actions[:, i + 1]] = 1
            # sigma = 1  # 0 -> tree backup, 1 -> sarsa
            probs = (1 - sigma[i]) * probs + sigma[i] * probs_sarsa

            Q = q_targets[:, i + 1]
            done = torch.isinf(rewards[:, i + 1])
            V_next = self.gamma * torch.sum(probs * Q, dim=-1) * (~done)

            Q_i = rewards[:, i] + V_next
            Q_i[torch.isinf(Q_i)] = 0  # to prevent propagating dummy rewards

            q_targets[batch_index, i, actions[:, i]] = Q_i

        return q_targets[batch_index, 0, actions[:, 0]]

    def _train_on_batch(self, x, y, debug=False):
        self.q_eval.train()

        self.optimizer.zero_grad()
        y_pred = self.q_eval(x)
        loss = self.loss(y_pred, y)
        loss.backward()

        pre_step_state = copy.deepcopy(
            self.q_eval.state_dict())  # temporary variable
        # tmp_weight = self.q_eval.V.weight.detach().clone()
        # print(self.q_eval.dense0.weight[:, 0:10])
        # print(self.q_eval.dense0.weight.grad[:, 0:10])
        if not debug:
            self.optimizer.step()
        # print(self.q_eval.dense0.weight[:, 0:10])
        # print('\n')
        # print((self.q_eval.V.weight.detach() - tmp_weight) / tmp_weight)
        # print('\n')

        if debug:
            print(f'loss={loss}')

        # optional gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 35.)
        # for p in self.q_eval.parameters():
        #     torch.nn.utils.clip_grad_norm_(p, 10.)

        # LOGGING
        alpha = 0.95  # exponential smoothing for stochastic statistics

        self.optimization_steps += 1
        self.avg_loss = self.avg_loss * alpha + (1 - alpha) * loss.item()
        # gradients norm for logging
        grad_norm = 0
        weight_norm = 0
        for name, p in self.q_eval.named_parameters():
            # gradients related stuff
            if p.grad is not None and p.requires_grad:
                p_grad_norm = p.grad.detach().data.norm(2).item()
                grad_norm += p_grad_norm ** 2
                if 'weight' in name:
                    self.per_layer_grad_norm[name] = self.per_layer_grad_norm[name] * alpha + (
                        1 - alpha) * p_grad_norm
            # weight related stuff
            if 'weight' in name:
                relative_change = (self.q_eval.state_dict()[
                                   name] - pre_step_state[name]) / pre_step_state[name]
                relative_change = relative_change.abs().mean().item()
                self.relative_weight_updates[name] = self.relative_weight_updates[name] * alpha + (
                    1 - alpha) * relative_change
                weight_norm += self.q_eval.state_dict()[name].norm(2) ** 2
        grad_norm = grad_norm ** 0.5
        weight_norm = weight_norm ** 0.5
        self.grad_magnitude = self.grad_magnitude * \
            alpha + (1 - alpha) * grad_norm
        self.weight_norm = self.weight_norm * alpha + (1 - alpha) * weight_norm

        with torch.no_grad():
            # average q value
            avg_q = y_pred.max(dim=1).values.mean().item()
            self.avg_q_value = self.avg_q_value * alpha + (1 - alpha) * avg_q

            # Q values before-after alignment
            a = self.q_eval(x) - y_pred
            b = y - y_pred
            cos_sim = torch.nn.CosineSimilarity(dim=1)(a, b).mean().item()
            self.q_update_alignment = alpha * \
                self.q_update_alignment + (1 - alpha) * cos_sim

            # feature alignment
            # self.features_alignment = 0

        return loss

    def save_model(self, fname=None):
        if fname is None:
            fname = self.model_file
        torch.save(self.q_eval.state_dict(), fname)
        print(f'Saved PyTorch Model State to {fname}')

    def load_model(self, fname=None):
        if fname is None:
            fname = self.model_file
        # self.q_eval = NeuralNetwork()
        self.q_eval.load_state_dict(torch.load(fname))
        self._update_network_parameters()
        self.epsilon = self.epsilon_min
