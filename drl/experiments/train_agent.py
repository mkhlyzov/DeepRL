import logging
import pathlib
import time

from gym.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
import pandas as pd

import drl.experiments as experiments
import drl.utils as utils
from drl.utils import GET_TIME


class Trainer_old(object):
    def __init__(
        self,
        agent: object = None,
        env_fn: callable = None,
        samples_per_update: int = 8,
        metrics: list = [],
        verbose: int = 1,
        log_dir: str = None
    ):
        """
        Args:
            agent: \
            env_fn (lambda): Function that returns environment instance.
            num_episodes (int or float('inf')): unbounded by number of episodes \
                if float('inf').
            num_steps (int or float('inf')): number of interactions with \
                environment. Unbounbed if float('inf').
            samples_per_update (int): Environment steps per optimization step.
            metrics (list(string)): metrics to consider when outputting \
                agent's performanse.
            verbose (int): Amount of output desired.
                0: no output at all,
                1: prints intermediate results to terminal,
                2: additionally saves intermediate results to csv on disc,
                3: additionally saves intermediate results to tensorboard.
        """
        self.logger = logging.getLogger(__name__)

        self.agent = agent
        self.env_fn = env_fn
        self.samples_per_update = samples_per_update
        self.verbose = verbose

        self.log_dir = log_dir
        if log_dir is not None:
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._supported_metrics = [
            'loss', 'grad_norm', 'weight_norm',
            # 'q_update_alignment', 'avg_q_value',
            'features_dot', 'features_cos'
        ]
        self.metrics = metrics
        if metrics == 'all':
            self.metrics = self._supported_metrics
        self.alpha = 0.05  # EMA smoothing coefficient for metrics
        self.reset()

    def reset(self):
        if self.log_dir and self.to_csv:
            self.report_file = pathlib.Path(self.log_dir).joinpath(
                str(int(time.time())) + '.csv')

        self.env_steps_taken = 0
        self.episodes_played = 0
        self.optimization_steps_taken = 0

        self.train_scores = []
        self.frames_per_episode = []
        self.time_per_env_step = 0

        self.debug_info = {}
        self.report_info = {
            'env_steps': [],
            'optimization_steps': [],
            'episodes': [],
            'train_score': [],
            'eval_score': [],
            'time_per_env_step': [],
        }
        for key in self.metrics:
            self.debug_info[key] = 0
            self.report_info[key] = []

    def train(
        self,
        agent: object = None,
        env_fn: callable = None,
        num_episodes: int = float('inf'),
        num_steps: int = float('inf'),
        eval_freq: int = float('inf'),
        report_freq: int = float('inf'),
        eval_episodes: int = float('inf'),
        eval_steps: int = float('inf'),
        no_ops_evaluation: int = 0,
        reset: bool = False,
        plot: bool = True,
        to_csv: bool = False
    ):
        """Train agent on self.env

        Args:
            reset (bool): Flag to reset all previous training history and train \
                from scratch. Persists state between subsequent calls by default.
        """
        if agent is not None:
            self.agent = agent
        if env_fn is not None:
            self.env_fn = env_fn
        if (reset is True) or (self.env_steps_taken == 0):
            self.reset()
            self.logger.info('Starting training procedure from scratch.')
        else:
            self.logger.info('Continueing training from last step.')
        env = self.env_fn()

        t0 = GET_TIME()
        observation = env.reset()
        self._on_episode_start()

        while (
            self.env_steps_taken < num_steps
            and self.episodes_played < num_episodes
        ):
            action = self.agent.action(observation)

            observation_, reward, done, _ = env.step(action)
            self.score += reward

            self.observation_hist.append(observation)
            self.action_hist.append(action)
            self.reward_hist.append(reward)
            self.done_hist.append(done)

            observation = observation_

            self.env_steps_taken += 1
            if self.env_steps_taken % self.samples_per_update == 0:
                debug_info = self.agent.learn(debug=True)
                self.optimization_steps_taken += 1
                self._process_debug_info(debug_info)
            if done:
                # RESET ENVIRONMENT AND GO ON WITH TRAINING
                self._on_episode_end()
                observation = env.reset()
                self._on_episode_start()
            if self.env_steps_taken % eval_freq == 0:
                # Any number % float('inf') returns that number
                pass
                # EVALUATE PERFORMANCE, SNAPSHOT metrics.
                # ALL CALCULATIONS HERE
                self.time_per_env_step = (GET_TIME() - t0)  # / eval_freq
                self._eval(eval_episodes, eval_steps, no_ops_evaluation)
                self._print_latest_statistics()
                t0 = GET_TIME()

                if to_csv:
                    self.to_csv(path=self.report_file)

            if self.env_steps_taken % report_freq == 0:
                # Any number % float('inf') returns that number
                pass
                # PLOT PERFORMANCE, DUMP RESULTS ON DISC.
                # NO CALCULATIONS HERE
                if to_csv:
                    self.to_csv(path=self.report_file)
                if plot:
                    self.plot()
        env.close()

    def _on_training_start(self):
        pass

    def _on_episode_start(self):
        self.score = 0
        self.env_step_start = self.env_steps_taken
        self.observation_hist = []
        self.action_hist = []
        self.reward_hist = []
        self.done_hist = []

    def _on_episode_end(self):
        self.episodes_played += 1
        self.train_scores.append(self.score)
        self.frames_per_episode.append(
            self.env_steps_taken - self.env_step_start)

        self.agent.process_trajectory(
            self.observation_hist,
            self.action_hist,
            self.reward_hist,
            self.done_hist
        )

    def _process_debug_info(self, debug_info):
        if debug_info is None:
            return
        for key in self.metrics:
            self.debug_info[key] = debug_info[key] * self.alpha + \
                self.debug_info[key] * (1 - self.alpha)

    def _eval(self, num_episodes, num_steps, no_ops):
        eval_scores = experiments.evaluate_agent(
            self.agent, self.env_fn(),
            num_episodes=num_episodes,
            num_steps=num_steps,
            no_ops=no_ops
        )
        # CONSTRUCT report_info FROM debug_info
        window = min(len(self.train_scores), 100)
        self.report_info['env_steps'].append(self.env_steps_taken)
        self.report_info['optimization_steps'].append(
            self.optimization_steps_taken)
        self.report_info['episodes'].append(self.episodes_played)
        self.report_info['train_score'].append(
            sum(self.train_scores[-window:]) / window)
        self.report_info['eval_score'].append(
            sum(eval_scores) / len(eval_scores))
        self.report_info['time_per_env_step'].append(self.time_per_env_step)

        for key in self.metrics:
            self.report_info[key].append(self.debug_info[key])

    def _print_latest_statistics(self):
        """Prints the most recent statistics (and metrics)

        Outputs:
        #step: int, train_score: int, test_score: int,
        frames: int, avg_time: float,
        custom metrics(?):
            weights_norm, features_alignment, avg_q_value,
            q_update_alignment, gradient_norm
        """
        def pretty_int(i):
            if i > 1_000:
                return str(i // 1_000) + 'k'
            else:
                return str(i)

        if len(self.report_info['eval_score']) == 0:
            self.logger.info('No data to print. Try again later.')

        window = min(len(self.frames_per_episode), 100)
        fpe = sum(self.frames_per_episode[-window:]) // window

        s = f'env_step {pretty_int(self.report_info["env_steps"][-1])}   ' + \
            f'e={self.report_info["episodes"][-1]}  ' + \
            f'train_score={self.report_info["train_score"][-1]:.1f}   ' + \
            f'eval_score={self.report_info["eval_score"][-1]:.1f}   ' + \
            f'frames={fpe}  ' + \
            f'time_taken={self.report_info["time_per_env_step"][-1]:.1f}'
        self.logger.info(s)

    def to_csv(self, path):
        # Create parent directory if not exists
        pathlib.Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.report_info)
        df.to_csv(path, index=False)

    def plot(self):
        plot_data = {
            'samples': self.report_info['env_steps'],
            'train_score': self.report_info['train_score'],
            'eval_score': self.report_info['eval_score'],
        }
        utils.plot(plot_data, 'samples')


class Trainer(object):
    def __init__(
        self,
        agent: object = None,
        env_fn: callable = None,
        samples_per_update: float = 1,
        metrics: list = [],
        num_envs: int = 8,
        log_dir: str = None,
        multiprocessing=False,
    ):
        """
        Args:
            agent: \
            env_fn (lambda): Function that returns environment instance.
            num_episodes (int or float('inf')): unbounded by number of \
                episodes if float('inf').
            num_steps (int or float('inf')): number of interactions with \
                environment. Unbounbed if float('inf').
            samples_per_update (int): Environment steps per optimization step.
            metrics (list(string)): metrics to consider when outputting \
                agent's performanse.
        """
        self.logger = logging.getLogger(__name__)
        self.log_dir = log_dir

        self.agent = agent
        env_fns = [env_fn for _ in range(num_envs)]
        ENV_CONSTRUCTOR = AsyncVectorEnv if multiprocessing else SyncVectorEnv
        self.train_env = ENV_CONSTRUCTOR(env_fns)
        self.eval_env = ENV_CONSTRUCTOR(env_fns)
        self.samples_per_update = samples_per_update

        self._supported_metrics = [
            'loss', 'grad_norm', 'weight_norm',
            # 'q_update_alignment', 'avg_q_value',
            'features_dot', 'features_cos'
        ]
        self.metrics = metrics
        if metrics == 'all':
            self.metrics = self._supported_metrics
        else:
            self.metrics = [m for m in metrics if m in self._supported_metrics]
        self.alpha = 0.05  # EMA smoothing coefficient for metrics rolling
        self.reset()

    def __del__(self):
        del self.train_env
        del self.eval_env

    def reset(self):
        if self.to_csv:
            log_dir = self.log_dir if self.log_dir else pathlib.Path.cwd()
            self.report_file = pathlib.Path(log_dir).joinpath(
                f'{int(time.time())}.csv')

        self.env_steps_taken = 0
        self.episodes_played = 0
        self.optimization_steps_taken = 0

        self.train_scores = []
        self.frames_per_episode = []
        self.time_per_env_step = 0

        self.debug_info = {}
        self.report_info = {
            'env_steps': [],
            'optimization_steps': [],
            'episodes': [],
            'train_score': [],
            'eval_score': [],
            'time_per_env_step': [],
        }
        for key in self.metrics:
            self.debug_info[key] = 0
            self.report_info[key] = []

    def train(
        self,
        *,
        num_episodes: int = float('inf'),
        num_steps: int = float('inf'),
        eval_freq: int = float('inf'),
        report_freq: int = float('inf'),
        eval_episodes: int = float('inf'),
        eval_steps: int = float('inf'),
        no_ops_evaluation: int = 0,
        reset: bool = False,
        plot: bool = True,
        to_csv: bool = False,
        sleep_after_evaluation: int = 0,
    ):
        """Train agent on self.env

        Args:
            reset (bool): Flag to reset all previous training history and train
                from scratch. Persistant state between subsequent calls by
                default.
        """
        self.eval_freq = eval_freq
        self.report_freq = report_freq
        if (reset is True) or (self.env_steps_taken == 0):
            self.reset()
            self.logger.info('Starting training procedure from scratch.')
        else:
            self.logger.info('Continueing training from last step.')

        t0 = GET_TIME()
        observation = self.train_env.reset()
        self._on_episode_start()

        while (
            self.env_steps_taken < num_steps
            and self.episodes_played < num_episodes
        ):
            action = self.agent.action(observation)

            observation_, reward, done, _ = self.train_env.step(action)
            self.score += reward
            self._update_history(observation, action, reward, done)
            observation = observation_

            self.env_steps_taken += self.train_env.num_envs

            for i, d in enumerate(done):
                if d:
                    self._on_episode_end(i)
                    self._on_episode_start(i)

            while (self._if_should_learn()):
                debug_info = self.agent.learn(debug=True)
                self.optimization_steps_taken += 1
                self._process_debug_info(debug_info)

            if self._if_should_evaluate():
                # EVALUATE PERFORMANCE, SNAPSHOT metrics.
                # ALL CALCULATIONS HERE
                self.time_per_env_step = GET_TIME() - t0
                self._eval(eval_episodes, eval_steps, no_ops_evaluation)
                self._print_latest_statistics()
                if to_csv:
                    self.to_csv(path=self.report_file)
                time.sleep(sleep_after_evaluation)
                t0 = GET_TIME()

            if self._if_should_report():
                # PLOT PERFORMANCE, DUMP RESULTS ON DISC.
                # NO CALCULATIONS HERE
                if to_csv:
                    self.to_csv(path=self.report_file)
                if plot:
                    self.plot()
        self.train_env.close()
        self.eval_env.close()

    def _on_training_start(self):
        pass

    def _on_episode_start(self, env_idx=None):
        if env_idx is None:
            self.score = np.zeros(self.train_env.num_envs)
            self.episode_started_at_steps = [
                self.env_steps_taken
            ] * self.train_env.num_envs

            self.observation_hist = [[] for _ in range(self.train_env.num_envs)]
            self.action_hist = [[] for _ in range(self.train_env.num_envs)]
            self.reward_hist = [[] for _ in range(self.train_env.num_envs)]
            self.done_hist = [[] for _ in range(self.train_env.num_envs)]
        else:
            self.score[env_idx] = 0
            self.episode_started_at_steps[env_idx] = self.env_steps_taken
            self.observation_hist[env_idx] = []
            self.action_hist[env_idx] = []
            self.reward_hist[env_idx] = []
            self.done_hist[env_idx] = []

    def _on_episode_end(self, env_idx):
        self.episodes_played += 1
        self.train_scores.append(self.score[env_idx])
        self.frames_per_episode.append(
            (self.env_steps_taken - self.episode_started_at_steps[env_idx])
            // self.train_env.num_envs
        )

        self.agent.process_trajectory(
            self.observation_hist[env_idx],
            self.action_hist[env_idx],
            self.reward_hist[env_idx],
            self.done_hist[env_idx]
        )

    def _update_history(self, observation, action, reward, done):
        for i, (o, a, r, d) in enumerate(zip(observation, action, reward, done)):
            self.observation_hist[i].append(o)
            self.action_hist[i].append(a)
            self.reward_hist[i].append(r)
            self.done_hist[i].append(d)

    def _if_should_learn(self):
        return (self.env_steps_taken / self.samples_per_update
                - self.optimization_steps_taken) > 0

    def _if_should_evaluate(self):
        lower = (self.env_steps_taken - self.train_env.num_envs) % self.eval_freq
        higher = self.env_steps_taken % self.eval_freq
        return lower > higher and self.env_steps_taken >= self.eval_freq

    def _if_should_report(self):
        lower = (self.env_steps_taken - self.train_env.num_envs) % self.report_freq
        higher = self.env_steps_taken % self.report_freq
        return lower > higher and self.env_steps_taken >= self.report_freq

    def _process_debug_info(self, debug_info):
        if debug_info is None:
            return
        for key in self.metrics:
            self.debug_info[key] = debug_info[key] * self.alpha + \
                self.debug_info[key] * (1 - self.alpha)

    def _eval(self, num_episodes, num_steps, no_ops):
        eval_scores = experiments.evaluate_agent(
            self.agent, self.eval_env,
            num_episodes=num_episodes,
            num_steps=num_steps,
            no_ops=no_ops,
        )
        # CONSTRUCT report_info FROM debug_info
        window = min(len(self.train_scores), 100)
        self.report_info['env_steps'].append(self.env_steps_taken)
        self.report_info['optimization_steps'].append(
            self.optimization_steps_taken)
        self.report_info['episodes'].append(self.episodes_played)
        self.report_info['train_score'].append(
            sum(self.train_scores[-window:]) / window)
        self.report_info['eval_score'].append(
            sum(eval_scores) / len(eval_scores))
        self.report_info['time_per_env_step'].append(self.time_per_env_step)

        for key in self.metrics:
            self.report_info[key].append(self.debug_info[key])

    def _print_latest_statistics(self):
        """Prints the most recent statistics (and metrics)

        Outputs:
        #step: int, train_score: int, test_score: int,
        frames: int, avg_time: float,
        custom metrics(?):
            weights_norm, features_alignment, avg_q_value,
            q_update_alignment, gradient_norm
        """
        def pretty_int(i):
            if i > 1_000:
                return str(i // 1_000) + 'k'
            else:
                return str(i)

        if len(self.report_info['eval_score']) == 0:
            self.logger.info('No data to print. Try again later.')

        window = min(len(self.frames_per_episode), 100)
        fpe = sum(self.frames_per_episode[-window:]) // window

        s = f'env_step {pretty_int(self.report_info["env_steps"][-1])}  ' + \
            f'e={self.report_info["episodes"][-1]}  ' + \
            f'train_score={self.report_info["train_score"][-1]:.1f}   ' + \
            f'eval_score={self.report_info["eval_score"][-1]:.1f}   ' + \
            f'frames={fpe}  ' + \
            f'time_taken={self.report_info["time_per_env_step"][-1]:.1f}'
        self.logger.info(s)

    def to_csv(self, path):
        # Create parent directory if not exists
        pathlib.Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.report_info)
        df.to_csv(path, index=False)

    def plot(self):
        plot_data = {
            'samples': self.report_info['env_steps'],
            'train_score': self.report_info['train_score'],
            'eval_score': self.report_info['eval_score'],
        }
        utils.plot(plot_data, 'samples')
