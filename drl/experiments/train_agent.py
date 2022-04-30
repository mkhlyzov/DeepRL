import json
import time

import pandas as pd

import drl.experiments as experiments
import drl.utils as utils


class Trainer(object):
    def __init__(
        self,
        agent=None,
        env_fn=None,
        num_episodes=None,
        steps_per_episode=None,
        num_steps=None,
        samples_per_update=8,
        metricses=[],
        verbose=1,
        tensorboard=False
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
            metricses (list(string)): Metricses to consider when outputting \
                agent's performanse.
            verbose (int): Amount of output desired.
                0: no output at all,
                1: prints intermediate results to terminal,
                2: additionally saves intermediate results to csv on disc,
                3: additionally saves intermediate results to tensorboard.
        """
        self.agent = agent
        self.env_fn = env_fn
        self.samples_per_update = samples_per_update
        self.verbose = verbose

        self._supported_metricses = [
            'loss', 'grad_norm', 'weight_norm',
            'features_alignment', 'q_update_alignment', 'avg_q_value'
        ]
        self.metricses = metricses
        self.alpha=0.05  # EMA smoothing coefficient for metricses
        self.reset()

    def reset(self):
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
            'train_score': [],
            'eval_score': [],
            'time_per_env_step': [],
        }
        for key in self.metricses:
            self.debug_info[key] = 0
            self.report_info[key] = []

    def train(
        self,
        agent=None,
        env_fn=None,
        num_episodes=float('inf'),
        steps_per_episode=float('inf'),
        num_steps=float('inf'),
        eval_freq=float('inf'),
        report_freq=float('inf'),
        eval_episodes=float('inf'),
        eval_steps=float('inf'),
        reset=False,
        no_ops_evaluation=0
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
            print('Starting training procedure from scratch.')
        else:
            print('Continueing training from last step.')
        print('Creating environment instance, please wait...')
        env = self.env_fn()
        print('Environment is ready, starting training.')

        t0 = time.time()
        observation = env.reset()
        self._on_episode_start()

        while (
            self.env_steps_taken < num_steps
            and self.episodes_played < num_episodes
        ):
            action = self.agent.choose_action(observation)

            observation_, reward, done, _ = env.step(action)
            self.score += reward

            self.observation_hist.append(observation)
            self.action_hist.append(action)
            self.reward_hist.append(reward)
            self.done_hist.append(done)

            observation = observation_

            self.env_steps_taken += 1
            if self.env_steps_taken % self.samples_per_update == 0:
                debug_info = self.agent.learn()
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
                # EVALUATE PERFORMANCE, SNAPSHOT METRICSES.
                # ALL CALCULATIONS HERE
                self.time_per_env_step = (time.time() - t0)  # / eval_freq
                self._eval(eval_episodes, steps_per_episode,
                           eval_steps, no_ops_evaluation)
                self._print_latest_statistics()
                t0 = time.time()

            if self.env_steps_taken % report_freq == 0:
                # Any number % float('inf') returns that number
                pass
                # PLOT PERFORMANCE, DUMP RESULTS ON DISC.
                # NO CALCULATIONS HERE
                self.report()

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
        
        self.agent.process_episode(
            self.observation_hist,
            self.action_hist,
            self.reward_hist,
            self.done_hist
        )

    def _process_debug_info(self, debug_info):
        for key in self.metricses:
            new_value = debug_info[key]

            if len(self.debug_info[key]) != 0:
                new_value = new_value * self.alpha + \
                    self.debug_info[key][-1] * (1 - self.alpha)

            self.report[key].append(new_value)

    def _eval(self, num_episodes, steps_per_episode, num_steps, no_ops):
        eval_scores = experiments.evaluate_agent(
            self.agent, self.env_fn(),
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            num_steps=num_steps,
            no_ops=no_ops
        )
        # CONSTRUCT report_info FROM debug_info
        window = min(len(self.train_scores), 100)
        self.report_info['env_steps'].append(self.env_steps_taken)
        self.report_info['optimization_steps'].append(self.optimization_steps_taken)
        self.report_info['train_score'].append(sum(self.train_scores[-window:]) / window)
        self.report_info['eval_score'].append(sum(eval_scores) / len(eval_scores))
        self.report_info['time_per_env_step'].append(self.time_per_env_step)

    def _print_latest_statistics(self):
        """Prints the most recent statistics (metricses)
        """
        if len(self.report_info['eval_score']) == 0:
            print('No data to print. Try again later.')

        window = min(len(self.frames_per_episode), 100)
        fpe = sum(self.frames_per_episode[-window:]) // window

        s = f'env_step {self.report_info["env_steps"][-1]}   ' + \
            f'train_score={self.report_info["train_score"][-1]:.1f}   ' + \
            f'eval_score={self.report_info["eval_score"][-1]:.1f}   ' + \
            f'frames={fpe}  ' + \
            f'time_taken={self.report_info["time_per_env_step"][-1]:.1f}'
        print(s)

    def report(self, path=None):
        """Outputs latest agent's metricses

        Outputs:
        #step: int, train_score: int, test_score: int,
        frames: int, avg_time: float,
        custom metricses(?):
            weights_norm, features_alignment, avg_q_value,
            q_update_alignment, gradient_norm
        """
        plot_data = {
            'samples': self.report_info['env_steps'],
            'train_score': self.report_info['train_score'],
            'eval_score': self.report_info['eval_score'],
        }
        utils.plot(plot_data, 'samples')

        if path:
            # Write statistics to file
            df = pf.DataFrame(self.report_info)
            df.to_csv(path, index=False)
            # with open('../logs/test2.txt', 'w') as f:
            #     json.dump(scores, f)

def train_agent(
    agent,
    env,
    samples_per_update=8,
    num_episodes=None,
    num_steps=None,
    evaluate=False,
    logging=0
):
    """Run multiple training episodes

    Args:
        agent (Agent): Agent used for trainig.
        env (Environment): Environment used for training.
        samples_per_update (int): \
        num_episodes (int or None): \
        num_steps (int or None): \
        evaluate (bool): \
        logging (int): 0 to disable logging, 1 for minimal logging, \
            2 for full logging
    Returns:
    """
    pass
