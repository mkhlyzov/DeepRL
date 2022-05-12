import numpy as np


class VectorEnv():
    def __init__(
        self,
        env_fn: callable,
        num_envs: int
    ):
        if num_envs <= 0:
            raise ValueError('num_envs should be positive: {num_envs}')
        self._assert_env_fn_is_valid(env_fn)

        self.observation_space = env_fn().observation_space
        self.action_space = env_fn().action_space
        self._envs = [env_fn() for _ in range(num_envs)]

        self._last_obs = [None] * self.num_envs

    def _assert_env_fn_is_valid(self, env_fn):
        if not callable(env_fn):
            raise ValueError()
        env = env_fn()
        if not (
            hasattr(env, 'reset')
            and hasattr(env, 'step')
            and hasattr(env, 'close')
            and hasattr(env, 'action_space')
            and hasattr(env, 'observation_space')
        ):
            raise ValueError()

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.num_envs)
        obs = [
            env.reset() if m else o
            for m, env, o in zip(mask, self._envs, self._last_obs)
        ]
        self._last_obs = obs
        return obs

    def step(self, vectorized_action, auto_reset=False):
        assert hasattr(vectorized_action, '__len__')
        assert len(vectorized_action) == self.num_envs

        results = [
            env.step(a) for env, a in zip(self._envs, vectorized_action)
        ]  # list of tuples
        self._last_obs, rews, dones, infos = zip(*results)  # tuple of tuples
        if auto_reset and any(dones):
            self.reset(dones)
        return self._last_obs, rews, dones, infos  # returns 4 tuples
        # return np.array(self._last_obs), np.array(rews), np.array(dones), infos

    def seed(self, seeds):
        for env, seed in zip(self._envs, seeds):
            env.seed(seed)

    def close(self):
        for env in self._envs:
            env.close()

    @property
    def num_envs(self):
        return len(self._envs)
