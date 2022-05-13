import marshal
import multiprocessing
import signal
import types

import gym
import numpy as np
from torch.distributions.utils import lazy_property


def _worker(remote, marshaled_env_fn):
    env_fn = marshal.loads(marshaled_env_fn)
    env_fn = types.FunctionType(env_fn, globals())
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == "reset":
                ob = env.reset()
                remote.send(ob)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "spec":
                remote.send(env.spec)
            elif cmd == "seed":
                remote.send(env.seed(data))
            else:
                raise NotImplementedError
    finally:
        env.close()


class MultiprocessVectorEnv():
    """VectorEnv where each env is run in its own subprocess.
    """

    def __init__(self, env_fn, num_envs):
        print(multiprocessing.get_all_start_methods())
        # multiprocessing.get_context(method=None)
        print(multiprocessing.get_start_method())

        marshaled_env_fn = marshal.dumps(env_fn.__code__)

        forkserver_available = \
            "forkserver" in multiprocessing.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.worker_remotes = zip(
            *[ctx.Pipe() for _ in range(num_envs)])
        self.workers = [
            ctx.Process(
                target=_worker,
                args=(worker_remote, marshaled_env_fn,),
                daemon=True
            ) for worker_remote in self.worker_remotes
        ]
        for worker in self.workers:
            worker.start()

        self.last_obs = [None] * num_envs
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.closed = False

    def __del__(self):
        if not self.closed:
            self.close()

    def step(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        self.last_obs, rews, dones, infos = zip(*results)
        return np.stack(self.last_obs), np.stack(rews), np.stack(dones), infos

    def reset(self, mask=None):
        self._assert_not_closed()
        if mask is None:
            mask = np.ones(self.num_envs)
        for m, remote in zip(mask, self.remotes):
            if m:
                remote.send(("reset", None))

        obs = [
            remote.recv() if m else o
            for m, remote, o in zip(mask, self.remotes, self.last_obs)
        ]
        self.last_obs = obs
        return obs

    def close(self):
        self._assert_not_closed()
        self.closed = True
        for remote in self.remotes:
            remote.send(("close", None))
        for worker in self.workers:
            worker.join()
            worker.close()

    def seed(self, seeds=None):
        self._assert_not_closed()
        if seeds is not None:
            if isinstance(seeds, int):
                seeds = [seeds] * self.num_envs
            elif isinstance(seeds, list):
                if len(seeds) != self.num_envs:
                    raise ValueError(
                        "length of seeds must be same as num_envs {}".format(
                            self.num_envs
                        )
                    )
            else:
                raise TypeError(
                    "Type of Seeds {} is not supported.".format(type(seeds))
                )
        else:
            seeds = [None] * self.num_envs

        for remote, seed in zip(self.remotes, seeds):
            remote.send(("seed", seed))
        results = [remote.recv() for remote in self.remotes]
        return results

    @lazy_property
    def spec(self):
        self._assert_not_closed()
        self.remotes[0].send(("spec", None))
        spec = self.remotes[0].recv()
        return spec

    @property
    def num_envs(self):
        return len(self.remotes)

    def _assert_not_closed(self):
        assert not self.closed, "This env is already closed"
