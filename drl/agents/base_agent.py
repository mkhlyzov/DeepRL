from gym import Space


class BaseAgent(object):
    """Base class for a reinforcement learning agent

    Constructor arguments:
        env: gym.env = None
        env_fn: callable = None
        observation_space: gym.spaces = None
        action_space: gym.spaces = None
    """

    def __init__(self, *args, **kwargs):
        self._setup_spaces(*args, **kwargs)

    def _setup_spaces(self, *args, **kwargs):
        env = kwargs.get('env')
        env_fn = kwargs.get('env_fn')
        observation_space = kwargs.get('observation_space')
        action_space = kwargs.get('action_space')

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

    def action(self, state):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

    def remember(self, *args):
        raise NotImplementedError()

    def save_model(self, fname=None):
        raise NotImplementedError()

    def load_model(self, fname=None):
        raise NotImplementedError()
