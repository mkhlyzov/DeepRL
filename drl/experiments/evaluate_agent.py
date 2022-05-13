import numpy as np

from drl.envs import VectorEnv, MultiprocessVectorEnv


"""Single-env version of "evaluate_agent"
    Hangs in here because it works a bit faster than "_evaluate_agent_vec_env"
    with pseudi-vectorized env (num_envs = 1)
"""
def _evaluate_agent_single_env(
    agent,
    env,
    num_episodes=float('inf'),
    steps_per_episode=float('inf'),
    num_steps=float('inf'),
    no_ops=0,
):
    """Run multiple evaluation episodes

    Args:
        agent (Agent): Agent used for trainig.
        env (Environment or callable): Environment used for training.
        num_episodes (int or float('inf')): \
        steps_per_episode (int or float('inf')): \
        num_steps (int or float('inf')): \
        no_ops (int): Number of random actions (no-ops) for agent to take \
            at the beginning of each new episode.

    Returns:
        List of scores that agent achieved during evaluation
    """
    if isinstance(num_episodes, int):
        assert num_steps == float('inf')
    elif isinstance(num_steps, int):
        assert num_episodes == float('inf')
    else:
        raise ValueError()

    scores = []
    episodes_played, env_steps_total, env_steps, score = 0, 0, 0, 0
    observation = env.reset()
    while (episodes_played < num_episodes) and (env_steps_total < num_steps):
        if env_steps < no_ops:
            action = env.action_space.sample()
        else:
            action = agent.action(observation)
        observation_, reward, done, _ = env.step(action)
        score += reward
        observation = observation_

        env_steps += 1
        env_steps_total += 1
        if done or (env_steps == steps_per_episode):
            scores.append(score)
            score, env_steps = 0, 0
            episodes_played += 1
            observation = env.reset()
            # agent.q_eval.reset_noise()

    return scores


def _evaluate_agent_vec_env(
    agent,
    env,
    num_episodes=float('inf'),
    steps_per_episode=float('inf'),
    num_steps=float('inf'),
    no_ops=0,
):
    """Run multiple evaluation episodes

    Args:
        agent (Agent): Agent used for trainig.
        env_fn (callable): Environment used for training.
        num_episodes (int or float('inf')): \
        steps_per_episode (int or float('inf')): \
        num_steps (int or float('inf')): \
        no_ops (int): Number of random actions (no-ops) for agent to take \
            at the beginning of each new episode.
        num_envs (int): Dimentionality of vectorized envirenment to test on

    Returns:
        List of scores that agent achieved during evaluation
    """
    if isinstance(num_episodes, int):
        assert num_steps == float('inf')
    elif isinstance(num_steps, int):
        assert num_episodes == float('inf')
    else:
        raise ValueError()

    scores = []
    episodes_played, env_steps_total = 0, 0
    env_steps, score = np.zeros(env.num_envs), np.zeros(env.num_envs)

    observation = env.reset()
    while (episodes_played < num_episodes) and (env_steps_total < num_steps):
        action = agent.action(observation)
        random_action = np.array(
            [env.action_space.sample() for i in range(env.num_envs)])
        mask = (env_steps < no_ops)
        action[mask] = random_action[mask]

        observation_, reward, done, _ = env.step(action)
        score += np.array(reward)
        observation = observation_

        env_steps += 1
        env_steps_total += env.num_envs

        done = np.array(done)
        done[env_steps == steps_per_episode] = True
        observation = env.reset(done)

        for i in range(env.num_envs):
            if not done[i]:
                continue
            scores.append(score[i])
            score[i], env_steps[i] = 0, 0
            episodes_played += 1

    return scores


def evaluate_agent(
    agent,
    env=None,
    env_fn=None,
    num_envs=None,

    num_episodes=float('inf'),
    steps_per_episode=float('inf'),
    num_steps=float('inf'),
    no_ops=0,
):
    if callable(env):
        env, env_fn = env_fn, env

    if env_fn is not None:
        assert callable(env_fn), "env_fn must be a callable constructor."
        assert env is None, "T0o many arguments."

        if num_envs is None or num_envs == 1:
            env = env_fn()
        else:
            env = VectorEnv(env_fn, num_envs)
    elif env is None:
        raise ValueError("Expected testing environment (or constructor) \
            to be passed.")
    elif num_envs is not None:
        raise ValueError("Too many arguments.")

    if isinstance(env, (VectorEnv, MultiprocessVectorEnv)):
        return _evaluate_agent_vec_env(
            agent, env,
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            num_steps=num_steps,
            no_ops=no_ops,
        )
    else:
        # We assume env is valid
        return _evaluate_agent_single_env(
            agent, env,
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            num_steps=num_steps,
            no_ops=no_ops,
        )

    if env_fn is not None:
        # Close environment that we created inside this funciton
        env.close()
