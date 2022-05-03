"""TODO:
- Implement no-ops evaluation
- add rendering / video captures
"""

def evaluate_agent(
    agent,
    env,
    num_episodes=float('inf'),
    steps_per_episode=float('inf'),
    num_steps=float('inf'),
    no_ops=0
):
    """Run multiple evaluation episodes

    Args:
        agent (Agent): Agent used for trainig.
        env (Environment or callable): Environment used for training. Eihter \
            an environment object or a function that creates one.
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
    if callable(env):
        env = env()

    scores = []
    episodes_played, env_steps_total, env_steps, score = 0, 0, 0, 0
    observation = env.reset()
    while (episodes_played < num_episodes) and (env_steps_total < num_steps):
        if env_steps < no_ops:
            action = env.action_space.sample()
        else:
            action = agent.choose_action(observation)
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
    env.close()
    
    return scores
