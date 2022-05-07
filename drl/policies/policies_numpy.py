import numpy as np


class Policy(object):
    """Base interface
    """

    def __init__(self):
        pass

    def sample(self, probs):
        if len(probs.shape) == 1:
            return np.random.choice(probs.shape[-1], p=probs)
        elif len(probs.shape) == 2:
            actions = [
                np.random.choice(probs.shape[-1], p=p) for p in probs]
            return np.array(actions)
        else:
            raise ValueError()

    def probs(self, q):
        # Return distribution over actions
        raise NotImplementedError()

    def action(self, q):
        probs = self.probs(q)
        return self.sample(probs)

    def update(self, *args, **kwargs):
        # Update self to be similar to input policy
        raise NotImplementedError()


class GreedyPolicy(Policy):
    """
    Assignes maximum probability to actions with maximum Q value.
    If there are several actions with same maximum Q value, choose first one
    (argmax function returns first one)
    among them.
    """

    def __init__(self):
        super(GreedyPolicy, self).__init__()

    def probs(self, q):
        d = np.isclose(q, np.max(q, axis=-1, keepdims=True))
        d = d * (1 / d.sum(axis=-1, keepdims=True, dtype=q.dtype))
        return d


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        super(EpsilonGreedyPolicy, self).__init__()
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

    def probs(self, q):
        d = np.isclose(q, np.max(q, axis=-1, keepdims=True))
        d = d / d.sum(axis=-1, keepdims=True)
        d = d * (1 - self.epsilon) + self.epsilon / q.shape[-1]
        return d

    def update(self, new_epsilon):
        assert 0 <= new_epsilon <= 1
        self.epsilon = new_epsilon


class BoltzmannPolicy(Policy):
    """
    p(a|s) = softmax(Q(s,a) / tau)
    tau->0      yields argmax one-hot encoding
    tau->inf    yields evenly distributed random sampling
    """

    def __init__(self, temperature):
        super(BoltzmannPolicy, self).__init__()
        assert temperature > 0
        self.tau = temperature

    def probs(self, q):
        raise NotImplementedError()

    def update(self, new_temperature):
        assert new_temperature > 0
        self.tau = new_temperature


class MellowMaxPolicy(Policy):
    """ Policy based on mellowmax operator
    https://arxiv.org/abs/1612.05628
    https://arxiv.org/abs/2012.09456
    """

    def __init__(self, w):
        super(MellowMaxPolicy, self).__init__()
        self.w = w

    def probs(self, q):
        raise NotImplementedError()

    def update(self, new_w):
        self.w = new_w


class MixedPolicy(Policy):
    """Implements superposition of policies (distribution-wise)
    """

    def __init__(self, policies: list, weights: list = None):
        super(MixedPolicy, self).__init__()
        self.policies = policies
        self.weights = weights
        if self.weights is None:
            self.weights = [1.] * len(self.policies)

    def probs(self, q):
        d = np.zeros(q.shape)
        for i, policy in enumerate(self.policies):
            d += policy.distribution(q) * self.weights[i]

        return d / sum(self.weights)
