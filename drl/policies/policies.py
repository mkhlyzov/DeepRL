import numpy as np


class Policy(object):
    """Base interface
    """

    def __init__(self):
        pass

    def distribution(self, q):
        # Return distribution over actions
        pass

    def action(self, q: np.array):
        # Sample action from self.distribution
        # Operation supports minibatches only

        if len(q.shape) == 1:
            # single state
            dist = self.distribution(q)
            action = np.random.choice(q.shape[-1], p=dist)
            return action

        elif len(q.shape) == 2:
            # minibatch
            dist = self.distribution(q)
            actions = np.zeros(q.shape[0], dtype=np.int8)
            for i in range(q.shape[0]):
                action = np.random.choice(q.shape[-1], p=dist[i])
                actions[i] = action
            return actions

        else:
            raise ValueError('Wrong shape of input data')

    def update(self, *args, **kwargs):
        # Update self to be similar to input policy
        pass


class GreedyPolicy(Policy):
    """
    Assignes maximum probability to actions with maximum Q value.
    If there are several actions with same maximum Q value, choose first one
    (argmax function returns first one)
    among them.
    """

    def __init__(self):
        super(GreedyPolicy, self).__init__()

    def distribution(self, q):
        # Return distribution over actions
        d = np.equal(q, np.max(q, axis=-1, keepdims=True))
        d = d * (1 / d.sum(axis=-1, keepdims=True))
        return d

    def action(self, q):
        # Sample action from self.distribution
        return np.argmax(q, axis=-1)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        super(EpsilonGreedyPolicy, self).__init__()
        assert (epsilon >= 0 and epsilon <= 1)
        self.epsilon = epsilon

    def distribution(self, q):
        # Return distribution over actions
        d = np.isclose(q, np.max(q, axis=-1, keepdims=True))
        d = d * (1 - self.epsilon) / d.sum(axis=-1, keepdims=True)
        d += self.epsilon / q.shape[-1]
        return d

    def update(self, new_epsilon):
        if new_epsilon >= 0 and new_epsilon <= 1:
            self.epsilon = new_epsilon


class BoltzmannPolicy(Policy):
    """
    p(a|s) = softmax(Q(s,a) / tau)
    tau->0      yields argmax one-hot encoding
    tau->inf    yields evenly distributed random sampling
    """

    def __init__(self, temperature):
        super(BoltzmannPolicy, self).__init__()
        self.tau = temperature

    def distribution(self, q):
        raise NotImplementedError()

    def update(self, new_temperature):
        self.tau = new_temperature


class MellowMaxPolicy(Policy):
    """ Policy based on mellowmax operator
    https://arxiv.org/abs/1612.05628
    https://arxiv.org/abs/2012.09456
    """

    def __init__(self, w):
        super(MellowMaxPolicy, self).__init__()
        self.w = w

    def distribution(self, q):
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

    def distribution(self, q):
        d = np.zeros(q.shape)
        for i, policy in enumerate(self.policies):
            d += policy.distribution(q) * self.weights[i]

        return d / sum(self.weights)
