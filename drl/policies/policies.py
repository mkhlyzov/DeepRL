import numpy as np
import torch


class Policy(object):
    """Base interface
    """

    def __init__(self):
        pass

    def sample(self, probs):
        """Samples actions given probabilities

            Sopports both minibatches and single data points.
            Supports both torch.Tensor and numpy.ndarray inputs.
            Returned data is always integer or iterable<int> because gym
        environments don't support tensor inputs.
        """
        if len(probs.shape) > 2:
            raise ValueError()

        if isinstance(probs, torch.Tensor):
            return self._sample_pytorch(probs)
        elif isinstance(probs, np.ndarray):
            return self._sample_numpy(probs)
        else:
            raise ValueError()

    def _sample_numpy(self, probs):
        if len(probs.shape) == 1:
            return np.random.choice(probs.shape[-1], p=probs)
        elif len(probs.shape) == 2:
            actions = [
                np.random.choice(probs.shape[-1], p=p) for p in probs]
            return np.array(actions)
        else:
            raise ValueError()

    def _sample_pytorch(self, probs):
        action = torch.multinomial(probs, 1)
        if len(action) == 1:
            return action[0].item()
        return np.array(action).squeeze()

    def distribution(self, q):
        # Return distribution over actions
        raise NotImplementedError()

    def action(self, q):
        probs = self.distribution(q)
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

    def distribution(self, q):
        if isinstance(q, torch.Tensor):
            return self._distribution_pytorch(q)
        elif isinstance(q, np.ndarray):
            return self._distribution_numpy(q)
        else:
            raise ValueError()

    def _distribution_numpy(self, q):
        d = np.isclose(q, np.max(q, axis=-1, keepdims=True))
        d = d * (1 / d.sum(axis=-1, keepdims=True, dtype=q.dtype))
        return d

    def _distribution_pytorch(self, q):
        d = torch.isclose(q, torch.max(q, dim=-1, keepdim=True)[0])
        d = d * (1 / d.sum(dim=-1, keepdims=True, dtype=q.dtype))
        return d


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
