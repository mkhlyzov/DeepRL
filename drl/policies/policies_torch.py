import torch


class Policy(object):
    """Base interface
    """

    def __init__(self):
        pass

    def sample(self, probs):
        """Samples actions given probabilities
        """
        if len(probs.shape) > 2:
            raise ValueError()

        action = torch.multinomial(probs, 1)
        return action.squeeze(-1)

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
    def _probs(self, q):
        d = torch.isclose(q, torch.max(q, dim=-1, keepdim=True)[0])
        d = d * (1 / d.sum(dim=-1, keepdims=True, dtype=q.dtype))
        return d

    def probs(self, q):
        # Faster implementation
        return (q == q.max(dim=-1, keepdim=True)[0]).to(q.dtype)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        super(EpsilonGreedyPolicy, self).__init__()
        assert (0 <= epsilon <= 1)
        self.epsilon = epsilon

    def _probs(self, q):
        d = torch.isclose(q, torch.max(q, dim=-1, keepdims=True)[0])
        d = d * (1 / d.sum(dim=-1, keepdims=True, dtype=q.dtype))
        d = d * (1 - self.epsilon) + self.epsilon / q.shape[-1]
        return d

    def probs(self, q):
        # Faster implementation
        probs = GreedyPolicy.probs(self, q) * (1 - self.epsilon)
        probs += self.epsilon / q.shape[-1]
        return probs

    def update(self, new_epsilon):
        assert (0 <= new_epsilon <= 1)
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

    @torch.no_grad()
    def probs(self, q):
        q_ = (q / self.tau).float()
        return torch.nn.functional.softmax(q_, dim=-1).to(q.dtype)

    def update(self, new_temperature):
        assert new_temperature > 0
        self.tau = new_temperature


class MellowMaxPolicy(Policy):
    """Policy based on mellowmax operator
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
        d = torch.zeros_like(q)
        for weight, policy in zip(self.weights, self.policies):
            d += policy.probs(q) * weight

        return d / sum(self.weights)
