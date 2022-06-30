import numpy as np
import torch


class NonNegative(torch.nn.Module):
    def forward(self, X):
        return X.abs()

    def right_inverse(self, X):
        return X.abs()


class WeightStandartisation(torch.nn.Module):
    """
    https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def forward(self, X):
        raise NotImplementedError()


class NoisyLinear(torch.nn.Linear):
    """
    Implementation of Noisy layer from the paper Noisy Networks for Exploration
    https://arxiv.org/abs/1706.10295

    https://paperswithcode.com/paper/noisy-networks-for-exploration

    main code: https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py
    fuctorized: https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/08.rainbow.ipynb
    worthy mentions:
    https://github.com/pfnet/pfrl/blob/master/pfrl/nn/noisy_linear.py
    https://github.com/thomashirtz/noisy-networks/blob/main/noisynetworks.py

    Potential idea how to generalize Noisy Layer on any Module:
    https://discuss.pytorch.org/t/feature-add-noisy-networks-for-exploration/4798
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            sigma_init: float = 0.017,
            use_factorised: bool = False,
            bias: float = True
    ):
        super(NoisyLinear, self).__init__(in_features,
                                          out_features, bias=bias)
        if not bias:
            # TODO: Adapt for no bias
            raise NotImplementedError()
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.use_factorised = use_factorised
        self.sigma_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = torch.nn.Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer(
            'epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Only init after all params added (otherwise super().__init__() fails)
        if hasattr(self, 'sigma_weight'):
            torch.nn.init.uniform_(
                self.weight, -np.sqrt(3 / self.in_features), np.sqrt(3 / self.in_features))
            torch.nn.init.uniform_(
                self.bias, -np.sqrt(3 / self.in_features), np.sqrt(3 / self.in_features))
            torch.nn.init.constant_(self.sigma_weight, self.sigma_init)
            torch.nn.init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, x):
        # w = self.weight + self.sigma_weight * self.epsilon_weight
        # b = self.bias + self.sigma_bias * self.epsilon_bias
        # w = w - w.mean(dim=1, keepdim=True)
        # w = w / w.std(dim=1, keepdim=True)
        return torch.nn.functional.linear(
            x,
            self.weight + self.sigma_weight * self.epsilon_weight,
            self.bias + self.sigma_bias * self.epsilon_bias
        )

    def reset_noise(self):
        if self.use_factorised:
            epsilon_in = torch.randn(
                self.in_features, device=self.weight.device)
            epsilon_in = epsilon_in.sign().mul(epsilon_in.abs().sqrt())
            epsilon_out = torch.randn(
                self.out_features, device=self.weight.device)
            epsilon_out = epsilon_out.sign().mul(epsilon_out.abs().sqrt())
            # outer product
            self.epsilon_weight = epsilon_out.ger(epsilon_in)
            self.epsilon_bias = epsilon_out
        else:
            self.epsilon_weight = torch.randn(
                self.out_features, self.in_features, device=self.weight.device)
            self.epsilon_bias = torch.randn(
                self.out_features, device=self.weight.device)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(
            self.out_features, self.in_features, device=self.weight.device)
        self.epsilon_bias = torch.zeros(
            self.out_features, device=self.weight.device)
