import numpy as np
import torch

from drl.estimators.nn import (
    NonNegative,
    NoisyLinear
)


class DuelingDeepQNetwork(torch.nn.Module):
    def __init__(self, input_dims, n_actions, neurons, noisy=True):
        super(DuelingDeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.neurons = [np.prod(input_dims), *neurons]

        self.dense_layers = []
        for i in range(len(self.neurons) - 1):
            self.dense_layers.append(
                # torch.nn.utils.parametrizations.orthogonal(layer)
                # torch.nn.utils.weight_norm(layer)
                # torch.nn.Linear(self.neurons[i], self.neurons[i + 1])
                NoisyLinear(self.neurons[i], self.neurons[i + 1])
            )
            self.register_module(f'dense{i}', self.dense_layers[i])

        # self.V = torch.nn.Linear(self.neurons[-1], 1)
        # self.A = torch.nn.Linear(self.neurons[-1], self.n_actions)
        self.V = NoisyLinear(self.neurons[-1], 1)
        self.A = NoisyLinear(self.neurons[-1], self.n_actions)
        # self._parametrize()

    def forward(self, x):
        x = self.features(x)

        V = self.V(x)
        A = self.A(x)
        # torch.max(input, dim,) -> tuple(values, indices)
        # torch.mean(input, dim) -> Tensor(values)
        Q = (V + (A - torch.mean(A, dim=1, keepdim=True)))

        return Q

    def features(self, x):
        x = torch.flatten(x, start_dim=1)
        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
            x = torch.nn.functional.selu(x)
            # x = torch.nn.functional.rrelu(x, training=self.training)
        return x

    def value(self, x):
        return self.V(self.features(x))

    def q_and_features(self, x):
        x = self.features(x)
        V = self.V(x)
        A = self.A(x)
        Q = (V + (A - torch.mean(A, dim=1, keepdim=True)))
        return Q, x

    def reset_noise(self):
        for name, child in self.named_children():
            if isinstance(child, NoisyLinear):
                child.reset_noise()

    def remove_noise(self):
        for name, child in self.named_children():
            if isinstance(child, NoisyLinear):
                child.remove_noise()

    def _parametrize(self):
        for name, child in self.named_children():
            if isinstance(child, NoisyLinear):
                torch.nn.utils.parametrize.register_parametrization(
                    child, 'sigma_weight', NonNegative())
                torch.nn.utils.parametrize.register_parametrization(
                    child, 'sigma_bias', NonNegative())
        # for i in range(len(self.dense_layers)):
        #     self.dense_layers[i] = torch.nn.utils.parametrizations.orthogonal(
        #         self.dense_layers[i])
        #     self.dense_layers[i] = torch.nn.utils.weight_norm(
        #         self.dense_layers[i])
