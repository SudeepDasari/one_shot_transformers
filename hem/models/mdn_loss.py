import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class MixtureDensityTop(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures=3):
        super().__init__()
        self._n_mixtures = n_mixtures
        self._out_dim = out_dim
        self._mean = nn.Linear(in_dim, out_dim * n_mixtures)
        self._alpha = nn.Linear(in_dim, n_mixtures)
        self._sigma_inv = nn.Linear(in_dim, n_mixtures)
    
    def forward(self, inputs):
        prev_shape = inputs.shape[:-1]
        mean = self._mean(inputs).reshape(list(prev_shape) + [self._n_mixtures, self._out_dim])
        sigma_inv = torch.exp(self._sigma_inv(inputs))
        alpha = F.softmax(self._alpha(inputs), -1)

        return mean, sigma_inv, alpha


def mixture_density_loss(real, mean, sigma_inv, alpha, eps=1e-5):
    C = real.shape[-1]
    exp_term = -0.5 * torch.sum(((real.unsqueeze(-2) - mean) ** 2), -1) * (sigma_inv ** 2)
    ln_frac_term = torch.log(alpha * sigma_inv + eps) - 0.5 * C * np.log(np.pi)
    expected_loss = -torch.logsumexp(ln_frac_term + exp_term, -1)
    return torch.mean(expected_loss)


class MixtureDensityLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self._eps = eps

    def forward(self, real, mean, sigma_inv, alpha):
        return mixture_density_loss(real, mean, sigma_inv, alpha, self._eps)


class MixtureDensitySampler:
    def __init__(self, model):
        self._mdn_model = model
    
    def forward(self, inputs):
        mean, sigma_inv, alpha = self._mdn_model(inputs)
        import pdb; pdb.set_trace()
    
    def __call__(self, inputs):
        return self.forward(inputs)
