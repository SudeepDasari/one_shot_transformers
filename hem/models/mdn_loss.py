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
        mean = self._mean(inputs).reshape((inputs.shape[0], self._n_mixtures, self._out_dim))
        sigma_inv = torch.exp(self._sigma_inv(inputs))
        alpha = F.softmax(self._alpha(inputs), 1)

        return mean, sigma_inv, alpha


def mixture_density_loss(real, mean, sigma_inv, alpha, eps=1e-5):
    C = real.shape[1]
    exp_term = -0.5 * torch.sum(torch.square(real[:, None] - mean), -1) * torch.square(sigma_inv)
    ln_frac_term = torch.log(alpha * sigma_inv + eps) - 0.5 * C * np.log(np.pi)
    expected_loss = -torch.logsumexp(ln_frac_term + exp_term, 1)
    return torch.mean(expected_loss)


class MixtureDensityLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self._eps = eps

    def forward(self, real, mean, sigma_inv, alpha):
        return mixture_density_loss(real, mean, sigma_inv, alpha, self._eps)
