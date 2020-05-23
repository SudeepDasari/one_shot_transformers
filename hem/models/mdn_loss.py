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
    ln_frac_term = torch.log(alpha * sigma_inv + eps) - 0.5 * C * np.log(2 * np.pi)
    expected_loss = -torch.logsumexp(ln_frac_term + exp_term, -1)
    return torch.mean(expected_loss)


class MixtureDensityLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self._eps = eps

    def forward(self, real, mean, sigma_inv, alpha):
        return mixture_density_loss(real, mean, sigma_inv, alpha, self._eps)


class GMMDistribution(torch.distributions.Distribution):
    def __init__(self, mean, sigma_inv, alpha, validate_args=None):
        assert mean.device == sigma_inv.device and mean.device == alpha.device, "all tensors must lie on same device!"
        batch_shape, event_shape = sigma_inv.shape[:-1], mean.shape[-1:]
        super().__init__(batch_shape, event_shape,validate_args)
        self._mean = mean
        self._sigma_inv = sigma_inv
        self._alpha = alpha
    
    def sample(self, sample_shape=torch.Size()):
        alpha_sampler = torch.distributions.Categorical(probs=self._alpha)
        with torch.no_grad():
            sample_alphas = alpha_sampler.sample(sample_shape)
            index, pad = [], [1 for _ in range(len(sample_shape))]
            for i, b in enumerate(self._batch_shape):
                ind_shape = pad + [1 if j != i else b for j in range(len(self._batch_shape))]
                index.append(torch.arange(b).view(ind_shape))
            index.append(sample_alphas)
            mean = self._mean[index]
            sigma_inv = self._sigma_inv[index]
            samples = torch.randn(self._extended_shape(sample_shape)).to(self._mean.device) / sigma_inv.unsqueeze(-1) + mean
        return samples

    @property
    def mean(self):
        return torch.sum(self._mean * self._alpha.unsqueeze(-1), -2)
    
    def log_prob(self, value, eps=1e-5):
        assert value.shape[-(len(self._batch_shape) + len(self._event_shape)):] == self._batch_shape + self._event_shape, "shapes must match!"
        C = self._event_shape[0]
        pad_shape = torch.Size([-1]) + self._batch_shape + self._event_shape

        mean, alpha, sigma_inv = self._mean[None], self._alpha[None], self._sigma_inv[None]
        exp_term = -0.5 * torch.sum(((value.view(pad_shape).unsqueeze(-2) - mean) ** 2), -1) * (sigma_inv ** 2)
        ln_frac_term = torch.log(alpha * sigma_inv + eps) - 0.5 * C * np.log(2 * np.pi)
        log_prob = torch.logsumexp(ln_frac_term + exp_term, -1)
        if len(value.shape) == len(self._batch_shape + self._event_shape):
            return log_prob[0]
        return log_prob

    @property
    def highest_mean(self):
        peaks = torch.argmax(self._alpha, -1)

        index = []
        for i, b in enumerate(self._batch_shape):
            ind_shape = [1 if j != i else b for j in range(len(self._batch_shape))]
            index.append(torch.arange(b).view(ind_shape))
        index.append(peaks)
        return self._mean[index]
