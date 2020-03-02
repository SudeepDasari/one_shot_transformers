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


class MixtureDensitySampler:
    def __init__(self, model):
        self._mdn_model = model
    
    def forward(self, inputs, n_samples=1):
        mean, sigma_inv, alpha = self._mdn_model(inputs)
        if len(alpha.shape) == 3:
            mean, sigma_inv, alpha = mean[:,-1], sigma_inv[:,-1], alpha[:,-1]
        
        actions = []
        for m, s_inv, a in zip(mean.cpu().numpy(), sigma_inv.cpu().numpy(), alpha.cpu().numpy()):
            lk, ac = float('-inf'), None
            for _ in range(n_samples):
                chosen = np.random.choice(a.shape[-1], p=a)
                ac_s = np.random.normal(size=m.shape[-1]) / s_inv[chosen] + m[chosen]
                l_s = np.exp(-np.sum(np.square(m[chosen] - ac_s)) * s_inv[chosen]) * s_inv[chosen] * a[chosen] / np.power(2 * np.pi, 0.5 * m.shape[-1])
                if l_s > lk:
                    lk = l_s
                    ac = ac_s
            actions.append(ac[None])
        return np.concatenate(actions, 0)
    
    def __call__(self, inputs):
        return self.forward(inputs)
