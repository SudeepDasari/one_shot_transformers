import numpy as np
from hem.models.mdn_loss import mixture_density_loss
import torch
import unittest


def gen_dist(B=1, T=1, C=1, N=2):
    mean, s_inv, alpha = np.random.normal(size=(B, T, N, C)) * 10, np.random.uniform(1e-3, 4, size=(B, T, N)), np.random.uniform(0, 5, size=(B, T, N))
    alpha = np.exp(alpha) / np.sum(np.exp(alpha))
    return mean, s_inv, alpha


def calc_log_prob(samp, mean, sigma_inv, alpha):
    losses = []
    for b in range(mean.shape[0]):
        for t in range(mean.shape[1]):
            s = samp[b, t]
            C = s.shape[0]
            m, si, a = mean[b, t], sigma_inv[b, t], alpha[b,t]

            l_gauss = np.exp(-0.5 * np.sum((m - s[None]) ** 2, 1) * si * si) * np.power(2 * np.pi, -C / 2) * si
            l = -np.log(np.sum(l_gauss * a))
            losses.append(l)
    return np.mean(losses)


def draw_sample(m, si, a):
    chosen = np.zeros((a.shape[0], a.shape[1]),  dtype=np.int32)
    for b in range(a.shape[0]):
        for t in range(a.shape[1]):
            chosen[b,t] = np.random.choice(a.shape[-1], p=a[b,t])
    
    sample = np.zeros((m.shape[0], m.shape[1], m.shape[-1]))
    
    for b in range(a.shape[0]):
        for t in range(a.shape[1]):
            c = chosen[b, t]
            mean = m[b, t, c]
            s_inv = si[b, t, c]
            sample[b, t] = np.random.normal(m.shape[-1]) / s_inv + mean
    return sample


class TestMDNLoss(unittest.TestCase):
    def test_mdn_loss(self):
        for _ in range(10):
            B, T, C, N = [np.random.randint(1, 10) for _ in range(4)]
            m_1, si_1, a_1 = gen_dist()
            m_2, si_2, a_2 = gen_dist()

            sample = draw_sample(m_1, si_1, a_1)
            l = mixture_density_loss(torch.from_numpy(sample), torch.from_numpy(m_1), torch.from_numpy(si_1), torch.from_numpy(a_1)).numpy()
            l2 = calc_log_prob(sample, m_1, si_1, a_1)
            if not np.isinf(l2):
                self.assertTrue(np.isclose(l, l2, atol=1e-2))
            l = mixture_density_loss(torch.from_numpy(sample), torch.from_numpy(m_2), torch.from_numpy(si_2), torch.from_numpy(a_2)).numpy()
            l2 = calc_log_prob(sample, m_2, si_2, a_2)
            if not np.isinf(l2):
                self.assertTrue(np.isclose(l, l2, atol=1e-2))
