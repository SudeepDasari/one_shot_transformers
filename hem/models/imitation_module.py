from hem.models import get_model, Trainer
from hem.models.mdn_loss import MixtureDensityTop, GMMDistribution
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class _Prior(nn.Module):
    def __init__(self, latent_dim, state_dim, context_dim, T):
        super().__init__()
        self._T = T
        self._context_proc = nn.Sequential(nn.Linear(T * context_dim, T * context_dim), nn.BatchNorm1d(context_dim * T), nn.ReLU(inplace=True), 
                                            nn.Linear(T * context_dim, T * context_dim), nn.BatchNorm1d(context_dim * T), nn.ReLU(inplace=True), 
                                            nn.Linear(T * context_dim, context_dim))
        self._final_proc = nn.Sequential(nn.Linear(context_dim + state_dim, context_dim + state_dim), nn.BatchNorm1d(context_dim + state_dim), nn.ReLU(inplace=True),
                                            nn.Linear(context_dim + state_dim, latent_dim * 2))
        self._l_dim = latent_dim

    def forward(self, s_0, context):
        assert context.shape[1] == self._T, "context times don't match!"
        context_embed = self._context_proc(context.view((context.shape[0], -1)))
        mean_ln_var = self._final_proc(torch.cat((context_embed, s_0), 1))
        mean, ln_var = mean_ln_var[:,:self._l_dim], mean_ln_var[:,self._l_dim:]
        covar = torch.diag_embed(torch.exp(ln_var))
        return MultivariateNormal(mean, covar)


class _Posterior(nn.Module):
    def __init__(self, latent_dim, in_dim, n_layers=1, rnn_dim=128, bidirectional=False):
        super().__init__()
        self._rnn = nn.LSTM(in_dim, rnn_dim, n_layers, bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self._out = nn.Linear(mult * 2 * rnn_dim, latent_dim * 2)
        self._l_dim = latent_dim

    def forward(self, states, actions):
        sa = torch.cat((states, actions), 2).transpose(0, 1)
        self._rnn.flatten_parameters()
        rnn_out, _ = self._rnn(sa)
        mean_ln_var = self._out(torch.cat((rnn_out[0], rnn_out[1]), 1))
        mean, ln_var = mean_ln_var[:,:self._l_dim], mean_ln_var[:,self._l_dim:]
        covar = torch.diag_embed(torch.exp(ln_var))
        return MultivariateNormal(mean, covar)

class LatentImitation(nn.Module):
    def __init__(self, config):
        super().__init__()
        # initialize visual embeddings
        embed = get_model(config['image_embedding'].pop('type'))
        self._embed = embed(**config['image_embedding'])

        latent_dim = config['latent_dim']
        self._prior = _Prior(latent_dim=latent_dim, **config['prior'])
        self._posterior = _Posterior(latent_dim=latent_dim, **config['posterior'])

        # action processing
        self._action_lstm = nn.LSTM(config['action_lstm']['in_dim'], config['action_lstm']['out_dim'], config['action_lstm'].get('n_layers', 1))
        self._mdn = MixtureDensityTop(config['action_lstm']['out_dim'], config['adim'], config['n_mixtures'])
    
    def forward(self, states, images, context, actions=None, ret_dist=True):
        img_embed = self._embed(images)
        context_embed = self._embed(context)
        states = torch.cat((img_embed, states), 2)

        prior = self._prior(states[:,0], context_embed)
        goal_latent = prior.rsample()
        posterior = self._posterior(states, actions) if actions is not None else prior

        if self.training:
            assert actions is not None
            sa_latent = posterior.rsample()
        else:
            sa_latent = prior.rsample()
        
        lstm_in = torch.cat((sa_latent, goal_latent), 1)[None].repeat((states.shape[1], 1, 1))
        lstm_in = torch.cat((lstm_in, states.transpose(0, 1)), 2)
        self._action_lstm.flatten_parameters()
        pred_embeds, _ = self._action_lstm(lstm_in)
        mu, sigma_inv, alpha = self._mdn(pred_embeds.transpose(0, 1))
        if ret_dist:
            return GMMDistribution(mu, sigma_inv, alpha), (posterior, prior)
        return (mu, sigma_inv, alpha), (posterior, prior)
