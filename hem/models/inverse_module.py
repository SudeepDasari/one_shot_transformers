import torch
import torch.nn as nn
import torch.nn.functional as F
from hem.models import get_model
from hem.models.traj_embed import _NonLocalLayer
from torchvision import models
from hem.models.discrete_logistic import DiscreteMixLogistic
import numpy as np


class _VisualFeatures(nn.Module):
    def __init__(self, latent_dim, context_T, embed_hidden=256):
        super().__init__()
        self._resnet_features = get_model('resnet')(output_raw=True, drop_dim=2, use_resnet18=True)
        self._temporal_process = nn.Sequential(_NonLocalLayer(512, 512, 128, dropout=0.2), nn.Conv3d(512, 512, (context_T, 1, 1), 1))
        self._to_embed = nn.Sequential(nn.Linear(1024, embed_hidden), nn.ReLU(inplace=True), nn.Linear(embed_hidden, latent_dim))

    def forward(self, images, forward_predict):
        assert len(images.shape) == 5, "expects [B, T, C, H, W] tensor!"
        features = self._resnet_features(images)
        if forward_predict:
            features = self._temporal_process(features.transpose(1, 2)).transpose(1, 2)
            features = torch.mean(features, 1, keepdim=True)

        features = F.softmax(features.reshape((features.shape[0], features.shape[1], features.shape[2], -1)), dim=3).reshape(features.shape)
        h = torch.sum(torch.linspace(-1, 1, features.shape[3]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 4), 3)
        w = torch.sum(torch.linspace(-1, 1, features.shape[4]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 3), 3)
        spatial_softmax = torch.cat((h, w), 2)
        return F.normalize(self._to_embed(spatial_softmax), dim=2)


class _DiscreteLogHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures, const_var=False):
        super().__init__()
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._n_mixtures, self._dist_size = n_mixtures, torch.Size((out_dim, n_mixtures))
        self._mu = nn.Linear(in_dim, out_dim * n_mixtures)
        if const_var:
            ln_scale = torch.randn(out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        else:
            self._ln_scale = nn.Linear(in_dim, out_dim * n_mixtures)
        self._logit_prob = nn.Linear(in_dim, out_dim * n_mixtures) if n_mixtures > 1 else None
    
    def forward(self, x):
        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))
        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(x).reshape((x.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)
        
        logit_prob = self._logit_prob(x).reshape((x.shape[:-1] + self._dist_size)) if self._n_mixtures > 1 else torch.ones_like(mu)
        return (mu, ln_scale, logit_prob)


class InverseImitation(nn.Module):
    def __init__(self, latent_dim, lstm_config, sdim=9, adim=8, context_T=3, n_mixtures=3, concat_state=True, const_var=False):
        super().__init__()
        # initialize visual embeddings
        self._embed = _VisualFeatures(latent_dim, context_T)

        # inverse modeling
        inv_dim = latent_dim * 2
        self._inv_model = nn.Sequential(nn.Linear(inv_dim, inv_dim), nn.ReLU(inplace=True), _DiscreteLogHead(inv_dim, adim, n_mixtures, const_var))

        # action processing
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._concat_state, self._n_mixtures = concat_state, n_mixtures
        self._action_module = nn.LSTM(int(2 * latent_dim + float(concat_state) * sdim), lstm_config['out_dim'], lstm_config['n_layers'])
        self._imitation_dist = _DiscreteLogHead(lstm_config['out_dim'], adim, n_mixtures, const_var)

    def forward(self, states, images, context, ret_dist=True):
        img_embed = self._embed(images, forward_predict=False)
        goal_embed = self._embed(torch.cat((images[:,:1], context), 1), forward_predict=True)
        states = torch.cat((img_embed, states), 2) if self._concat_state else img_embed
        
        # run inverse model
        inv_in = torch.cat((img_embed[:,:-1], img_embed[:,1:]), 2)
        mu_inv, scale_inv, logit_inv = self._inv_model(inv_in)

        # predict behavior cloning distribution
        lstm_in = img_embed[:,-1:].transpose(0, 1).repeat((states.shape[1], 1, 1))
        lstm_in = torch.cat((lstm_in, states.transpose(0, 1)), 2)
        self._action_module.flatten_parameters()
        ac_pred = self._action_module(lstm_in)[0].transpose(0, 1)
        mu_bc, scale_bc, logit_bc = self._imitation_dist(ac_pred)

        out = {}
        # package distribution in objects or as tensors
        if ret_dist:
            out['bc_distrib'] = DiscreteMixLogistic(mu_bc, scale_bc, logit_bc)
            out['inverse_distrib'] = DiscreteMixLogistic(mu_inv, scale_inv, logit_inv)
        else:
            out['bc_distrib'] = (mu_bc, scale_bc, logit_bc)
            out['inverse_distrib'] = (mu_inv, scale_inv, logit_inv)

        out['pred_goal'] = goal_embed
        out['img_embed'] = img_embed
        return out
