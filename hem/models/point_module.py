import torch
import torch.nn as nn
import torch.nn.functional as F
from hem.models import get_model
from hem.models.inverse_module import _VisualFeatures
from hem.models.traj_embed import NonLocalLayer, TemporalPositionalEncoding
from torchvision import models
from hem.models.discrete_logistic import DiscreteMixLogistic
import numpy as np
from torch.distributions import MultivariateNormal


class PointPredictor(nn.Module):
    def __init__(self, latent_dim, vis=dict()):
        super().__init__()
        # initialize visual embeddings
        self._embed = _VisualFeatures(latent_dim, **vis)

        # additional point loss on goal
        self._2_point = nn.Linear(latent_dim, 5)
    
    def forward(self, images, context):
        goal_latent = self._embed(images[:,:1], context, True)
        return self._pred_point(goal_latent, images.shape[3:])

    def _pred_point(self, goal_embed, im_shape, min_std=0.03):
        if self._2_point is None:
            return
        
        point_dist = self._2_point(goal_embed[:,0])
        mu = point_dist[:,:2]
        c1, c2, c3 = F.softplus(point_dist[:,2])[:,None], point_dist[:,3][:,None], F.softplus(point_dist[:,4])[:,None]
        scale_tril = torch.cat((c1 + min_std, torch.zeros_like(c2), c2, c3 + min_std), dim=1).reshape((-1, 2, 2))
        mu, scale_tril = [x.unsqueeze(1).unsqueeze(1) for x in (mu, scale_tril)]
        point_dist = MultivariateNormal(mu, scale_tril=scale_tril)

        h = torch.linspace(-1, 1, im_shape[0]).reshape((1, -1, 1, 1)).repeat((1, 1, im_shape[1], 1))
        w = torch.linspace(-1, 1, im_shape[1]).reshape((1, 1, -1, 1)).repeat((1, im_shape[0], 1, 1))
        hw = torch.cat((h, w), 3).repeat((goal_embed.shape[0], 1, 1, 1)).to(goal_embed.device)
        return point_dist.log_prob(hw)
