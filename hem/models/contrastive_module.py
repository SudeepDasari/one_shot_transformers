import torch
import torch.nn as nn
import torch.nn.functional as F
from hem.models import get_model
from hem.models.traj_embed import _NonLocalLayer
from torchvision import models
from hem.models.discrete_logistic import DiscreteMixLogistic
import numpy as np


class SplitContrastive(nn.Module):
    def __init__(self, agent_latent, scene_latent, n_domains=2, aux_dim=8, filter_chop=12):
        super().__init__()
        self._embed = get_model('resnet')(use_resnet18=True, drop_dim=2, output_raw=True)

        self._filter_chop = filter_chop
        self._spatial_softmax_to_agent = nn.Sequential(nn.Linear(filter_chop * 2, agent_latent), nn.ReLU(inplace=True), nn.Linear(agent_latent, agent_latent))
        self._to_scene = nn.Sequential(nn.Linear(512 - filter_chop, 512 - filter_chop), nn.ReLU(inplace=True),  nn.Linear(512 - filter_chop, scene_latent))

        self._domain_classifier = nn.Linear(agent_latent, n_domains)
        self._aux_regressor = nn.Sequential(nn.Linear(agent_latent, agent_latent), nn.ReLU(inplace=True), nn.Linear(agent_latent, aux_dim))

    def forward(self, imgs):
        vis_feat = self._embed(imgs)

        arm_feat = vis_feat[:,:self._filter_chop]
        arm_feat = F.softmax(arm_feat.view((arm_feat.shape[0], arm_feat.shape[1], -1)), dim=2).view(arm_feat.shape)
        h = torch.sum(torch.linspace(-1, 1, arm_feat.shape[2]).view((1, 1, -1)).to(arm_feat.device) * torch.sum(arm_feat, 3), 2)
        w = torch.sum(torch.linspace(-1, 1, arm_feat.shape[3]).view((1, 1, -1)).to(arm_feat.device) * torch.sum(arm_feat, 2), 2)
        arm_feat = self._spatial_softmax_to_agent(torch.cat((h, w), 1))

        pred_domain = self._domain_classifier(arm_feat)
        aux_pred = self._aux_regressor(arm_feat)

        scene_feat = self._to_scene(torch.mean(vis_feat[:,self._filter_chop:], (2,3)))
        scene_feat = F.normalize(scene_feat, dim=1)

        return {'scene_feat':scene_feat, 'arm_feat': arm_feat, 'aux_pred': aux_pred, 'pred_domain': pred_domain}


class GoalContrastive(nn.Module):
    def __init__(self, latent_dim, T=3):
        super().__init__()
        self._T = T
        self._vgg = models.vgg16(pretrained=True).features[:5]
        self._top_convs = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True), 
                            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True), 
                            nn.Conv2d(64, 64, 3, stride=2, padding=1))
        self._goal_inference_net = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, latent_dim))
        self._temporal_goal_inference = nn.Sequential(nn.Linear(128 * T, 128), nn.ReLU(inplace=True), nn.Linear(128, latent_dim))
    
    def forward(self, x):
        x_embed = self._conv_stack(x)
        has_T = len(x_embed.shape) > 4
        x_embed = x_embed.unsqueeze(1) if not has_T else x_embed
        x_embed = F.softmax(x_embed.reshape((x_embed.shape[0], x_embed.shape[1], x_embed.shape[2], -1)), dim=3).reshape(x_embed.shape)
        h = torch.sum(torch.linspace(-1, 1, x_embed.shape[3]).view((1, 1, 1, -1)).to(x.device) * torch.sum(x_embed, 4), 3)
        w = torch.sum(torch.linspace(-1, 1, x_embed.shape[4]).view((1, 1, 1, -1)).to(x.device) * torch.sum(x_embed, 3), 3)
        x_embed = torch.cat((h, w), 2)
        if has_T:
            assert x_embed.shape[1] == self._T, "x doesn't match time series!"
            x_embed = self._temporal_goal_inference(x_embed.reshape((x_embed.shape[0], -1)))
        else:
            x_embed = self._goal_inference_net(x_embed[:,0])
        x_embed = F.normalize(x_embed, dim=1)
        return x_embed

    def _conv_stack(self, imgs):
        reshaped = len(imgs.shape) > 4
        x = imgs.reshape((imgs.shape[0] * imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])) if reshaped else imgs
        x = self._vgg(x)
        x = self._top_convs(x)
        out = x.reshape((imgs.shape[0], imgs.shape[1], x.shape[1], x.shape[2], x.shape[3])) if reshaped else x
        return out


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


class ContrastiveImitation(nn.Module):
    def __init__(self, latent_dim, lstm_config, sdim=9, adim=8, queue_size=1600, context_T=3, n_mixtures=3, concat_state=True, hard_neg_samp=0.5, const_var=False):
        super().__init__()
        # initialize visual embeddings
        self._embed = _VisualFeatures(latent_dim, context_T)
        self._hard_neg_samp = hard_neg_samp

        # initialize contrastive queue
        contrast_queue = F.normalize(torch.randn((latent_dim, queue_size), dtype=torch.float32), 0)
        self._queue_ptr = 0
        self.register_buffer('contrast_queue', contrast_queue)

        # action processing
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._concat_state = concat_state
        self._is_rnn = lstm_config.get('is_rnn', True)
        if self._is_rnn:
            self._action_module = nn.LSTM(int(2 * latent_dim + float(concat_state) * sdim), lstm_config['out_dim'], lstm_config['n_layers'])
        else:
            self._action_module = nn.Sequential(nn.Linear(int(2 * latent_dim + float(concat_state) * sdim), lstm_config['out_dim']), nn.ReLU(inplace=True))
        self._dist_size = torch.Size((adim, n_mixtures))
        self._mu = nn.Linear(lstm_config['out_dim'], adim * n_mixtures)
        if const_var:
            ln_scale = torch.randn(adim, dtype=torch.float32) / np.sqrt(adim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        else:
            self._ln_scale = nn.Linear(lstm_config['out_dim'], adim * n_mixtures)
        self._logit_prob = nn.Linear(lstm_config['out_dim'], adim * n_mixtures) if n_mixtures > 1 else None

    def forward(self, states, images, context, n_noise=0, ret_dist=True, only_embed=False, img_embed=None):
        if only_embed:   # returns only the image embeddings. useful for shuffling batchnorm
            return self._embed(images, forward_predict=False)
        img_embed = self._embed(images, forward_predict=False) if img_embed is None else img_embed
        goal_embed = self._embed(torch.cat((images[:,:1], context), 1), forward_predict=True)
        states = torch.cat((img_embed, states), 2) if self._concat_state else img_embed
        
        # prepare inputs
        lstm_in = goal_embed.transpose(0, 1).repeat((states.shape[1], 1, 1))
        lstm_in = torch.cat((lstm_in, states.transpose(0, 1)), 2)
        
        # process inputs
        if self._is_rnn:
            self._action_module.flatten_parameters()
            ac_pred = self._action_module(lstm_in)[0].transpose(0, 1)
        else:
            ac_pred = self._action_module(lstm_in.transpose(0, 1))

        # predict action distribution
        mu = self._mu(ac_pred).reshape((ac_pred.shape[:-1] + self._dist_size))
        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(ac_pred).reshape((ac_pred.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)
        if self._logit_prob is not None:
            logit_prob = self._logit_prob(ac_pred).reshape((ac_pred.shape[:-1] + self._dist_size))
        else:
            logit_prob = torch.ones_like(mu)

        embeds = {'goal': goal_embed}
        if n_noise:
            embeds['positive'] = img_embed[:,-1:]
            n_hard_neg = max(int(img_embed.shape[1] * self._hard_neg_samp), 1)
            assert n_noise <= n_hard_neg, "{} hard negatives available but asked to sample {}!".format(n_hard_neg, n_noise)
            chosen = torch.multinomial(torch.ones((images.shape[0], n_hard_neg)), n_noise, replacement=False)
            embeds['negatives'] = img_embed[torch.arange(images.shape[0])[:,None], chosen]

        if ret_dist:
            return DiscreteMixLogistic(mu, ln_scale, logit_prob), embeds
        return (mu, ln_scale, logit_prob), embeds

    def append(self, keys):
        assert self.contrast_queue.shape[1] % keys.shape[1] == 0, "key shape must divide queue length!"
        self.contrast_queue[:,self._queue_ptr:self._queue_ptr+keys.shape[1]] = keys
        self._queue_ptr = (self._queue_ptr + keys.shape[1]) % self.contrast_queue.shape[1]
