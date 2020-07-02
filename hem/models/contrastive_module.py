import torch
import torch.nn as nn
import torch.nn.functional as F
from hem.models import get_model
from hem.models.traj_embed import _NonLocalLayer
from torchvision import models


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
        self._vgg = models.vgg16(pretrained=True).features[:10]
        self._top_convs = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, stride=2, padding=1))
        self._goal_inference_net = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, latent_dim))
        self._temporal_goal_inference = nn.Sequential(nn.Linear(256 * T, 256), nn.ReLU(inplace=True), nn.Linear(256, latent_dim))
    
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
