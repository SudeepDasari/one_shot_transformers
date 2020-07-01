import torch
import torch.nn as nn
import torch.nn.functional as F
from hem.models import get_model


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
    def __init__(self, latent_dim, embed_config, T=3):
        super().__init__()
        embed_type = embed_config.pop('type')
        self._T = T
        self._embed = get_model(embed_type)(out_dim=latent_dim, **embed_config)
        self._goal_inference_net = nn.Sequential(nn.Linear(latent_dim * T, latent_dim * T), nn.ReLU(inplace=True), nn.Linear(latent_dim * T, latent_dim))
    
    def forward(self, x):
        x_embed = self._embed(x)
        if len(x_embed.shape) > 2:
            assert x_embed.shape[1] == self._T, "x doesn't match time series!"
            x_embed = self._goal_inference_net(x_embed.reshape((x_embed.shape[0], -1)))
        x_embed = F.normalize(x_embed, dim=1)
        return x_embed
