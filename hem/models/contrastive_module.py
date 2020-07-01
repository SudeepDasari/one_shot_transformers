import torch
import torch.nn as nn
import torch.nn.functional as F
from hem.models import get_model
from hem.models.traj_embed import _NonLocalLayer


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
    def __init__(self, latent_dim, T=3, dropout=0.1, n_non_local=2):
        super().__init__()
        self._T = T
        self._features = get_model('resnet')(output_raw=True, use_resnet18=True, drop_dim=2)
        self._goal_inference_net = nn.Sequential(*[_NonLocalLayer(512, 512, 64, dropout) for _ in range(n_non_local)])
        self._temporal_pool = nn.Conv3d(512, 512, T, padding=(0, T-2, T-2))
        self._top = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, latent_dim))
    
    def forward(self, x):
        x_embed = self._features(x)
        has_T = len(x_embed.shape) > 4
        x_embed = x_embed.unsqueeze(2) if not has_T else x_embed.transpose(1, 2)
        x_embed = self._goal_inference_net(x_embed)
        if has_T:
            assert x_embed.shape[2] == self._T, "x doesn't match time series!"
            x_embed = self._temporal_pool(x_embed)
        x_embed = self._top(torch.mean(x_embed[:,:,0], (2, 3)))
        x_embed = F.normalize(x_embed, dim=1)
        return x_embed
