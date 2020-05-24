import torch
from hem.datasets import get_dataset
from hem.models import get_model, Trainer
import torch.nn as nn
import numpy as np
import copy
import os
from hem.models.traj_embed import _NonLocalLayer
import torch.nn.functional as F


class RobotCond(nn.Module):
    def __init__(self, points, sdim=9, adim=8, aux_dim=3, temperature=1, drop_dim=3, n_nonloc=2):
        super().__init__()
        self._vis_feat = get_model('resnet')(output_raw=True, drop_dim=drop_dim)

        # create depth proc
        c1, n1, a1 = nn.Conv2d(1, 32, 3, padding=1, stride=4), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        c2, n2, a2 = nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        c3, n3, a3 = nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        self._depth_proc = nn.Sequential(c1, n1, a1, c2, n2, a2, c3, n3, a3)
        self._proj_down = nn.Sequential(nn.Conv2d(1088, 1024, 1, padding=0, stride=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True))
        self._non_locs = nn.Sequential(*[_NonLocalLayer(1024, 1024, 128, temperature=temperature) for _ in range(n_nonloc)])
        self._temp_pool = nn.Sequential(*[nn.Conv3d(1024, points, 3, padding=(0, 1, 1), stride=1), nn.ReLU(inplace=True)])

        self._shared = nn.Sequential(nn.Linear(points*2 + sdim, points*2 + sdim), nn.BatchNorm1d(points*2 + sdim))
        self._action, self._aux = nn.Linear(points*2 + sdim, adim), nn.Linear(points*2 + sdim, aux_dim)
    
    def forward(self, context_imgs, robot_img, robot_depth, robot_state):
        ctx_emb = self._vis_feat(context_imgs)
        rbt_emb = self._vis_feat(robot_img)
        depth_emb = self._depth_proc(robot_depth[:,0])[:,None]
        rbt_emb = self._proj_down(torch.cat((rbt_emb, depth_emb), 2)[:,0])[:,None]

        ctx_rbt = torch.cat((ctx_emb, rbt_emb), 1).transpose(1, 2)
        non_loc = self._non_locs(ctx_rbt)
        spatial_in = torch.mean(self._temp_pool(non_loc), 2)

        # apply spatial softmax
        B, C, H, W = spatial_in.shape
        x = F.softmax(spatial_in.view((B, C, -1)), dim=2).view((B, C, H, W))
        h = torch.sum(torch.linspace(-1, 1, H).view((1, 1, -1)).to(x.device) * torch.sum(x, 3), 2)
        w = torch.sum(torch.linspace(-1, 1, W).view((1, 1, -1)).to(x.device) * torch.sum(x, 2), 2)
        spatial_softmax = torch.cat((h, w), 1)
        
        shared = self._shared(torch.cat((spatial_softmax, robot_state[:,0]), 1))
        return self._action(shared), self._aux(shared)

if __name__ == '__main__':
    trainer = Trainer('test_cond', "Trains Trajectory MoCo on input data", drop_last=True)
    config = trainer.config

    # build main model
    model = RobotCond(**config['conf'])

    # build loss_fn
    ac_loss = nn.SmoothL1Loss()
    aux_loss = nn.SmoothL1Loss()

    def train_forward(model, device, context, traj):
        context, rbt_state, rbt_images, rbt_depth = context.to(device), traj['states'][:,:-1].to(device), traj['images'][:,:-1].to(device), traj['depth'][:,:-1].to(device)
        actions, grip_loc = traj['actions'][:,0].to(device), traj['grip_location'][:,:3].to(device)

        pred_actions, pred_loc = model(context, rbt_images, rbt_depth, rbt_state)
        ac_l, aux_l = ac_loss(actions, pred_actions), aux_loss(grip_loc, pred_loc)
        loss = ac_l + aux_l

        stats = {'ac_l': ac_l.item(), 'aux_l': aux_l.item()}
        pl, gl = pred_loc.detach().cpu().numpy(), grip_loc.cpu().numpy()
        if config['dataset'].get('recenter_actions', False):
            for tens in (pl, gl):
                tens *= np.array([0.231, 0.447, 0.28409])[None]
                tens += np.array([0.647, 0.0308, 0.10047])[None]
        for d in range(3):
            stats['aux_{}'.format(d)] = np.mean(np.abs(pl[:,d] - gl[:,d]))
        return loss, stats

    trainer.train(model, train_forward)
