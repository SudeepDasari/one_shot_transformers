import torch
from hem.models.inverse_module import InverseImitation
from hem.models import Trainer
from hem.models.discrete_logistic import DiscreteMixLogistic
import numpy as np
import matplotlib.pyplot as plt
from hem.datasets.util import MEAN, STD
import cv2


if __name__ == '__main__':
    trainer = Trainer('bc_inv', "Trains Behavior Clone w/ inverse + goal on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    repeat_last = config.get('repeat_last', False)
    pnt_weight = config.get('pnt_weight', 0.1)
    goal_loss, goal_margin = config.get('goal_loss', True), config.get('goal_margin', -1)
    action_model = InverseImitation(**config['policy'])
    inv_loss_mult = config.get('inv_loss_mult', 1.0)
    def forward(m, device, context, traj, append=True):
        states, actions = traj['states'].to(device), traj['actions'].to(device)
        images = traj['images'].to(device)
        context = context['video'].to(device)

        if repeat_last:
            old_T = context.shape[1]
            context = context[:,-1:].repeat((1, old_T, 1, 1, 1))

        # compute predictions and action LL
        out = m(states, images, context, ret_dist=False)
        mu_bc, scale_bc, logit_bc = out['bc_distrib']
        action_distribution = DiscreteMixLogistic(mu_bc[:,:-1], scale_bc[:,:-1], logit_bc[:,:-1])
        l_bc = torch.mean(-action_distribution.log_prob(actions))
    
        # compute inverse model density
        inv_distribution = DiscreteMixLogistic(*out['inverse_distrib'])
        l_inv = inv_loss_mult * torch.mean(-inv_distribution.log_prob(actions))
        
        # compute goal embedding
        if not goal_loss:
            l_goal, goal_stat = 0, 0
        elif goal_margin < 0:
            l_goal = torch.mean(torch.sum((out['pred_goal'][:,0] - out['img_embed'][:,-1].detach()) ** 2, 1))
            goal_stat = l_goal.item()
        else:
            cos_sims = torch.matmul(out['pred_goal'], out['img_embed'].transpose(1, 2))
            goal_sim, other_sim = cos_sims[:,:,-1], cos_sims[:,0,:-1]
            l_goal = torch.mean(torch.nn.functional.relu(other_sim - goal_sim + goal_margin))
            goal_stat = l_goal.item()

        loss = l_goal + l_inv + l_bc
        stats = {'inverse_loss':l_inv.item(), 'bc_loss': l_bc.item(), 'goal_loss': goal_stat}

        if 'point_ll' in out:
            pnts = traj['points'].to(device).long()
            l_point = torch.mean(-out['point_ll'][range(pnts.shape[0]), pnts[:,-1,0], pnts[:,-1,1]])
            loss = loss + pnt_weight * l_point
            stats['point_loss'] = l_point.item()
            if trainer.is_img_log_step:
                points_img = torch.exp(out['point_ll'].detach())
                maxes = points_img.reshape((points_img.shape[0], -1)).max(dim=1)[0] + 1e-3
                stats['point_img'] = (points_img[:,None] / maxes.reshape((-1, 1, 1, 1))).repeat((1, 3, 1, 1))
                stats['point_img'] = 0.7 * stats['point_img'] + 0.3 * traj['target_images'][:,0].to(device)
                pnt_color = torch.from_numpy(np.array([0,1,0])).float().to(stats['point_img'].device).reshape((1, 3))
                for i in range(-5, 5):
                    for j in range(-5, 5):
                        h = torch.clamp(pnts[:,-1,0] + i, 0, images.shape[3] - 1)
                        w = torch.clamp(pnts[:,-1,1] + j, 0, images.shape[4] - 1)
                        stats['point_img'][range(pnts.shape[0]),:,h,w] = pnt_color

        mean_ac = np.clip(action_distribution.mean.detach().cpu().numpy(), -1, 1)
        mean_inv = np.clip(inv_distribution.mean.detach().cpu().numpy(), -1, 1)
        for d in range(actions.shape[2]):
            a_d = actions.cpu().numpy()[:,:,d]
            stats['bc_l1_{}'.format(d)] = np.mean(np.abs(mean_ac[:,:,d] - a_d))
            stats['inv_l1_{}'.format(d)] = np.mean(np.abs(mean_inv[:,:,d] - a_d))
        return loss, stats
    trainer.train(action_model, forward)
