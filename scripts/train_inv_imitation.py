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
    action_model = InverseImitation(**config['policy'])
    def forward(m, device, context, traj, append=True):
        states, actions = traj['states'].to(device), traj['actions'].to(device)
        images = traj['transformed'].to(device)
        context = context.to(device)

        # compute predictions and action LL
        out = m(states, images, context, ret_dist=False)
        mu_bc, scale_bc, logit_bc = out['bc_distrib']
        action_distribution = DiscreteMixLogistic(mu_bc[:,:-1], scale_bc[:,:-1], logit_bc[:,:-1])
        l_bc = torch.mean(-action_distribution.log_prob(actions))
    
        # compute inverse model density
        inv_distribution = DiscreteMixLogistic(*out['inverse_distrib'])
        l_inv = torch.mean(-inv_distribution.log_prob(actions))
        
        # compute goal embedding l2 loss
        l_goal = torch.mean(torch.sum((out['pred_goal'][:,0] - out['img_embed'][:,-1].detach()) ** 2, 1))

        loss = l_goal + l_inv + l_bc
        stats = {'inverse_loss':l_inv.item(), 'bc_loss': l_bc.item(), 'goal_loss': l_goal.item()}
        
        mean_ac = np.clip(action_distribution.mean.detach().cpu().numpy(), -1, 1)
        mean_inv = np.clip(inv_distribution.mean.detach().cpu().numpy(), -1, 1)
        for d in range(actions.shape[2]):
            a_d = actions.cpu().numpy()[:,:,d]
            stats['bc_l1_{}'.format(d)] = np.mean(np.abs(mean_ac[:,:,d] - a_d))
            stats['inv_l1_{}'.format(d)] = np.mean(np.abs(mean_inv[:,:,d] - a_d))
        return loss, stats
    trainer.train(action_model, forward)
