import torch
from hem.models import Trainer
from hem.models.baseline_module import ContextualLSTM
from hem.models.discrete_logistic import DiscreteMixLogistic
import numpy as np
import matplotlib.pyplot as plt
from hem.datasets.util import MEAN, STD
import cv2


if __name__ == '__main__':
    trainer = Trainer('bc_contextual', "Trains Behavior Clone w/ inverse + goal on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    action_model = ContextualLSTM(**config['policy'])
    def forward(m, device, context, traj, append=True):
        states, actions = traj['states'].to(device), traj['actions'].to(device)
        images = traj['images'].to(device)
        context = context['video'].to(device)

        # compute predictions and action LL
        mu_bc, scale_bc, logit_bc = m(states, images, context, ret_dist=False)
        action_distribution = DiscreteMixLogistic(mu_bc[:,:-1], scale_bc[:,:-1], logit_bc[:,:-1])
        l_bc = torch.mean(-action_distribution.log_prob(actions))
    
        stats = {'l_bc': l_bc.item()}
        mean_ac = np.clip(action_distribution.mean.detach().cpu().numpy(), -1, 1)
        for d in range(actions.shape[2]):
            a_d = actions.cpu().numpy()[:,:,d]
            stats['bc_l1_{}'.format(d)] = np.mean(np.abs(mean_ac[:,:,d] - a_d))
        return l_bc, stats
    trainer.train(action_model, forward)
