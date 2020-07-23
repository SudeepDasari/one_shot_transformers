import torch
import learn2learn as l2l
from hem.models import Trainer
from hem.models.baseline_module import DAMLNetwork
from hem.models.mdn_loss import GMMDistribution
import numpy as np
import matplotlib.pyplot as plt
from hem.datasets.util import MEAN, STD
import cv2


if __name__ == '__main__':
    trainer = Trainer('bc_daml', "Trains Behavior Clone w/ inverse + goal on input data", allow_val_grad=True)
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    action_model = DAMLNetwork(**config['policy'])
    action_model = l2l.algorithms.MAML(action_model, lr=config['policy']['maml_lr'], first_order=config['policy']['first_order'], allow_unused=True)
    inner_iters = config.get('inner_iters', 1)
    l2error = torch.nn.MSELoss()
    def forward(meta_model, device, context, traj, append=True):
        states, actions = traj['states'].to(device), traj['actions'].to(device)
        images = traj['images'].to(device)
        context = context.to(device)
        aux = traj['aux_pose'].to(device)
        
        # compute per task learned train loss and val loss
        error = 0
        bc_loss, aux_loss = [], []
        for task in range(states.shape[0]):
            learner = meta_model.clone()
            for _ in range(inner_iters):
                learner.adapt(learner(None, context[task], learned_loss=True)['learned_loss'])
            out = learner(states[task], images[task], ret_dist=False)
            l_aux = l2error(out['aux'], aux[task][None])
            mu, sigma_inv, alpha = out['action_dist']
            action_distribution = GMMDistribution(mu[1:-1], sigma_inv[1:-1], alpha[1:-1])
            l_bc = -torch.mean(action_distribution.log_prob(actions[task]))
            validation_loss = l_bc + l_aux
            error += validation_loss / states.shape[0]
            bc_loss.append(l_bc.item())
            aux_loss.append(l_aux.item())
    
        stats = {'l_bc': np.mean(bc_loss), 'aux_loss': np.mean(aux_loss)}
        return error, stats
    trainer.train(action_model, forward)
