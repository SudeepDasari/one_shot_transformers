import torch
from hem.models.imitation_module import LatentImitation
from hem.models import Trainer
from hem.models.discrete_logistic import DiscreteMixLogistic
import numpy as np
from hem.util import get_kl_beta


if __name__ == '__main__':
    trainer = Trainer('bc_latent', "Trains Behavior Clone model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    action_model = LatentImitation(config['policy'])
    def forward(m, device, context, traj):
        states, actions = traj['states'][:,:-1].to(device), traj['actions'].to(device)
        images = traj['images'][:,:-1].to(device) 
        context = context.to(device)

        (mu, ln_scale, logit_prob), kl = m(states, images, context, actions, ret_dist=False)
        action_distribution = DiscreteMixLogistic(mu, ln_scale, logit_prob)
        kl, kl_beta = torch.mean(kl), get_kl_beta(config, trainer.step)
        neg_ll = torch.mean(-action_distribution.log_prob(actions))
        loss = neg_ll + kl_beta * kl

        stats = {'neg_ll': neg_ll.item(), 'kl': kl.item(), 'kl_beta': kl_beta}
        mean_ac = np.clip(action_distribution.mean.detach().cpu().numpy(), -1, 1)
        for d in range(actions.shape[2]):
            stats['l1_{}'.format(d)] = np.mean(np.abs(mean_ac[:,:,d] - actions.cpu().numpy()[:,:,d]))
        return loss, stats
    trainer.train(action_model, forward)
