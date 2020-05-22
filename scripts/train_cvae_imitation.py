import torch
from hem.models.imitation_module import LatentImitation
from hem.models import Trainer
from hem.models.mdn_loss import MixtureDensityLoss, GMMDistribution
import numpy as np


if __name__ == '__main__':
    trainer = Trainer('bc_latent', "Trains Behavior Clone model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    action_model = LatentImitation(config['policy'])
    mdn_log_prob = MixtureDensityLoss()

    def forward(m, device, context, traj):
        states, actions = traj['states'][:,:-1].to(device), traj['actions'].to(device)
        images = traj['images'][:,:-1].to(device) 
        context = context.to(device)

        action_distribution, (posterior, prior) = m(states, images, context, actions)
        kl = torch.mean(torch.distributions.kl.kl_divergence(posterior, prior))
        neg_ll = torch.mean(-action_distribution.log_prob(actions))
        loss = neg_ll + config['kl_beta'] * kl

        stats = {'neg_ll': neg_ll.item(), 'kl': kl.item()}
        mean_ac = action_distribution.detach().mean.cpu().numpy()
        for d in range(actions.shape[2]):
            stats['l1_{}'.format(d)] = np.mean(np.abs(mean_ac[:,:,d] - actions.cpu().numpy()[:,:,d]))
        return loss, stats
    trainer.train(action_model, forward)
