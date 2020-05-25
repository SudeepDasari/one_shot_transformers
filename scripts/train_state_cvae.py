import torch
from hem.models.imitation_module import LatentStateImitation
from hem.models import Trainer
from hem.models.mdn_loss import GMMDistribution
import numpy as np


if __name__ == '__main__':
    trainer = Trainer('state_bc', "Trains Behavior Clone model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    action_model = LatentStateImitation(**config['policy'])
    def forward(m, device, states, actions, x_len, loss_mask):
        states, actions, x_len, loss_mask = states.to(device), actions.to(device), x_len.to(device), loss_mask.to(device)
        (mu, sigma_inv, alpha), kl, ss_p = m(states, actions, x_len, False, use_ss=True)
        action_distribution = GMMDistribution(mu, sigma_inv, alpha)
        kl = torch.mean(kl)
        neg_ll = torch.sum(-action_distribution.log_prob(actions) * loss_mask / torch.sum(loss_mask))
        loss = neg_ll + config['kl_beta'] * kl

        stats = {'neg_ll': neg_ll.item(), 'kl': kl.item(), 'schedule_samp': ss_p}
        mean_ac = action_distribution.mean.detach().cpu().numpy()
        real_ac, mask = actions.cpu().numpy(), loss_mask.cpu().numpy()
        for d in range(actions.shape[2]):
            stats['l1_{}'.format(d)] = np.sum(np.abs(mean_ac[:,:,d] - real_ac[:,:,d]) * mask / np.sum(mask))
        return loss, stats        
    trainer.train(action_model, forward)
