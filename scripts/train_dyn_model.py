import torch
from hem.models.image_generator import RSSM
from hem.models import Trainer
import numpy as np
from hem.datasets.util import STD, MEAN
import matplotlib.pyplot as plt
from hem.util import get_kl_beta
from torch.distributions import Normal
from hem.util import get_kl_beta
_STD, _MEAN = torch.from_numpy(STD.reshape((1, 1, 3, 1, 1))), torch.from_numpy(MEAN.reshape((1, 1, 3, 1, 1)))


if __name__ == '__main__':
    trainer = Trainer('rssm_dyn', "Trains img CVAE model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    rssm = RSSM(**config['rssm'])

    def forward(m, device, pairs, _):
        recon, kl = m(pairs['images'].to(device), pairs['states'].to(device), pairs['actions'].to(device), ret_recon=True)

        # recon log probs
        state_ll = torch.mean(Normal(recon['states'], 1).log_prob(pairs['states'].to(device))) if 'states' in recon else 0
        img_ll = torch.mean(Normal(recon['images'], 1).log_prob(pairs['target_images'].to(device)))
        ll = img_ll + state_ll

        # calculate kl with free nats and get beta term
        kl_loss = torch.mean(torch.nn.functional.relu(kl - config.get('free_nats', 0)))
        beta = get_kl_beta(config, trainer.step)

        # combine log likelihood and kl terms
        loss =  beta * kl_loss - ll
        
        stats = {'kl': kl_loss.item(), 'LL': ll.item(), 'images_l1': np.mean(np.abs(pairs['target_images'].numpy() - recon['images'].detach().cpu().numpy()))}
        if 'states' in recon:
            stats['img_LL'] = img_ll.item()
            stats['states_LL'] = state_ll.item()
            stats['state_l1'] = np.mean(np.abs(pairs['states'].numpy() - recon['states'].detach().cpu().numpy()))
        
        stats['summary_vid'] = torch.clamp(torch.cat((pairs['target_images'].detach().cpu(), recon['images'].detach().cpu()), 3) * _STD + _MEAN, 0, 1)
        return loss, stats        
    trainer.train(rssm, forward)
