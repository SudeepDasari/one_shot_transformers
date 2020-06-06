import torch
from hem.models.image_generator import RSSM
from hem.models import Trainer
import numpy as np
from hem.datasets.util import STD, MEAN
import matplotlib.pyplot as plt
from hem.util import get_kl_beta
_STD, _MEAN = torch.from_numpy(STD.reshape((1, 1, 3, 1, 1))), torch.from_numpy(MEAN.reshape((1, 1, 3, 1, 1)))


if __name__ == '__main__':
    trainer = Trainer('rssm_dyn', "Trains img CVAE model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    rssm = RSSM(**config['rssm'])
    img_loss = torch.nn.MSELoss()
    state_loss = torch.nn.MSELoss()

    def forward(m, device, pairs, _):
        recon, kl = m(pairs['images'].to(device), pairs['states'].to(device), pairs['actions'].to(device), ret_recon=True)
        recon_state_loss = state_loss(pairs['states'].to(device), recon['states']) if 'states' in recon else 0
        recon_img_loss = img_loss(pairs['target_images'].to(device), recon['images'])
        recon_loss = recon_state_loss + recon_img_loss
        kl_loss = torch.mean(torch.nn.functional.relu(torch.sum(kl, 1) - config.get('free_nats', 0)))

        loss = kl_loss + recon_loss
        stats = {'kl': torch.mean(kl).item(), 'recon': recon_loss.item()}
        if 'states' in recon:
            stats['img_recon'] = recon_img_loss.item()
            stats['states_recon'] = recon_state_loss.item()
        
        stats['summary_vid'] = torch.clamp(torch.cat((pairs['target_images'].detach().cpu(), recon['images'].detach().cpu()), 3) * _STD + _MEAN, 0, 1)
        return loss, stats        
    trainer.train(rssm, forward)
