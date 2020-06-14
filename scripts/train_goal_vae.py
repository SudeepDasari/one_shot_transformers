import torch
from hem.models.image_generator import GoalVAE
from hem.models import Trainer
import numpy as np
import matplotlib.pyplot as plt
from hem.util import get_kl_beta
from torch.distributions import Laplace


if __name__ == '__main__':
    trainer = Trainer('goal_vae', "Trains goal image model on input data")
    config = trainer.config
    
    # build Goal VAE
    gvae = GoalVAE(**config['model'])
    def forward(m, device, context, targets):
        start, goal, targets = context['start'].to(device), context['goal'].to(device), targets.to(device)
        pred, kl = m(start, goal)
        recon_ll, kl = torch.mean(Laplace(pred, 1).log_prob(targets)), torch.mean(kl)

        beta = get_kl_beta(config, trainer.step)
        loss = beta * kl - recon_ll

        target_vis = torch.clamp(targets.detach(), 0, 1)
        pred_vis = torch.clamp(pred.detach(), 0, 1)
        stats = {'kl': torch.mean(kl).item(), 
                'kl_beta': beta,
                'recon_ll': recon_ll.item(), 
                'l1': torch.mean(torch.abs(pred.detach() - targets)).item(), 
                'real_vs_pred': torch.cat((target_vis, pred_vis), 2)}
        return loss, stats        
    trainer.train(gvae, forward)
