import torch
from hem.models.image_generator import CondVAE
from hem.models import Trainer
import numpy as np
from hem.datasets.util import STD, MEAN
import matplotlib.pyplot as plt
from hem.util import get_kl_beta
_STD, _MEAN = torch.from_numpy(STD.reshape((3, 1, 1))), torch.from_numpy(MEAN.reshape((3, 1, 1)))


if __name__ == '__main__':
    trainer = Trainer('grip_cvae', "Trains img CVAE model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    cvae = CondVAE(**config['model'])
    l1_loss = torch.nn.SmoothL1Loss()

    def forward(m, device, context, targets):
        context, targets = context.to(device), targets.to(device)
        pred, kl = m(context)

        beta = get_kl_beta(config, trainer.step)
        loss = l1_loss(targets, pred) + beta * torch.mean(kl)

        target_vis = torch.clamp(targets.detach() * _STD.to(device)[None] + _MEAN.to(device)[None], 0, 1)
        pred_vis = torch.clamp(pred.detach() * _STD.to(device)[None] + _MEAN.to(device)[None], 0, 1)
        stats = {'kl': torch.mean(kl).item(), 'real': target_vis, 'pred': pred_vis}
        return loss, stats        
    trainer.train(cvae, forward)
