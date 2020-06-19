import torch
from hem.models.imitation_module import GoalStateRegression
from hem.models import Trainer
from hem.models.mdn_loss import GMMDistribution
import numpy as np
import matplotlib.pyplot as plt
from hem.datasets.util import MEAN, STD


if __name__ == '__main__':
    trainer = Trainer('pick_place', "Behavior Clone model on input data conditioned on teacher video")
    config = trainer.config

    # initialize behavior cloning
    model = GoalStateRegression(**config['policy'])
    l1_loss = torch.nn.SmoothL1Loss()
    def train_fn(m, device, context, start_img, targets):
        pred = m(context.to(device), start_img.to(device))
        loss = l1_loss(pred, targets.to(device))

        stats, targets, pred = {}, targets.numpy(), pred.detach().cpu().numpy()
        for d in range(targets.shape[1]):
            stats['l_d{}'.format(d)] = np.mean(np.abs(targets[:,d] - pred[:,d]))
        return loss, stats
    trainer.train(model, train_fn)
