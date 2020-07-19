import torch
from hem.models.point_module import PointPredictor
from hem.models import Trainer
import numpy as np
from hem.datasets.util import MEAN, STD
import cv2


if __name__ == '__main__':
    trainer = Trainer('point_pred', "Trains Point Predictor input data")
    config = trainer.config
    
    action_model = PointPredictor(**config['policy'])
    def forward(m, device, context, traj, append=True):
        images, pnts = traj['images'].to(device), traj['points'].to(device).long()
        context = context.to(device)

        # compute point prediction
        point_ll = m(images, context)
        loss = torch.mean(-point_ll[range(pnts.shape[0]), pnts[:,-1,0], pnts[:,-1,1]])
        stats = {}
        if trainer.is_img_log_step:
            points_img = torch.exp(point_ll.detach())
            maxes = points_img.reshape((points_img.shape[0], -1)).max(dim=1)[0] + 1e-3
            stats['point_img'] = (points_img[:,None] / maxes.reshape((-1, 1, 1, 1))).repeat((1, 3, 1, 1))
            stats['point_img'] = 0.7 * stats['point_img'] + 0.3 * traj['target_images'][:,0].to(device)
            pnt_color = torch.from_numpy(np.array([0,1,0])).float().to(stats['point_img'].device).reshape((1, 3))
            for i in range(-5, 5):
                for j in range(-5, 5):
                    h = torch.clamp(pnts[:,-1,0] + i, 0, images.shape[3] - 1)
                    w = torch.clamp(pnts[:,-1,1] + j, 0, images.shape[4] - 1)
                    stats['point_img'][range(pnts.shape[0]),:,h,w] = pnt_color
        return loss, stats
    trainer.train(action_model, forward)
