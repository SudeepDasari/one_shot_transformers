import torch
from hem.models import get_model, Trainer
from hem.models.mdn_loss import MixtureDensityLoss, MixtureDensityTop
import torch.nn as nn
import numpy as np


class ImitationModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        # initialize visual embeddings
        embed = get_model(config['embedding'].pop('type'))
        self._embed = embed(**config['embedding'])

        # processes observation embeddings into action latents
        action_model = get_model(config['action_model'].pop('type'))
        self._action_model = action_model(**config['action_model'])

        # creates mixture density output
        self._use_mdn = config.get('use_mdn', 'mdn' in config)
        if self._use_mdn:
            self._mdn = MixtureDensityTop(**config['mdn'])
    
    def forward(self, states, images, depth=None):
        vis_embed = self._embed(images, depth)
        ac_embed = torch.cat((vis_embed, states), -1)
        if self._use_mdn:
            return self._mdn(self._action_model(ac_embed))
        return self._action_model(ac_embed)

    @property
    def use_mdn(self):
        return self._use_mdn


if __name__ == '__main__':
    trainer = Trainer('bc', "Trains Behavior Clone model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    model = ImitationModule(config)
    loss = MixtureDensityLoss() if model.use_mdn else nn.SmoothL1Loss

    def forward(m, device, traj, _):
        states, actions = traj['states'][:,:-1].to(device), traj['actions'].to(device)
        images = traj['images'][:,:-1].to(device)
        
        import pdb; pdb.set_trace()
        if m.use_mdn:
            mean, sigma_inv, alpha = m(states, images)
            max_alpha = np.argmax(alpha.detach().cpu().numpy(), 2)
            tallest_mean = mean.detach().cpu().numpy()[np.arange(mean.shape[0]).reshape((-1, 1)), np.arange(mean.shape[1]).reshape((1, -1)), max_alpha]
            return loss(actions, mean, sigma_inv, alpha), {'ac_delta': np.mean(np.linalg.norm(tallest_mean - actions, 2))}
        return loss(actions, m(states, images)), {}
    trainer.train(model, forward)
