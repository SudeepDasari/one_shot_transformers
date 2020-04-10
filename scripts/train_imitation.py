import torch
from hem.models import get_model, Trainer
from hem.models.mdn_loss import MixtureDensityLoss, MixtureDensityTop
import torch.nn as nn


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
        self._mdn = MixtureDensityTop(**config['mdn'])

    
    def forward(self, states, images, depth=None):
        vis_embed = self._embed(images, depth)
        ac_embed = torch.cat((vis_embed, states), -1)
        mean, sigma_inv, alpha = self._mdn(self._action_model(ac_embed))
        return mean, sigma_inv, alpha


if __name__ == '__main__':
    trainer = Trainer('bc', "Trains Behavior Clone model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    model = ImitationModule(config)
    loss = MixtureDensityLoss()

    def forward(m, device, traj, _):
        states, actions = traj['states'][:,:-1].to(device), traj['actions'].to(device)
        images = traj['images'][:,:-1].to(device)
        
        mean, sigma_inv, alpha = m(states, images)
        l_mdn = loss(actions, mean, sigma_inv, alpha)
        return l_mdn, {}
    trainer.train(model, forward)
