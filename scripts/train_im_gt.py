import torch
from hem.models import get_model, Trainer
from hem.models.util import batch_inputs
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
    
    def forward(self, joints, images, depth=None):
        vis_embed = self._embed(images, depth)
        ac_embed = torch.cat((vis_embed, joints), -1)
        return self._mdn(self._action_model(ac_embed))


if __name__ == '__main__':
    trainer = Trainer('bc', "Trains Behavior Clone model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    model = ImitationModule(config)
    loss = MixtureDensityLoss()

    def forward(m, device, pairs, _):
        states, actions = batch_inputs(pairs, device)
        images = states['images'][:,:-1]
        joints = states['joints']
        
        depth = None
        if 'depth' in states:
            depth = states['depth'][:,:-1]

        if config.get('output_pos', True):
            actions = torch.cat((joints[:,1:,:7], actions[:,:,-1][:,:,None]), 2)
        
        mean, sigma_inv, alpha = m(joints[:,:-1], images, depth)
        l_mdn = loss(actions, mean, sigma_inv, alpha)
        return l_mdn, {}

    trainer.train(model, forward)
