import torch
from hem.models import get_model, Trainer
from hem.models.util import batch_inputs
import torch.nn as nn
import torch.nn.functional as F


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
        in_dim, self._out_dim, self._n_out = config['top']['in_dim'], config['top']['out_dim'], config['top']['n_outputs']
        self._acs = nn.Linear(in_dim, self._out_dim * self._n_out)
        self._alphas = nn.Linear(in_dim, self._n_out)
    
    def forward(self, joints, images, depth=None):
        vis_embed = self._embed(images, depth)
        ac_embed = torch.cat((vis_embed, joints), -1)
        action_model = self._action_model(ac_embed)

        acs = self._acs(action_model).reshape((action_model.shape[0], action_model.shape[1], self._n_out, self._out_dim))
        alphas = F.softmax(self._alphas(action_model), -1)
        return acs, alphas


if __name__ == '__main__':
    trainer = Trainer('bc', "Trains Behavior Clone model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    model = ImitationModule(config)
    huber = torch.nn.SmoothL1Loss(reduction='mean')

    def forward(m, device, pairs, _):
        states, actions = batch_inputs(pairs, device)
        images = states['images'][:,:-1]
        joints = states['joints']
        depth = None
        if 'depth' in states:
            depth = states['depth'][:,:-1]

        if config.get('output_pos', True):
            actions = torch.cat((joints[:,1:,:7], actions[:,:,-1][:,:,None]), 2)

        acs, alphas = m(joints[:,:-1], images, depth)
        pred_actions = torch.sum(acs * alphas.unsqueeze(-1), 2)
        l_ac = torch.mean(torch.sum((pred_actions - actions) ** 2, 2))
        return l_ac, dict()

    trainer.train(model, forward)
