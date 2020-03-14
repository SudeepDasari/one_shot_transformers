import torch
from hem.models import get_model, Trainer
from hem.models.util import batch_inputs
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
        self._top = nn.Linear(**config['top'])

        self._aux = False
        if 'auxiliary' in config:
            self._aux = True
            hidden = config['auxiliary'].get('hidden_dim', 16)
            l1 = nn.Linear(config['auxiliary']['in_dim'], hidden)
            a1 = nn.ReLU(inplace=True)
            l2 = nn.Linear(hidden, config['auxiliary']['out_dim'])
            self._aux_linear = nn.Sequential(l1, a1, l2)
    
    def forward(self, joints, images, depth=None):
        vis_embed = self._embed(images, depth)
        ac_embed = torch.cat((vis_embed, joints[:,:,-2:]), -1)
        aux_pred = None

        if self._aux:
            aux_pred = self._aux_linear(vis_embed)
            ac_embed = torch.cat((ac_embed, aux_pred.detach()), -1)
        
        pred_actions = self._top(self._action_model(ac_embed))
        return pred_actions, aux_pred


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
        import pdb; pdb.set_trace()
        pred_actions, pred_state = m(joints[:,:-1], images, depth)
        l_ac = huber(pred_actions, actions)

        stats = dict(act=l_ac.item())
        if pred_state is not None:
            state_loss = torch.mean(torch.sum((pred_state - joints[:,:-1,:7]) ** 2, (1, 2)))
            stats['aux_loss'] = state_loss.item()
        return l_ac + config['auxiliary'].get('weight', 0.5) * state_loss, stats
    trainer.train(model, forward)
