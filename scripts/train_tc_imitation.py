import torch
from hem.models import get_model, Trainer
import torch.nn as nn
import numpy as np
import cv2
from hem.datasets.util import MEAN, STD, SAWYER_DEMO_PRIOR
import copy
import os


class ImitationModule(nn.Module):
    def __init__(self, embed, config):
        super().__init__()
        self._ac_bins = config['policy']['ac_bins']
        prior = config['policy'].pop('prior', None)
        if prior == 'sawyer_demos':
            prior = torch.from_numpy(SAWYER_DEMO_PRIOR.astype(np.float32).reshape((1, 1, -1)))
            self.register_buffer('_prior', prior)
        elif prior is not None:
            prior = torch.from_numpy(prior.astype(np.float32).reshape((1, 1, -1)))
            self.register_buffer('_prior', prior)
        else:
            self._prior = 0

        self._pe = config['policy']['pos_enc']
        self._embed = embed
        self._goal_state = get_model(config['goal_state'].pop('type'))
        self._goal_state = self._goal_state(**config['goal_state'])

        self._stack_len = config['policy']['action_stack_len']
        goal_module, goal_dim = [], config['policy']['goal_dim']
        for _ in range(self._stack_len):
            goal_module.extend([nn.Linear(goal_dim, goal_dim), nn.ReLU(inplace=True), nn.BatchNorm1d(goal_dim)])
        self._goal_module = nn.ModuleList(goal_module)

        state_module, state_dim = [], config['policy']['state_dim']
        for _ in range(self._stack_len):
            state_module.extend([nn.Linear(state_dim, state_dim), nn.ReLU(inplace=True), nn.LayerNorm(state_dim)])
        self._state_module = nn.ModuleList(state_module)

        # LSTM
        self._lstm = nn.LSTM(state_dim + goal_dim, config['policy']['lstm_out'], batch_first=True)
        self._to_logits = nn.Linear(config['policy']['lstm_out'], sum(config['policy']['ac_bins']))
    
    def forward(self, context, images, state):
        context_embed, img_embed = self._embed(context), self._embed(images)
        goal, img_state = self._goal_state(context_embed, img_embed)
        state = torch.cat((self.pe(state), img_state), 2)

        for l in range(self._stack_len):
            # goal process
            layer, ac, norm = [self._goal_module[3 * l + i] for i in range(3)]
            goal = norm(ac(layer(goal)) + goal)

            # state process
            layer, ac, norm = [self._state_module[3 * l + i] for i in range(3)]
            state = norm(ac(layer(state)) + state)

        goal = goal[:, None].repeat((1, state.shape[1], 1))
        state_goal = torch.cat((state, goal), 2)
        self._lstm.flatten_parameters()
        logits = self._to_logits(self._lstm(state_goal)[0]) + self._prior
        ac_logits, bins = [], logits
        for a in self._ac_bins:
            ac_logits.append(bins[:,:,:a])
            bins = bins[:,:,a:]
        return ac_logits

    def pe(self, state):
        st = []
        for d, n_p in enumerate(self._pe):
            if not n_p:
                st.append(state[:,:,d:d+1])
                continue
            new_s = []
            for l in range(n_p):
                new_s.append(torch.sin((2 ** l) * np.pi * state[:,:,d:d+1]))
                new_s.append(torch.cos((2 ** l) * np.pi * state[:,:,d:d+1]))
            st.extend(new_s)
        return torch.cat(st, -1)


if __name__ == '__main__':
    trainer = Trainer('bc', "Trains Behavior Clone model on input data", drop_last=True)
    config = trainer.config
    
    # build embedding model
    restore = config['embedding'].pop('restore', '')
    embed = get_model(config['embedding'].pop('type'))
    embed = embed(**config['embedding'])
    if restore:
        restore_model = torch.load(os.path.expanduser(restore), map_location=torch.device('cpu'))
        embed.load_state_dict(restore_model.state_dict(), strict=False)
        del restore_model

    # build Imitation Module
    policy = ImitationModule(embed, config)
    policy = policy.to(trainer.device)

    # build loss function
    cross_entropy = nn.CrossEntropyLoss()
    def forward(pi, device, context, traj):
        context = context.to(device)
        states, images = traj['states'][:,:-1].to(device), traj['images'][:,:-1].to(device)
        actions = traj['actions'].type(torch.long).to(device)

        action_logits = pi(context, images, states)
        
        loss, stats = 0, {}
        for d, logits in enumerate(action_logits):
            l_d = cross_entropy(logits.view((-1, logits.shape[-1])), actions[:,:,d].view(-1))
            acc = np.mean(np.argmax(logits.detach().cpu().numpy(), 2) == actions[:,:,d].cpu().numpy())
            stats['l_{}'.format(d)], stats['acc_{}'.format(d)] = l_d.item(), acc
            loss = loss + l_d
        return loss, stats
    trainer.train(policy, forward)
