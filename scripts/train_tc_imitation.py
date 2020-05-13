import torch
from hem.models import get_model, Trainer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from hem.datasets.util import MEAN, STD, SAWYER_DEMO_PRIOR
import copy
import os


class ImitationModule(nn.Module):
    def __init__(self, embed, config):
        super().__init__()
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
        self._goal_state_embed = get_model(config['goal_state'].pop('type'))
        self._goal_state_embed = self._goal_state_embed(**config['goal_state'])

        self._in_dim = config['policy']['state_dim'] + config['policy']['goal_dim']
        # state goal stack
        self._stack_len = config['policy'].get('sg_stack', 0)
        if self._stack_len:
            sg_stack = []
            for _ in range(self._stack_len):
                sg_stack.extend([nn.Linear(self._in_dim, self._in_dim), nn.ReLU(inplace=True)])
            self._sg_stack = nn.Sequential(*sg_stack)
        # Action Processing
        self._is_lstm = config['policy'].get('lstm', True)
        if self._is_lstm:
            self._ac_proc = nn.LSTM(self._in_dim, config['policy']['ac_out'], batch_first=True)
        else:
            linear, ac = nn.Linear(self._in_dim, config['policy']['ac_out']), nn.ReLU(inplace=True)
            self._ac_proc = nn.Sequential(linear, ac)
        
        self._discrete_actions = config['policy']['discrete_actions']
        if self._discrete_actions:
            self._ac_bins = config['policy']['ac_bins']
            self._to_logits = nn.Linear(config['policy']['ac_out'], sum(config['policy']['ac_bins']))
        else:
            self._pred_acs = nn.Linear(config['policy']['ac_out'], config['policy']['adim'])
        
        self._aux_dim = config['policy'].get('aux_dim', 0)
        if self._aux_dim:
            self._aux_pred = nn.Linear(self._in_dim , self._aux_dim)
    
    def forward(self, context, images, state):
        context_embed, img_embed = F.normalize(self._embed(context), dim=-1), F.normalize(self._embed(images), dim=-1)
        goal, img_state = self._goal_state_embed(context_embed, img_embed)
        state_goal = torch.cat((self.pe(state), img_state, goal[:, None]), 2)
        if self._stack_len:
            state_goal = self._sg_stack(state_goal)
        
        aux = self._aux_pred(state_goal) if self._aux_dim else None
        return self._predict_actions(state_goal), aux

    def _predict_actions(self, state_goal):
        if self._is_lstm:
            self._ac_proc.flatten_parameters()
            ac_proc = self._ac_proc(state_goal)[0]
        else:
            ac_proc = self._ac_proc(state_goal)

        if self._discrete_actions:
            logits = self._to_logits(ac_proc) + self._prior
            ac_logits, bins = [], logits
            for a in self._ac_bins:
                ac_logits.append(bins[:,:,:a])
                bins = bins[:,:,a:]
            return ac_logits
        else:
            return self._pred_acs(ac_proc)        

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
    trainer = Trainer('bc_tc', "Trains Behavior Clone model on input data", drop_last=True)
    config = trainer.config
    
    # build embedding model
    restore, freeze = config['embedding'].pop('restore', ''), config['embedding'].pop('freeze', False)
    embed = get_model(config['embedding'].pop('type'))
    embed = embed(**config['embedding'])
    if restore:
        restore_model = torch.load(os.path.expanduser(restore), map_location=torch.device('cpu'))
        embed.load_state_dict(restore_model.state_dict(), strict=False)
        del restore_model
    if freeze:
        assert restore, "doesn't make sense to freeze random weights"
        for p in embed.parameters():
            p.requires_grad = False

    # build Imitation Module
    policy = ImitationModule(embed, config)
    policy = policy.to(trainer.device)

    # build loss function
    cross_entropy = nn.CrossEntropyLoss()
    def forward(pi, device, context, traj):
        context = context.to(device)
        grip_location = traj['grip_location'][:,:3].to(device)
        states, images = traj['states'][:,:-1].to(device), traj['images'][:,:-1].to(device)
        actions = traj['actions'].to(device)

        predicted_actions, aux = pi(context, images, states)        
        if isinstance(predicted_actions, list):
            actions = actions.type(torch.long)
            loss, stats = 0, {}
            for d, logits in enumerate(predicted_actions):
                l_d = cross_entropy(logits.view((-1, logits.shape[-1])), actions[:,:,d].view(-1))
                acc = np.mean(np.argmax(logits.detach().cpu().numpy(), 2) == actions[:,:,d].cpu().numpy())
                stats['l_{}'.format(d)], stats['acc_{}'.format(d)] = l_d.item(), acc
                loss = loss + l_d
        else:
            stats = {}
            aux_loss = 0
            if aux is not None:
                aux_loss = torch.mean(torch.sum((grip_location - aux) ** 2, 1))
                aux_pad = aux
                if aux.shape[-1] < predicted_actions.shape[-1]:
                    aux_pad = torch.cat((aux_pad, torch.zeros((aux.shape[0], predicted_actions.shape[-1] - aux.shape[-1])).to(device)), 1)
                predicted_actions = predicted_actions + aux_pad[:,None]
                stats['aux_loss'] = aux_loss.item()

            loss = torch.mean(torch.sum((actions - predicted_actions) ** 2, -1)) + aux_loss * config.get('aux_lambda', 1)
            actions, predicted_actions = actions.cpu().numpy(), predicted_actions.detach().cpu().numpy()
            for d in range(actions.shape[-1]):
                stats['l_{}'.format(d)] = np.mean(np.abs(actions[:,:,d] - predicted_actions[:,:,d]))
        return loss, stats
    trainer.train(policy, forward)
