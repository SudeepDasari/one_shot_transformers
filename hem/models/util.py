import torch


def batch_inputs(pairs, device='cpu'):
    device = torch.device(device)
    n_actions = int((len(pairs) - 1) / 2)
    
    actions = torch.cat([pairs['a_{}'.format(t)][:,None] for t in range(1, n_actions + 1)], 1).to(device)
    states = dict(
                images=torch.cat([pairs['s_{}'.format(t)]['image'][:,None] for t in range(n_actions + 1)], 1).to(device),
                states=torch.cat([pairs['s_{}'.format(t)]['state'][:,None] for t in range(n_actions + 1)], 1).to(device)
            )
    if 'depth' in pairs['s_0']:
        states['depth'] = torch.cat([pairs['s_{}'.format(t)]['depth'][:,None] for t in range(n_actions + 1)], 1).to(device)
    return states, actions
