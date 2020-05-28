import torch
from hem.models.imitation_module import LatentStateImitation, _VidPrior
from hem.models import Trainer
from hem.models.mdn_loss import GMMDistribution
import numpy as np


if __name__ == '__main__':
    trainer = Trainer('state_imcontext_bc', "Trains Behavior Clone model on input data")
    config = trainer.config

    # build Imitation Module and MDN Loss
    action_model = LatentStateImitation(**config['policy'])
    if config.get('action_restore', ''):
        restore_weights = torch.load(config.get('action_restore'), map_location=torch.device('cpu')).state_dict()
        action_model.load_state_dict(restore_weights, strict=False)

    # build image prior and restore weights if needed
    if config.get('replace_prior', False):
        img_prior = _VidPrior(**config['policy']['vid_prior'])
        action_model.replace_prior(img_prior)
    optim_weights = action_model.parameters() if config.get('train_both', True) else img_prior.parameters()

    # build losses
    l1_loss = torch.nn.SmoothL1Loss()
    aux_l1_loss = torch.nn.SmoothL1Loss()

    def get_kl_beta(config, step):
        beta = config['kl_beta']
        assert beta >=0
        if 'kl_anneal' in config:
            beta_0, start, end = config['kl_anneal']
            assert end > start and beta_0 >= 0
            alpha = min(max(float(step - start) / (end - start), 0), 1)
            beta = (1 - alpha) * beta_0 + alpha * beta
        return beta

    def forward(m, device, states, actions, x_len, loss_mask, aux_loc, aux_mask, im_context):
        states, actions, x_len, loss_mask = states.to(device), actions.to(device), x_len.to(device), loss_mask.to(device)
        aux_loc, aux_mask, im_context = aux_loc.to(device), aux_mask.to(device), im_context.to(device)

        pred_acs, (kl, aux) = m(states, actions, x_len, False, force_ss=True, context=im_context, sample_prior=not config.get('train_both', True))
        if len(pred_acs) == 3:
            mu, sigma_inv, alpha = pred_acs
            action_distribution = GMMDistribution(mu, sigma_inv, alpha)
            recon_loss = torch.sum(-action_distribution.log_prob(actions) * loss_mask / torch.sum(loss_mask))
            pred_acs = action_distribution.mean.detach().cpu().numpy()
        else:
            recon_loss = torch.sum(l1_loss(pred_acs, actions) * loss_mask / torch.sum(loss_mask))
            pred_acs = pred_acs.detach().cpu().numpy()

        aux_loss = torch.sum(aux_l1_loss(aux, aux_loc) * aux_mask / (torch.sum(aux_mask) + 1e-3)) if aux is not None else 0
        kl = torch.mean(kl)
        kl_beta = get_kl_beta(config, trainer.step)
        loss = config.get('recon_lambda', 1) * recon_loss + kl_beta * kl + aux_loss
        stats = {'recon_loss': recon_loss.item(), 'kl': kl.item(), 'schedule_samp': action_model.ss_p, 'kl_beta': kl_beta}
        if aux is not None:
            stats['aux_loss'] = aux_loss.item()

        real_ac, mask = actions.cpu().numpy(), loss_mask.cpu().numpy()
        for d in range(actions.shape[2]):
            stats['l1_{}'.format(d)] = np.sum(np.abs(pred_acs[:,:,d] - real_ac[:,:,d]) * mask / np.sum(mask))
        
        if action_model.training:
            action_model.increment_ss()

        return loss, stats        
    trainer.train(action_model, forward, optim_weights=optim_weights)
