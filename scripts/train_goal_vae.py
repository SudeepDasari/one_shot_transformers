import torch
from hem.models.image_generator import GoalVAE
from hem.models import Trainer
import numpy as np
import matplotlib.pyplot as plt
from hem.util import get_kl_beta
from torch.distributions import Laplace


if __name__ == '__main__':
    trainer = Trainer('goal_vae', "Trains goal image model on input data")
    config = trainer.config
    
    # build Goal VAE
    gvae = GoalVAE(**config['model'])
    cross_entropy = torch.nn.CrossEntropyLoss()
    aux_pos_loss = torch.nn.SmoothL1Loss()
    def forward(m, device, context, targets):
        start, goal, targets = context['start'].to(device), context['goal'].to(device), targets.to(device)
        arm_label, arm_pos = context['is_agent'].to(device), context['pos'].to(device)
        m_dict = m(start, arm_label, goal)
        recon_ll, kl = torch.mean(Laplace(m_dict['pred'], 1).log_prob(targets)), torch.mean(m_dict['kl'])
        aux_class_pred = cross_entropy(m_dict['aux_label'], arm_label)
        aux_pos_pred = aux_pos_loss(m_dict['aux_pos'][arm_label.bool()], arm_pos[arm_label.bool()])

        beta = get_kl_beta(config, trainer.step)
        loss = beta * kl - recon_ll + config['alpha_pos'] * aux_pos_pred + config['alpha_class'] * aux_class_pred

        target_vis = torch.clamp(targets.detach(), 0, 1)
        pred_vis = torch.clamp(m_dict['pred'].detach(), 0, 1)
        classifier_acc = np.mean(np.sum(np.argmax(m_dict['aux_label'].detach().cpu().numpy(), 1) == arm_label.cpu().numpy()))
        stats = {'kl': kl.item(), 
                'kl_beta': beta,
                'recon_ll': recon_ll.item(), 
                'l1': torch.mean(torch.abs(m_dict['pred'].detach() - targets)).item(), 
                'real_vs_pred': torch.cat((target_vis, pred_vis), 2),
                'classifier_loss': aux_class_pred.item(),
                'pos_loss': aux_pos_pred.item(),
                'classifier_acc': classifier_acc}
        return loss, stats        
    trainer.train(gvae, forward)
