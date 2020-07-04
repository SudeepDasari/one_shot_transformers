import torch
from hem.models.contrastive_module import ContrastiveImitation
from hem.models import Trainer
from hem.models.discrete_logistic import DiscreteMixLogistic
import numpy as np
import matplotlib.pyplot as plt
from hem.datasets.util import MEAN, STD
import cv2


if __name__ == '__main__':
    trainer = Trainer('bc_gc', "Trains Behavior Clone w/ contrastive goal model on input data")
    config = trainer.config
    
    # build Imitation Module and MDN Loss
    temperature, lambda_c = config.get('temperature', 0.07), config.get('contrastive_loss_weight', 1)
    hard_negs = config.get('n_hard_neg', 3)
    action_model = ContrastiveImitation(**config['policy'])
    contrastive_loss = torch.nn.CrossEntropyLoss()
 
    def forward(m, device, context, traj, append=True):
        states, actions = traj['states'][:,:-1].to(device), traj['actions'].to(device)
        images = traj['images'][:,:-1].to(device) 
        context = context.to(device)

        # compute predictions and action LL
        (mu, ln_scale, logit_prob), embeds = m(states, images, context, hard_negs, ret_dist=False)
        action_distribution = DiscreteMixLogistic(mu, ln_scale, logit_prob)
        neg_ll = torch.mean(-action_distribution.log_prob(actions))
        
        # compute contrastive loss on goal prediction
        batch_queries = torch.cat((embeds['positive'], embeds['negatives']), 1)
        queue = action_model.contrast_queue.transpose(0, 1)[None].repeat((batch_queries.shape[0], 1, 1))
        all_queries = torch.cat((batch_queries, queue), 1)
        logits = torch.matmul(embeds['goal'], all_queries.transpose(1, 2))[:,0] / temperature
        l_contrastive = contrastive_loss(logits, torch.zeros(batch_queries.shape[0]).long().to(device))

        # append to queue if in train mode
        if append:
            action_model.append(batch_queries.detach().reshape((-1, batch_queries.shape[-1])).transpose(0, 1))

        # calculate total loss and statistics
        loss = neg_ll + lambda_c * l_contrastive
        stats = {'neg_ll': neg_ll.item(), 'contrastive_loss': l_contrastive.item()}
        mean_ac = np.clip(action_distribution.mean.detach().cpu().numpy(), -1, 1)
        for d in range(actions.shape[2]):
            stats['l1_{}'.format(d)] = np.mean(np.abs(mean_ac[:,:,d] - actions.cpu().numpy()[:,:,d]))
        top_k = torch.topk(logits.detach(), 5, dim=1)[1].cpu().numpy()
        stats['acc_1'] = np.sum(top_k[:,0] == 0) / batch_queries.shape[0]
        stats['acc_5'] = np.sum([ar.any() for ar in top_k == 0]) / batch_queries.shape[0]
        return loss, stats
    
    def val_forward(m, device, context, traj):
        return forward(m, device, context, traj, append=False)
    trainer.train(action_model, forward, val_fn=val_forward)
