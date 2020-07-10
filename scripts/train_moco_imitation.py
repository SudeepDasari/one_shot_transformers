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
    temperature = config.get('temperature', 0.07)
    lambda_c, lambda_ll = config.get('contrastive_loss_weight', 1), config.get('likelihood_loss_weight', 1)
    lambda_p = config.get('point_loss_weight', 0.5)
    n_noise, hard_neg_frac = config.get('n_hard_neg', 5), config.get('hard_neg_frac', 0.5)
    action_model = ContrastiveImitation(**config['policy'])
    contrastive_loss = torch.nn.CrossEntropyLoss()
    point_loss = torch.nn.CrossEntropyLoss()
 
    def forward(m, device, context, traj, append=True):
        states, actions = traj['states'].to(device), traj['actions'].to(device)
        images, pnts = traj['images'].to(device), traj['points']
        transformed_imgs = traj['transformed'].to(device)
        pnt_labels = torch.argmax(pnts.reshape((pnts.shape[0], pnts.shape[1], -1)), dim=2).to(device)
        context = context.to(device)

        # compute predictions and action LL
        (mu, ln_scale, logit_prob), embeds = m(states, images, context, ret_dist=False)
        action_distribution = DiscreteMixLogistic(mu[:,:-1], ln_scale[:,:-1], logit_prob[:,:-1])
        neg_ll = torch.mean(-action_distribution.log_prob(actions))
    
        # compute contrastive loss on images
        img_embed = embeds['img_embed']
        trans_flat = transformed_imgs.reshape((-1, 1, transformed_imgs.shape[2], transformed_imgs.shape[3], transformed_imgs.shape[4]))
        order = list(range(trans_flat.shape[0])); np.random.shuffle(order)
        trans_embed = m(None, trans_flat[order], None, only_embed=True)[np.argsort(order)]
        trans_embed = trans_embed.reshape((transformed_imgs.shape[0], transformed_imgs.shape[1], -1))
        fr_logits = torch.matmul(img_embed.reshape(-1, img_embed.shape[-1]), trans_embed.reshape(-1, trans_embed.shape[-1]).transpose(0, 1)) / temperature
        fr_labels = torch.arange(img_embed.shape[0] * img_embed.shape[1]).to(device)
        frame_contrastive = contrastive_loss(fr_logits, fr_labels)

        # sample positives and negatives for goal prediction
        positive = trans_embed[:,-1:]
        n_hard_neg = max(int(trans_embed.shape[1] * hard_neg_frac), 1)
        assert n_noise <= n_hard_neg, "{} hard negatives available but asked to sample {}!".format(n_hard_neg, n_noise)
        chosen = torch.multinomial(torch.ones((images.shape[0], n_hard_neg)), n_noise, replacement=False)
        negatives = trans_embed[torch.arange(images.shape[0])[:,None], chosen]

        # compute contrastive loss on goal prediction
        batch_queries = torch.cat((positive, negatives), 1)
        batch_queries = batch_queries.reshape((1, -1, batch_queries.shape[-1]))
        logits = torch.matmul(embeds['goal'], batch_queries.transpose(1, 2))[:,0] / temperature
        labels = torch.arange(logits.shape[0]) * (n_noise + 1)
        goal_contrastive = contrastive_loss(logits, labels.to(device))
        l_contrastive = frame_contrastive + goal_contrastive

        # compute point loss
        l_points = point_loss(embeds['goal_point'], pnt_labels[:,-1])

        # calculate total loss and statistics
        loss = lambda_ll * neg_ll + lambda_c * l_contrastive + lambda_p * l_points
        stats = {'neg_ll': neg_ll.item(), 'contrastive_loss': l_contrastive.item(), 'l_points': l_points.item(), 'frame': frame_contrastive.item(), 'goal': goal_contrastive.item()}
        mean_ac = np.clip(action_distribution.mean.detach().cpu().numpy(), -1, 1)
        for d in range(actions.shape[2]):
            stats['l1_{}'.format(d)] = np.mean(np.abs(mean_ac[:,:,d] - actions.cpu().numpy()[:,:,d]))
        
        # calculate frame stats
        top_k = torch.topk(fr_logits.detach(), 5, dim=1)[1].cpu().numpy()
        stats['fr_acc_1'] = np.sum(top_k[:,0] == fr_labels.cpu().numpy()) / fr_labels.shape[0]
        stats['fr_acc_5'] = np.sum([ar.any() for ar in top_k == fr_labels.cpu().numpy()[:,None]]) / fr_labels.shape[0]

        # calculate goal stats
        top_k = torch.topk(logits.detach(), 5, dim=1)[1].cpu().numpy()
        stats['goal_acc_1'] = np.sum(top_k[:,0] == labels.cpu().numpy()) / labels.shape[0]
        stats['goal_acc_5'] = np.sum([ar.any() for ar in top_k == labels.cpu().numpy()[:,None]]) / labels.shape[0]

        if trainer.is_img_log_step:
            vis_goal_point = torch.nn.functional.softmax(embeds['goal_point'].detach(), dim=1).reshape((-1, 1, 24, 32))
            real = torch.cat((torch.zeros_like(pnts[:,-1:]), pnts[:,-1:], torch.zeros_like(pnts[:,-1:])), 1)
            pred = torch.cat((vis_goal_point, torch.zeros_like(vis_goal_point), torch.zeros_like(vis_goal_point)), 1)
            stats['points'] = real.cpu() + pred.cpu()
        return loss, stats
    def val_forward(m, device, context, traj):
        return forward(m, device, context, traj, append=False)
    trainer.train(action_model, forward, val_fn=val_forward)
