import torch
from hem.models.contrastive_module import SplitContrastive
from hem.models import Trainer
import numpy as np
import matplotlib.pyplot as plt
from hem.datasets.util import STD, MEAN
_STD, _MEAN = torch.from_numpy(STD.astype(np.float32)).reshape((1, 3, 1, 1)), torch.from_numpy(MEAN.astype(np.float32)).reshape((1, 3, 1, 1))


if __name__ == '__main__':
    trainer = Trainer('split_contrastive', "Trains split contrastive model on input data")
    config = trainer.config
    
    # build Goal VAE
    sc_model = SplitContrastive(**config['model'])
    contrastive_loss = torch.nn.CrossEntropyLoss()
    aux_classification_loss = torch.nn.CrossEntropyLoss()
    aux_pos_loss = torch.nn.SmoothL1Loss()
    temperature = config['temperature']
    def forward(m, device, Q, P, N, aux):
        B = Q.shape[0]
        Q, P, N = [t.to(device) for t in (Q, P, N)]
        order = list(range(Q.shape[0] * 3)); np.random.shuffle(order)
        queries = torch.cat((Q, P, N), 0)[order]
        outputs = {k: v[np.argsort(order)].split(Q.shape[0]) for k, v in m(queries).items()}

        labels = torch.arange(B).to(device)
        sf_Q, sf_P, sf_N = outputs['scene_feat']

        QP_T, QN_T = torch.matmul(sf_Q, sf_P.transpose(0, 1)), torch.matmul(sf_Q, sf_N.transpose(0, 1))
        QQ_T = torch.matmul(sf_Q, sf_Q.transpose(0, 1))
        QQother_T = torch.masked_select(QQ_T, torch.eye(B).view((B,B)).to(device) < 1).view((B, B-1))
        logit_tensor = torch.cat((QP_T, QN_T, QQother_T), 1) / temperature

        l_contrastive = contrastive_loss(logit_tensor, labels)
        l_domain_pred = aux_classification_loss(outputs['pred_domain'][0], aux['is_agent'].long().to(device))
        l_aux_pos = aux_pos_loss(outputs['aux_pred'][0][aux['is_agent']], aux['pos'][aux['is_agent']].to(device))
        loss = l_aux_pos + l_domain_pred + l_contrastive

        stats = {'aux': l_aux_pos.item(), 'domain': l_domain_pred.item(), 'contrastive': l_contrastive.item()}
        stats['domain_acc'] = np.mean(np.argmax(outputs['pred_domain'][0].detach().cpu().numpy(), 1) == aux['is_agent'].long().numpy())
        top_k = torch.topk(logit_tensor, 5, dim=1)[1].detach().cpu().numpy()
        stats['acc_1'] = np.sum(top_k[:,0] == labels.cpu().numpy()) / B
        stats['acc_5'] = np.sum([ar.any() for ar in top_k == labels.cpu().numpy()[:,None]]) / B

        if trainer.is_img_log_step:
            arrs = [P, N, Q]
            all_imgs = []
            for b in np.random.choice(B, size=min(config.get('vis', 16), B), replace=False): 
                img_q = Q[b]
                img_matches = top_k[b][:2]
                
                for i in img_matches:
                    a = int(i // B)
                    idx = i % B
                    idx = idx + 1 if a == 2 and idx >= b else idx
                    img_q = torch.cat((img_q, arrs[a][idx]), 2)
                all_imgs.append(img_q[None])
            stats['img_matches'] = torch.clamp(torch.cat(all_imgs, 0) * _STD.to(device) + _MEAN.to(device), 0, 1)
        return loss, stats
    trainer.train(sc_model, forward)
