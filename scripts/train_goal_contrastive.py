import torch
from hem.models.contrastive_module import GoalContrastive
from hem.models import Trainer
import numpy as np
import matplotlib.pyplot as plt
from hem.datasets.util import STD, MEAN
_STD, _MEAN = torch.from_numpy(STD.astype(np.float32)).reshape((1, 3, 1, 1)), torch.from_numpy(MEAN.astype(np.float32)).reshape((1, 3, 1, 1))


if __name__ == '__main__':
    trainer = Trainer('goal_contrastive', "Trains goal contrastive model on input data")
    config = trainer.config
    
    # build Goal VAE
    sc_model = GoalContrastive(**config['model'])
    contrastive_loss = torch.nn.CrossEntropyLoss()
    temperature = config['temperature']
    def forward(m, device, Q, P, N):
        Q, P, N = [t.to(device) for t in (Q, P, N)]
        PN = torch.cat((P[:,None], N), 1)
        order = list(range(PN.shape[0] * PN.shape[1])); np.random.shuffle(order)

        Q_embed = m(Q)
        PN_embed = m(PN.reshape((PN.shape[0] * PN.shape[1], PN.shape[2], PN.shape[3], PN.shape[4]))[order])
        PN_embed = PN_embed[np.argsort(order)].reshape((PN.shape[0], PN.shape[1], PN_embed.shape[-1]))

        logits = torch.matmul(Q_embed[:,None], PN_embed.transpose(1, 2))[:,0] / temperature        
        loss = contrastive_loss(logits, torch.zeros(P.shape[0], dtype=torch.long).to(device))

        stats = {}
        top_k = torch.topk(logits.detach(), 5, dim=1)[1].cpu().numpy()
        stats['acc_1'] = np.sum(top_k[:,0] == 0) / Q.shape[0]
        stats['acc_5'] = np.sum([ar.any() for ar in top_k == 0]) / Q.shape[0]
        return loss, stats

    trainer.train(sc_model, forward)
