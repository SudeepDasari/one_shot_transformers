import torch
from hem.datasets import get_dataset
from hem.models import get_model, Trainer
import torch.nn as nn
import numpy as np
import copy


class _AdvMoCoWrapper(nn.Module):
    def __init__(self, mq, mk, c, alpha):
        super().__init__()
        self._alpha = alpha
        self._mq = mq
        self._mk = mk
        self._c = c
        for p_q, p_k in zip(mq.parameters(), mk.parameters()):
            p_k.data.mul_(0).add_(1, p_q.detach().data)

    def forward(self, b1, b2):
        q = self._mq(b1)
        with torch.no_grad():
            k = self._mk(b2).detach()
        q = nn.functional.normalize(q, dim=1)
        return q, nn.functional.normalize(k, dim=1), self._c(q)

    def momentum_update(self):
        with torch.no_grad():
            for p_q, p_k in zip(self._mq.parameters(), self._mk.parameters()):
                p_k.data.mul_(self._alpha).add_(1 - self._alpha, p_q.detach().data)

    def save_module(self):
        return self._mq

    def p1_params(self):
        return self._mq.parameters()
    
    def p2_params(self):
        return self._c.parameters()


class AgentClassifier(nn.Module):
    def __init__(self, layer_def, n_agents=2):
        super().__init__()
        layers = []
        for p, n in zip(layer_def[:-1], layer_def[1:]):
            layers.append(nn.Linear(p, n))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(layer_def[-1], n_agents))
        self._n = nn.Sequential(*layers)
    
    def forward(self, x):
        return self._n(x)


if __name__ == '__main__':
    trainer = Trainer('traj_MoCo', "Trains Adversarial Trajectory MoCo on input data", comp_grad=False, drop_last=True)
    config = trainer.config
    
    # get MoCo params
    queue_size = config['moco_queue_size']
    assert queue_size % config['batch_size'] == 0, "queue_size should be divided by batch_size evenly"
    alpha, temperature = config.get('momentum_alpha', 0.999), config.get('temperature', 0.07)
    assert 0 <= alpha <= 1, "alpha should be in [0,1]!"

    # build classifier module
    c = AgentClassifier(**config['classifier'])

    # build main model
    model_class = get_model(config['model'].pop('type'))
    model = model_class(**config['model'])
    moco_model = _AdvMoCoWrapper(model, model_class(**config['model']), c, alpha)

    # initialize queue
    moco_queue = torch.randn((model.dim, queue_size), dtype=torch.float32).to(trainer.device)
    moco_ptr, last_k = 0, None

    # build loss_fn
    cross_entropy = torch.nn.CrossEntropyLoss()

    def train_forward(model, device, b1, l1, b2):
        import pdb; pdb.set_trace()
        global moco_queue, moco_ptr, temperature, last_k
        if last_k is not None:
            moco_queue[:,moco_ptr:moco_ptr+b1.shape[0]] = last_k
            moco_ptr = (moco_ptr + config['batch_size']) % moco_queue.shape[1]

        moco_model.momentum_update()
        labels = torch.zeros(b1.shape[0], dtype=torch.long).to(device)

        # order for shuffled bnorm
        order = list(range(config['batch_size']))
        np.random.shuffle(order)
        b1, b2 = b1.to(device), b2.to(device)
        q, k, pred_agent = model(b1, b2[order])
        k = k[np.argsort(order)]

        l_pos = torch.matmul(q[:,None], k[:,:,None])[:,0]
        l_neg = torch.matmul(q, moco_queue)
        logits = torch.cat((l_pos, l_neg), 1) / temperature

        loss_class = cross_entropy(pred_agent, l1)
        loss_embed = cross_entropy(logits, labels) - loss_class

        last_k = k.transpose(1, 0)
        top_k = torch.topk(logits, 5, dim=1)[1].cpu().numpy()
        acc_1 = np.sum(top_k[:,0] == 0) / b1.shape[0]
        acc_5 = np.sum([ar.any() for ar in top_k == 0]) / b1.shape[0]
        return [loss_embed, loss_class], {'acc1': acc_1, 'acc5': acc_5}
    
    def val_forward(model, device, b1, l1, b2):
        global last_k, train_forward
        rets = train_forward(model, device, b1, l1, b2)
        last_k = None
        return rets
    trainer.train(moco_model, train_forward, moco_model.wrapped_model, val_fn=val_forward)
