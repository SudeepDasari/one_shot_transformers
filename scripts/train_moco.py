import torch
from hem.datasets import get_dataset
from hem.models import get_model, Trainer
import torch.nn as nn
import numpy as np
import copy
import os


class _MocoQueue(nn.Module):
    def __init__(self, dim, queue_size):
        super().__init__()
        self._moco_ptr = 0
        moco_queue = torch.randn((dim, queue_size), dtype=torch.float32)
        self.register_buffer('moco_queue',  moco_queue)

    def append(self, last_k):
        assert self.moco_queue.shape[1] % last_k.shape[1] == 0, "key shape must divide moco_queue!"
        self.moco_queue[:,self._moco_ptr:self._moco_ptr+last_k.shape[1]] = last_k
        self._moco_ptr = (self._moco_ptr + last_k.shape[1]) % self.moco_queue.shape[1]


class _MoCoWrapper(nn.Module):
    def __init__(self, mq, mk, alpha, queue_size):
        super().__init__()
        self._alpha = alpha
        self._mq = mq
        self._mk = mk
        self._moco_queue = _MocoQueue(self._mq.dim, queue_size)
        for p_q, p_k in zip(mq.parameters(), mk.parameters()):
            p_k.data.mul_(0).add_(1, p_q.detach().data)

    def forward(self, b1, b2):
        q = self._mq(b1)
        with torch.no_grad():
            k = self._mk(b2).detach()
        return nn.functional.normalize(q, dim=1), nn.functional.normalize(k, dim=1)

    def momentum_update(self):
        with torch.no_grad():
            for p_q, p_k in zip(self._mq.parameters(), self._mk.parameters()):
                p_k.data.mul_(self._alpha).add_(1 - self._alpha, p_q.detach().data)

    def save_fn(self, save_path, step):
        torch.save(self._mq, save_path  + '-{}.pt'.format(step))
        torch.save(self._moco_queue, save_path  + '-queue-{}.pt'.format(step))

    @property
    def moco_queue(self):
        return self._moco_queue.moco_queue
    
    def enqueue(self, last_k):
        self._moco_queue.append(last_k)


if __name__ == '__main__':
    trainer = Trainer('traj_MoCo', "Trains Trajectory MoCo on input data", drop_last=True)
    config = trainer.config
    
    # get MoCo params
    queue_size = config['moco_queue_size']
    assert queue_size % config['batch_size'] == 0, "queue_size should be divided by batch_size evenly"
    alpha, temperature = config.get('momentum_alpha', 0.999), config.get('temperature', 0.07)
    assert 0 <= alpha <= 1, "alpha should be in [0,1]!"

    # build main model
    model_class = get_model(config['model'].pop('type'))
    model = model_class(**config['model'])
    moco_model = _MoCoWrapper(model, model_class(**config['model']), alpha, queue_size)

    # initialize queue
    last_k = None

    # build loss_fn
    cross_entropy = torch.nn.CrossEntropyLoss()

    def train_forward(model, device, b1, b2, hard_negatives=None):
        global temperature, last_k
        if last_k is not None:
            model.enqueue(last_k)

        moco_model.momentum_update()
        labels = torch.zeros(b1.shape[0], dtype=torch.long).to(device)

        # order for shuffled bnorm
        b1, b2 = b1.to(device), b2.to(device)
        if hard_negatives is not None:
            b2 = torch.cat((b2, hard_negatives.to(device)), 0)
        order = list(range(b2.shape[0])); np.random.shuffle(order)
        q, k = model(b1, b2[order])
        k = k[np.argsort(order)]

        l_neg = torch.matmul(q, model.moco_queue)
        if hard_negatives is not None:
            k, hn = k[:b1.shape[0]], k[b1.shape[0]:]
            l_neg = torch.cat((torch.matmul(q[:,None], hn[:,:,None])[:,0], l_neg), 1)
        l_pos = torch.matmul(q[:,None], k[:,:,None])[:,0]
        logits = torch.cat((l_pos, l_neg), 1) / temperature
        loss = cross_entropy(logits, labels)

        last_k = k.transpose(1, 0)
        if hard_negatives is not None:
            last_k = torch.cat((last_k, hn.transpose(1, 0)), 1)
        top_k = torch.topk(logits, 5, dim=1)[1].cpu().numpy()
        acc_1 = np.sum(top_k[:,0] == 0) / b1.shape[0]
        acc_5 = np.sum([ar.any() for ar in top_k == 0]) / b1.shape[0]
        return loss, {'acc1': acc_1, 'acc5': acc_5}
    
    def val_forward(model, device, b1, b2, hard_negatives=None):
        global last_k, train_forward
        rets = train_forward(model, device, b1, b2, hard_negatives)
        last_k = None
        return rets
    trainer.train(moco_model, train_forward, save_fn=moco_model.save_fn, val_fn=val_forward)
