from hem.util import parse_basic_config
import torch
import argparse
from hem.datasets import get_dataset
from hem.models import get_model, Trainer
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import datetime


if __name__ == '__main__':
    trainer = Trainer('tcc', "Trains Temporal Cycle Consistent Embedding on input data")
    config = trainer.config
    
    # build main model
    model_class = get_model(config['model'].pop('type'))
    model = model_class(**config['model'])

    # build "trailing" model
    model_trail = model_class(**config['model'])
    for p, p_trail in zip(model.parameters(), model_trail.parameters()):
        p_trail.data.mul_(0).add_(1, p.detach().data)

    # build loss and queue
    cross_entropy = nn.CrossEntropyLoss()
    V_q = []

    def forward(m, device, t1, t2, _):
        import pdb; pdb.set_trace()
        # copy trailing model to correct device and apply momentum contrast to weights
        model_trail.to(device)
        alpha = config['momentum_alpha']
        assert 0 <= alpha <= 1, "alpha should be in [0,1]!"
        for p, p_trail in zip(model.parameters(), model_trail.parameters()):
            p_trail.data.mul_(alpha).add_(1 - alpha, p.detach().data)
        
        t1, t2 = t1.to(device), t2.to(device)
        U = m(t1)
        with torch.no_grad():
            V_batch = model_trail(t2).detach()
        V_q.append(V_batch.reshape((-1, U.shape[-1])))
        if len(V_q) > config['queue_size']:
            V_q = V_q[1:]
        V = torch.cat(V_q, 0)[None]

        batch_size = t1.shape[0]
        B, chosen_i = np.arange(batch_size), np.random.randint(t1.shape[1], size=batch_size)
        deltas = torch.sum((U[B,chosen_i][:,None] - V) ** 2, dim=2)
        v_hat = torch.sum(torch.nn.functional.softmax(-deltas, dim=1)[:,:,None] * V, dim=1)
        class_logits = -torch.sum((v_hat[:,None] - U) ** 2, dim=2)
        
        loss = cross_entropy(class_logits, torch.from_numpy(chosen_i).to(device))
        argmaxes = np.argmax(class_logits.detach().cpu().numpy(), 1)
        accuracy_stat = np.sum(argmaxes == chosen_i) / batch_size
        error_stat = np.sqrt(np.sum(np.square(argmaxes - chosen_i))) / batch_size
        return loss, dict(accuracy=accuracy_stat, error=error_stat)
    trainer.train(model, forward)
