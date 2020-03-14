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
    
    
    # parser model
    model_class = get_model(config['model'].pop('type'))
    model = model_class(**config['model'])
    cross_entropy = nn.CrossEntropyLoss()

    def forward(m, device, t1, t2, _):
        t1, t2 = t1.to(device), t2.to(device)
        U = model(t1)
        V = model(t2)

        B, chosen_i = np.arange(config['batch_size']), np.random.randint(t1.shape[1], size=config['batch_size'])
        deltas = torch.sum((U[B,chosen_i][:,None] - V) ** 2, dim=2)
        v_hat = torch.sum(torch.nn.functional.softmax(-deltas, dim=1)[:,:,None] * V, dim=1)
        class_logits = -torch.sum((v_hat[:,None] - U) ** 2, dim=2)
        loss = cross_entropy(class_logits, torch.from_numpy(chosen_i).to(device))

        argmaxes = np.argmax(class_logits.detach().cpu().numpy(), 1)
        accuracy_stat = np.sum(argmaxes == chosen_i) / config['batch_size']
        error_stat = np.sqrt(np.sum(np.square(argmaxes - chosen_i))) / config['batch_size']
            
        return loss, dict(accuracy=accuracy_stat, error=error_stat)
    trainer.train(model, forward)
