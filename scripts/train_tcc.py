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
        U = m(t1)
        V = m(t2)

        B, chosen_i = np.arange(config['batch_size']), np.random.randint(t1.shape[1], size=config['batch_size'])
        deltas = torch.sum((U[B,chosen_i][:,None] - V) ** 2, dim=2)
        v_hat = torch.sum(torch.nn.functional.softmax(-deltas, dim=1)[:,:,None] * V, dim=1)
        class_logits = -torch.sum((v_hat[:,None] - U) ** 2, dim=2)
        
        betas = torch.softmax(class_logits, 1)
        arange = torch.arange(betas.shape[1], dtype=torch.float32).to(device)
        mu = torch.sum(arange[None] * betas, 1)
        sigma_squares = torch.sum(((arange[None] - mu[:,None]) ** 2) * betas, 1)
        mu_error = (mu - torch.from_numpy(chosen_i.astype(np.float32)).to(device)) ** 2
        loss = torch.mean(mu_error / (sigma_squares + 1e-6) + config.get('lambda', 0.5) * torch.log(sigma_squares + 1e-6))
        
        argmaxes = np.argmax(class_logits.detach().cpu().numpy(), 1)
        accuracy_stat = np.sum(argmaxes == chosen_i) / config['batch_size']
        error_stat = np.sqrt(np.sum(np.square(argmaxes - chosen_i))) / config['batch_size']
        mu_error = torch.mean(torch.abs(mu - torch.from_numpy(chosen_i.astype(np.float32)))).item()
        avg_sigma = torch.mean(torch.sum(torch.abs(arange[None] - mu[:,None])  * betas, 1)).item()
        return loss, dict(accuracy=accuracy_stat, error=error_stat, mu=mu_error, sigma=avg_sigma)
    trainer.train(model, forward)
