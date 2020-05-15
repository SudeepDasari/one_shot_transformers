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


def classification_loss(chosen_i, class_logits, device='cuda:0'):
    batch_size = class_logits.shape[0]
    loss = torch.nn.functional.cross_entropy(class_logits, torch.from_numpy(chosen_i).to(device))
    argmaxes = np.argmax(class_logits.detach().cpu().numpy(), 1)
    accuracy_stat = np.sum(argmaxes == chosen_i) / batch_size
    error_stat = np.sqrt(np.sum(np.square(argmaxes - chosen_i))) / batch_size
    return loss, dict(accuracy=accuracy_stat, error=error_stat)


def regression_loss(chosen_i, class_logits, device='cuda:0', sigma_lambda=0.1):
    betas = torch.softmax(class_logits, 1)
    arange = torch.arange(betas.shape[1], dtype=torch.float32).to(device)
    mu = torch.sum(arange[None] * betas, 1)
    sigma_squares = torch.sum(((arange[None] - mu[:,None]) ** 2) * betas, 1)
    mu_error = (mu - torch.from_numpy(chosen_i.astype(np.float32)).to(device)) ** 2
    loss = torch.mean(mu_error / (sigma_squares + 1e-4) + sigma_lambda * torch.log(sigma_squares + 1e-4))
    
    batch_size = class_logits.shape[0]
    argmaxes = np.argmax(class_logits.detach().cpu().numpy(), 1)
    accuracy_stat = np.sum(argmaxes == chosen_i) / batch_size
    error_stat = np.sqrt(np.sum(np.square(argmaxes - chosen_i))) / batch_size
    mu_error = torch.mean(torch.abs(mu - torch.from_numpy(chosen_i.astype(np.float32)).to(device))).item()
    avg_sigma = torch.mean(torch.sum(torch.abs(arange[None] - mu[:,None])  * betas, 1)).item()
    return loss, dict(accuracy=accuracy_stat, error=error_stat, mu=mu_error, sigma=avg_sigma)


if __name__ == '__main__':
    trainer = Trainer('tcc', "Trains Temporal Cycle Consistent Embedding on input data")
    config = trainer.config
    
    # parser model
    model_class = get_model(config['model'].pop('type'))
    model = model_class(**config['model'])

    def forward(m, device, t1, t2):
        t1, t2 = t1.to(device), t2.to(device)
        U = m(t1)
        V = m(t2)

        batch_size = t1.shape[0]
        B, chosen_i = np.arange(batch_size), np.random.randint(t1.shape[1], size=batch_size)
        deltas = torch.sum((U[B,chosen_i][:,None] - V) ** 2, dim=2)
        v_hat = torch.sum(torch.nn.functional.softmax(-deltas, dim=1)[:,:,None] * V, dim=1)
        class_logits = -torch.sum((v_hat[:,None] - U) ** 2, dim=2)
        
        if config['loss_type'] == 'regression':
            return regression_loss(chosen_i, class_logits, device, config.get('lambda', 0.1))
        return classification_loss(chosen_i, class_logits, device)
    trainer.train(model, forward)
