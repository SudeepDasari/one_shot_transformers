from hem.parse_util import parse_basic_config
import torch
import argparse
from hem.datasets import get_dataset
from hem.models import get_model
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument('experiment_file', type=str, help='path to YAML experiment config file')
    args = parser.parse_args()
    config = parse_basic_config(args.experiment_file)
    
    # parse dataset
    dataset_class = get_dataset(config['dataset'].pop('type'))
    dataset = dataset_class(**config['dataset'], mode='train')
    val_dataset = dataset_class(**config['dataset'], mode='validation')
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=min(cpu_count(), config.get('n_workers', cpu_count()))) 
    
    # parser model
    model_class = get_model(config['model'].pop('type'))
    model = model_class(**config['model'])
    model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    writer = SummaryWriter(log_dir=config.get('summary_log_dir', None))
    i = 0
    for t1, t2 in train_loader:
        optimizer.zero_grad()
        U = model(t1.cuda())
        V = model(t2.cuda())

        B, chosen_i = np.arange(config['batch_size']), np.random.randint(t1.shape[1], size=config['batch_size'])
        deltas = torch.sum((U[B,chosen_i][:,None] - V) ** 2, dim=2)
        v_hat = torch.sum(torch.nn.functional.softmax(-deltas, dim=1)[:,:,None] * V, dim=1)
        class_logits = -torch.sum((v_hat[:,None] - U) ** 2, dim=2)
        
        betas = torch.nn.functional.softmax(class_logits, dim=1)
        mu = torch.sum(betas * torch.arange(t1.shape[1]).cuda().view((1, -1)), dim=1)
        sigma_square = torch.sum(betas * (torch.arange(t1.shape[1]).cuda()[None] - mu.view((-1, 1))) ** 2, dim=1)
        delta_loss = ((torch.from_numpy(chosen_i).cuda() - mu) ** 2) / sigma_square 
        log_std_loss = config['log_sigma_weight'] * torch.log(sigma_square) / 2
        
        loss = torch.sum(delta_loss + log_std_loss) / config['batch_size']
        loss.backward()
        optimizer.step()
        i += 1

        print(loss.item())
        writer.add_scalar('Loss/train', loss.item(), i)
        writer.add_scalar('DeltaLoss/train', torch.sum(delta_loss).item() / config['batch_size'], i)
        writer.file_writer.flush()
