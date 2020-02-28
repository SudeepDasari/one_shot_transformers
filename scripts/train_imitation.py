from hem.util import parse_basic_config
import torch
import argparse
from hem.datasets import get_dataset
from hem.models import get_model
from hem.models.util import batch_inputs
from hem.models.mdn_loss import MixtureDensityLoss, MixtureDensityTop
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import datetime


if __name__ == '__main__':
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument('experiment_file', type=str, help='path to YAML experiment config file')
    args = parser.parse_args()
    config = parse_basic_config(args.experiment_file)
    
    # initialize device
    device = torch.device("cuda:0")

    # parse dataset
    dataset_class = get_dataset(config['dataset'].pop('type'))
    dataset = dataset_class(**config['dataset'], mode='train')
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('loader_workers', cpu_count()))
    
    # parser model
    model_class = get_model(config['model'].pop('type'))
    model = model_class(**config['model'])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    mdn = MixtureDensityTop(**config['mdn'])
    mdn.to(device)

    # optimizer
    loss = MixtureDensityLoss()
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    writer = SummaryWriter(log_dir=config.get('summary_log_dir', './bc_log_{}-{}_{}-{}-{}'.format(now.hour, now.minute, now.day, now.month, now.year)))
    save_path = config.get('save_path', './bc_weights_{}-{}_{}-{}-{}'.format(now.hour, now.minute, now.day, now.month, now.year))
    n_saves = 0

    step = 0
    loss_stat = 0
    for _ in range(config.get('epochs', 10)):
        for pairs, _ in train_loader:
            optimizer.zero_grad()
            states, actions = batch_inputs(pairs, device)
            mean, sigma_inv, alpha = mdn(model(states['images'][:,:-1]))
            l_i = loss(actions, mean, sigma_inv, alpha)
            l_i.backward()
            optimizer.step()
            
            # calculate iter stats
            mod_step = step % config.get('log_freq', 10)
            loss_stat = (l_i.item() + mod_step * loss_stat) / (mod_step + 1)
            
            end = '\r'
            if mod_step == config.get('log_freq', 10) - 1:
                writer.add_scalar('loss/train', loss_stat, step)
                writer.file_writer.flush()
                end = '\n'
            
            print('step {0}: loss={1:.4f}'.format(step, loss_stat), end=end)
            step += 1

            if step % config.get('save_freq', 10000) == 0:
                torch.save(model.state_dict(), save_path + '-{}'.format(n_saves))
                n_saves += 1