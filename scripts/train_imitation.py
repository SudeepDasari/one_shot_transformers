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
    parser.add_argument('--device', type=int, default=None, help='target device (uses all if not specified)')
    args = parser.parse_args()
    config = parse_basic_config(args.experiment_file)
    
    # initialize device
    def_device = 0 if args.device is None else args.device
    device = torch.device("cuda:{}".format(def_device))

    # parse dataset
    dataset_class = get_dataset(config['dataset'].pop('type'))
    dataset = dataset_class(**config['dataset'], mode='train')
    val_dataset = dataset_class(**config['dataset'], mode='val')
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('loader_workers', cpu_count()))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    
    # parser model
    model_class = get_model(config['model'].pop('type'))
    base_model = model_class(**config['model'])
    mdn = MixtureDensityTop(**config['mdn'])
    model = nn.Sequential(base_model, mdn)
    if torch.cuda.device_count() > 1 and args.device is None:
        model = nn.DataParallel(model)
    model.to(device)

    # optimizer
    loss = MixtureDensityLoss()
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    writer = SummaryWriter(log_dir=config.get('summary_log_dir', './bc_log_{}-{}_{}-{}-{}'.format(now.hour, now.minute, now.day, now.month, now.year)))
    save_path = config.get('save_path', './bc_weights_{}-{}_{}-{}-{}'.format(now.hour, now.minute, now.day, now.month, now.year))
    n_saves = 0

    step = 0
    loss_stat = 0
    val_iter = iter(val_loader)
    for _ in range(config.get('epochs', 10)):
        for pairs, _ in train_loader:
            optimizer.zero_grad()
            states, actions = batch_inputs(pairs, device)
            mean, sigma_inv, alpha = model(states['images'][:,:-1])
            l_i = loss(actions, mean, sigma_inv, alpha)
            l_i.backward()
            optimizer.step()
            
            # calculate iter stats
            mod_step = step % config.get('log_freq', 10)
            loss_stat = (l_i.item() + mod_step * loss_stat) / (mod_step + 1)
            
            end = '\r'
            if mod_step == config.get('log_freq', 50) - 1:
                try:
                    val_pairs, _ = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_pairs, _ = next(val_iter)
                states, actions = batch_inputs(val_pairs, device)
                mean, sigma_inv, alpha = model(states['images'][:,:-1])
                val_l = loss(actions, mean, sigma_inv, alpha)

                writer.add_scalar('loss/val', val_l.item(), step)
                writer.add_scalar('loss/train', loss_stat, step)
                writer.file_writer.flush()
                end = '\n'
            
            print('step {0}: loss={1:.4f}'.format(step, loss_stat), end=end)
            step += 1

            if step % config.get('save_freq', 5000) == 0:
                torch.save(model.state_dict(), save_path + '-{}'.format(step))
