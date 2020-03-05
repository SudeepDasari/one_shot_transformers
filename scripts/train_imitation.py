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
    parser.add_argument('--device', type=int, default=None, nargs='+', help='target device (uses all if not specified)')
    args = parser.parse_args()
    config = parse_basic_config(args.experiment_file)
    
    # initialize device
    def_device = 0 if args.device is None else args.device[0]
    device = torch.device("cuda:{}".format(def_device))

    # parse dataset
    dataset_class = get_dataset(config['dataset'].pop('type'))
    dataset = dataset_class(**config['dataset'], mode='train')
    val_dataset = dataset_class(**config['dataset'], mode='val')
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('loader_workers', cpu_count()))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    
    # parser model
    embed = get_model(config['embedding'].pop('type'))
    embed = embed(**config['embedding'])
    rnn = get_model(config['rnn'].pop('type'))
    rnn = rnn(**config['rnn'])
    mdn = MixtureDensityTop(**config['mdn'])
    model = nn.Sequential(embed, rnn, mdn)
    if torch.cuda.device_count() > 1 and args.device is None:
        model = nn.DataParallel(model)
    elif args.device is not None and len(args.device) > 1:
        model = nn.DataParallel(model, device_ids=args.device)
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
            states, actions = batch_inputs(pairs, device)
            images = states['images'][:,:-1]
            if 'depth' in states:
                depth = states['depth'][:,:-1]
                inputs = [images, depth]
            else:
                inputs = images

            if config.get('output_state', True):
                actions = torch.cat((states['states'][:,1:], actions[:,:,-1][:,:,None]), 2)
            
            optimizer.zero_grad()
            mean, sigma_inv, alpha = model(inputs)
            l_i = loss(actions, mean, sigma_inv, alpha)
            l_i.backward()
            optimizer.step()
            
            # calculate iter stats
            log_freq = config.get('log_freq', 20)
            mod_step = step % log_freq
            loss_stat = (l_i.item() + mod_step * loss_stat) / (mod_step + 1)
                        
            if mod_step == log_freq - 1:
                try:
                    val_pairs, _ = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_pairs, _ = next(val_iter)

                with torch.no_grad():
                    states, actions = batch_inputs(val_pairs, device)
                    images = states['images'][:,:-1]
                    if 'depth' in states:
                        depth = states['depth'][:,:-1]
                        inputs = [images, depth]
                    else:
                        inputs = images
                    mean, sigma_inv, alpha = model(inputs)
                    val_l = loss(actions, mean, sigma_inv, alpha)

                writer.add_scalar('loss/val', val_l.item(), step)
                writer.add_scalar('loss/train', loss_stat, step)
                writer.file_writer.flush()
                print('step {0}: loss={1:.4f} \t val loss={1:.4f}'.format(step, loss_stat, val_l.item()))
            else:
                print('step {0}: loss={1:.4f}'.format(step, loss_stat))
            step += 1

            if step % config.get('save_freq', 5000) == 0:
                torch.save(model, save_path + '-{}.pt'.format(step))
