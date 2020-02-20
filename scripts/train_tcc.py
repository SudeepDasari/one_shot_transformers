from hem.util import parse_basic_config
import torch
import argparse
from hem.datasets import get_dataset
from hem.models import get_model
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

    # optimizer
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    writer = SummaryWriter(log_dir=config.get('summary_log_dir', './tcc_log_{}-{}_{}-{}-{}'.format(now.hour, now.minute, now.day, now.month, now.year)))
    save_path = config.get('save_path', './tcc_weights_{}-{}_{}-{}-{}'.format(now.hour, now.minute, now.day, now.month, now.year))
    n_saves = 0

    step = 0
    loss_stat, accuracy_stat, error_stat = 0, 0, 0
    for _ in range(config.get('epochs', 10)):
        for t1, t2 in train_loader:
            t1, t2 = t1[1].to(device), t2[1].to(device)
            optimizer.zero_grad()

            U = model(t1)
            V = model(t2)

            B, chosen_i = np.arange(config['batch_size']), np.random.randint(t1.shape[1], size=config['batch_size'])
            deltas = torch.sum((U[B,chosen_i][:,None] - V) ** 2, dim=2)
            v_hat = torch.sum(torch.nn.functional.softmax(-deltas, dim=1)[:,:,None] * V, dim=1)
            class_logits = -torch.sum((v_hat[:,None] - U) ** 2, dim=2)
            
            loss = cross_entropy(class_logits, torch.from_numpy(chosen_i).to(device)) 
            loss.backward()
            optimizer.step()
            
            # calculate iter stats
            mod_step = step % config.get('log_freq', 10)
            loss_stat = (loss.item() + mod_step * loss_stat) / (mod_step + 1)
            argmaxes = np.argmax(class_logits.detach().cpu().numpy(), 1)
            accuracy_stat = (np.sum(argmaxes == chosen_i) / config['batch_size'] + mod_step * accuracy_stat) / (mod_step + 1)
            error_stat = (np.sqrt(np.sum(np.square(argmaxes - chosen_i))) / config['batch_size'] + mod_step * error_stat) / (mod_step + 1)
            
            end = '\r'
            if mod_step == config.get('log_freq', 10) - 1:
                writer.add_scalar('loss/train', loss_stat, step)
                writer.add_scalar('accuracy/train', accuracy_stat, step)
                writer.add_scalar('error/train', error_stat, step)
                writer.file_writer.flush()
                end = '\n'
            
            print('step {0}: loss={1:.4f} \t\t accuracy={2:2.3f} \t\t error={3:2.3f}'.format(step, loss_stat, accuracy_stat, error_stat), end=end)
            step += 1

            if step % config.get('save_freq', 10000) == 0:
                torch.save(model.state_dict(), save_path + '-{}'.format(n_saves))
                n_saves += 1
