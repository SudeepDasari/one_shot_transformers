from hem.util import parse_basic_config
import torch
from torch.utils.data import DataLoader
import argparse
from hem.datasets import get_dataset
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import torch.nn as nn
import os
import shutil
import copy
import yaml
from hem.models.lr_scheduler import build_scheduler
import torchvision


class Trainer:
    def __init__(self, save_name='train', description="Default model trainer", drop_last=False):
        now = datetime.datetime.now()
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('experiment_file', type=str, help='path to YAML experiment config file')
        parser.add_argument('--save_path', type=str, default='', help='path to place model save file in during training (overwrites config)')
        parser.add_argument('--device', type=int, default=None, nargs='+', help='target device (uses all if not specified)')
        args = parser.parse_args()
        self._config = parse_basic_config(args.experiment_file)
        save_config = copy.deepcopy(self._config)
        if args.save_path:
            self._config['save_path'] = args.save_path

        # initialize device
        def_device = 0 if args.device is None else args.device[0]
        self._device = torch.device("cuda:{}".format(def_device))
        self._device_list = args.device

        # parse dataset class and create train/val loaders
        dataset_class = get_dataset(self._config['dataset'].pop('type'))
        dataset = dataset_class(**self._config['dataset'], mode='train')
        val_dataset = dataset_class(**self._config['dataset'], mode='val')
        self._train_loader = DataLoader(dataset, batch_size=self._config['batch_size'], shuffle=True, num_workers=self._config.get('loader_workers', cpu_count()), drop_last=drop_last, worker_init_fn=lambda w: np.random.seed(np.random.randint(2 ** 29) + w))
        self._val_loader = DataLoader(val_dataset, batch_size=self._config['batch_size'], shuffle=True, num_workers=min(1, self._config.get('loader_workers', cpu_count())), drop_last=True, worker_init_fn=lambda w: np.random.seed(np.random.randint(2 ** 29) + w))

        # set of file saving
        save_dir = os.path.join(self._config.get('save_path', './'), '{}_ckpt-{}-{}_{}-{}-{}'.format(save_name, now.hour, now.minute, now.day, now.month, now.year))
        save_dir = os.path.expanduser(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False)
        self._writer = SummaryWriter(log_dir=os.path.join(save_dir, 'log'))
        self._save_fname = os.path.join(save_dir, 'model_save')
        self._step = None

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def train(self, model, train_fn, weights_fn=None, val_fn=None, save_fn=None, optim_weights=None):
        # wrap model in DataParallel if needed and transfer to correct device
        if self.device_count > 1:
            model = nn.DataParallel(model, device_ids=self.device_list)
        model = model.to(self._device)
        
        # initializer optimizer and lr scheduler
        optim_weights = optim_weights if optim_weights is not None else model.parameters()
        optimizer, scheduler = self._build_optimizer_and_scheduler(optim_weights)

        # initialize constants:
        epochs = self._config.get('epochs', 1)
        vlm_alpha = self._config.get('vlm_alpha', 0.6)
        log_freq = self._config.get('log_freq', 20)
        self._img_log_freq = img_log_freq = self._config.get('img_log_freq', 500)
        assert img_log_freq % log_freq == 0, "log_freq must divide img_log_freq!"
        save_freq = self._config.get('save_freq', 5000)

        if val_fn is None:
            val_fn = train_fn

        self._step = 0
        train_stats = {'loss': 0}
        val_iter = iter(self._val_loader)
        vl_running_mean = None
        for e in range(epochs):
            for inputs in self._train_loader:
                self._zero_grad(optimizer)
                loss_i, stats_i = train_fn(model, self._device, *inputs)
                self._step_optim(loss_i, self._step, optimizer)
                
                # calculate iter stats
                mod_step = self._step % log_freq
                train_stats['loss'] = (self._loss_to_scalar(loss_i) + mod_step * train_stats['loss']) / (mod_step + 1)
                for k, v in stats_i.items():
                    if isinstance(v, torch.Tensor):
                        assert len(v.shape) >= 4, "assumes 4dim BCHW image tensor!"
                        train_stats[k] = v
                    if k not in train_stats:
                        train_stats[k] = 0
                    train_stats[k] = (v + mod_step * train_stats[k]) / (mod_step + 1)
                
                if mod_step == 0:
                    try:
                        val_inputs = next(val_iter)
                    except StopIteration:
                        val_iter = iter(self._val_loader)
                        val_inputs = next(val_iter)

                    with torch.no_grad():
                        model = model.eval()
                        val_loss, val_stats = val_fn(model, self._device, *val_inputs)
                        model = model.train()
                        val_loss = self._loss_to_scalar(val_loss)

                    # update running mean stat
                    if vl_running_mean is None:
                        vl_running_mean = val_loss
                    vl_running_mean = val_loss * vlm_alpha + vl_running_mean * (1 - vlm_alpha)

                    self._writer.add_scalar('loss/val', val_loss, self._step)
                    for stats_dict, mode in zip([train_stats, val_stats], ['train', 'val']):
                        for k, v in stats_dict.items():
                            if isinstance(v, torch.Tensor) and self.step % img_log_freq == 0:
                                if len(v.shape) == 5:
                                    self._writer.add_video('{}/{}'.format(k, mode), v.cpu(), self._step)
                                else:
                                    v_grid = torchvision.utils.make_grid(v.cpu(), padding=5)
                                    self._writer.add_image('{}/{}'.format(k, mode), v_grid, self._step)
                            elif not isinstance(v, torch.Tensor):
                                self._writer.add_scalar('{}/{}'.format(k, mode), v, self._step)
                    self._writer.file_writer.flush()
                    print('epoch {3}/{4}, step {0}: loss={1:.4f} \t val loss={2:.4f}'.format(self._step, train_stats['loss'], vl_running_mean, e, epochs))
                else:
                    print('step {0}: loss={1:.4f}'.format(self._step, train_stats['loss']), end='\r')
                self._step += 1

                if self._step % save_freq == 0:
                    if save_fn is not None:
                        save_fn(self._save_fname, self._step)
                    else:
                        save_module = model
                        if weights_fn is not None:
                            save_module = weights_fn()
                        elif isinstance(model, nn.DataParallel):
                            save_module = model.module
                        torch.save(save_module, self._save_fname + '-{}.pt'.format(self._step))
                    if self._config.get('save_optim', False):
                        torch.save(optimizer.state_dict(), self._save_fname + '-optim-{}.pt'.format(self._step))
            scheduler.step(val_loss=vl_running_mean)

    @property
    def device_count(self):
        if self._device_list is None:
            return torch.cuda.device_count()
        return len(self._device_list)

    @property
    def device_list(self):
        if self._device_list is None:
            return [i for i in range(torch.cuda.device_count())]
        return copy.deepcopy(self._device_list)

    @property
    def device(self):
        return copy.deepcopy(self._device)

    def _build_optimizer_and_scheduler(self, optim_weights):
        optimizer = torch.optim.Adam(optim_weights, self._config['lr'], weight_decay=self._config.get('weight_decay', 0))
        return optimizer, build_scheduler(optimizer, self._config.get('lr_schedule', {}))

    def _step_optim(self, loss, step, optimizer):
        loss.backward()
        optimizer.step()

    def _zero_grad(self, optimizer):
        optimizer.zero_grad()

    def _loss_to_scalar(self, loss):
        return loss.item()

    @property
    def step(self):
        if self._step is None:
            raise Exception("Optimization has not begun!")
        return self._step

    @property
    def is_img_log_step(self):
        return self._step % self._img_log_freq == 0
