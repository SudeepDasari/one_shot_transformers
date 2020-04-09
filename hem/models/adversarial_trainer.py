from hem.models.trainer import Trainer
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


class AdversarialTrainer(Trainer):

    def _build_optimizer_and_scheduler(self, model):
        optim1 = torch.optim.Adam(model.p1_params(), self._config['lr1'])
        optim2 = torch.optim.Adam(model.p2_params(), self._config['lr2'])
        lr_schedule = self._config.get('lr_schedule', None)
        assert not lr_schedule, 'Schedule not implemented yet for adversarial training!'
        return [optim1, optim2], None
    
    def _step_optim(self, loss, step, optimizer):
        l1, l2 = loss
        o1, o2 = optimizers
        
        # calculate player1's gradient and step
        l1.backward()
        o1.step()

        # zero out player1's grad in player2's buffers, then calc player2's grad and step
        if step % self._config.get('o2_per_1', 1) == 0:
            o2.zero_grad()
            l2.backward()
            o2.step()

    def _zero_grad(self, optimizers):
        [o.zero_grad() for o in optimizers]

    def _loss_to_scalar(self, loss):
        return loss.item()
