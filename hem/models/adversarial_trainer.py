from hem.models.trainer import Trainer
import torch
import torch.nn as nn


class AdversarialTrainer(Trainer):
    def _build_optimizer_and_scheduler(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module
        optim1 = torch.optim.Adam(model.p1_params(), self._config['lr1'])
        optim2 = torch.optim.Adam(model.p2_params(), self._config['lr2'])
        lr_schedule = self._config.get('lr_schedule', None)
        assert not lr_schedule, 'Schedule not implemented yet for adversarial training!'
        return [optim1, optim2], None
    
    def _step_optim(self, loss, step, optimizers):
        l1, l2 = loss
        o1, o2 = optimizers
        
        # calculate player1's gradient and step
        if step % self._1_step_per_2 == 0:
            l1.backward(retain_graph=True)
            o1.step()

        # zero out player1's grad in player2's buffers, then calc player2's grad and step
        if step % self._2_step_per_1 == 0:
            o2.zero_grad()
            l2.backward()
            o2.step()

    def _zero_grad(self, optimizers):
        [o.zero_grad() for o in optimizers]

    def _loss_to_scalar(self, losses):
        l1, l2 = losses
        return l1.item() + l2.item()

    @property
    def _2_step_per_1(self):
        o2_per_1 = self._config.get('o2_per_1', 1)
        if isinstance(o2_per_1, int):
            return o2_per_1
        o2_per_1, start = o2_per_1
        if self.step < start:
            return 1
        return o2_per_1

    @property
    def _1_step_per_2(self):
        o1_per_2 = self._config.get('o1_per_2', 1)
        if isinstance(o1_per_2, int):
            return o1_per_2
        o1_per_2, start = o1_per_2
        if self.step < start:
            return 1
        return o1_per_2
