from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from utils.misc.custom_module import ELoss


class LandmarkWrapper(nn.Module):
    def __init__(self,
                 model,
                 hparams,
                 train_dataset,
                 eval_dataset):
        super(LandmarkWrapper, self).__init__()
        self.model = model
        self.hparams = hparams
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eloss = ELoss()

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch['image'], batch['landmark']
        x = x.permute(0, 3, 1, 2).float() / 255
        y = y.float() / x.size(-1)
        y_hat = self.forward(x).reshape(x.size(0), -1, 2)
        loss = self.eloss(y_hat, y)

        right_bottom, _ = y.max(dim=1, keepdim=True)
        left_top, _ = y.min(dim=1, keepdim=True)
        diag_dis = torch.norm(right_bottom - left_top,
                              p=2, dim=2, keepdim=True)
        score = torch.norm(y_hat - y, p=2, dim=2,
                           keepdim=True) / diag_dis
        score = score.squeeze().mean(-1) * 100
        return {'loss': loss, 'score': score}

    def _epoch(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        score = torch.cat([x['score'] for x in outputs], dim=0).mean()
        return {'loss': loss, 'score': score}

    def training_step(self, batch):
        self.model.train()
        return self._step(batch)

    def training_epoch(self, outputs):
        return self._epoch(outputs)

    def validation_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self._step(batch)

    def validation_epoch(self, outputs):
        return self._epoch(outputs)

    def configure_optimizers(self):
        self.model.train()

        decay = []
        decay_s = []
        no_decay = []
        for m in self.model.modules():
            if not m.training:
                continue
            elif isinstance(m, nn.Conv2d):
                if (m.in_channels == m.out_channels) and (m.in_channels == m.groups):
                    # depth-wise
                    decay_s.append(m.weight)
                else:
                    decay.append(m.weight)
                if m.bias is not None:
                    no_decay.append(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                no_decay.append(m.weight)
                no_decay.append(m.bias)
            elif isinstance(m, nn.Linear):
                decay.append(m.weight)
                no_decay.append(m.bias)
        params = [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": decay_s, "weight_decay": self.hparams.weight_decay * 0.1},
        ]

        optimizer = torch.optim.SGD(
            params,
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum)

        scheduler = OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            total_steps=self.hparams.num_steps,
            pct_start=0.1)
        return optimizer, scheduler

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          shuffle=True, pin_memory=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=8)
