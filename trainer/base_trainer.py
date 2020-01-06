from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import time
import torch
from utils.misc import saver


class BaseTrainer(object):
    def __init__(self,
                 model_dir,
                 fine_tune=False,
                 num_step=1000,
                 log_every_n_step=100):
        super(BaseTrainer, self).__init__()
        self.model_dir = model_dir
        self.fine_tune = fine_tune
        self.num_step = num_step
        self.log_every_n_step = log_every_n_step

    def fit(self, model):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model.to(device)
        optimizer, scheduler = model.configure_optimizers()

        step = 0
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        checkpoint = saver.load_checkpoint(self.model_dir, model)

        if checkpoint and not self.fine_tune:
            step = checkpoint['epoch']
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        def train_data_generator():
            while True:
                for d in model.train_dataloader():
                    for k in d.keys():
                        d[k] = d[k].to(device, non_blocking=True)
                    yield d
        train_data_iter = train_data_generator()
        batch_size = model.train_dataloader().batch_size

        time_now = time.time()
        log_tensor = []
        while step < self.num_step:
            batch_data = next(train_data_iter)
            optimizer.zero_grad()

            ret_dict = model.training_step(batch_data)
            log_tensor.append(ret_dict)
            loss = ret_dict['loss'].mean()

            loss.backward()
            optimizer.step()
            scheduler.step(step)
            step += 1

            if step % self.log_every_n_step == 0:
                img_per_sec = int(self.log_every_n_step *
                                  batch_size / (time.time() - time_now))
                training_log_str = 'Train Step:{} Lr:{:.6f} Images/Sec:{} '.format(
                    step, optimizer.param_groups[0]["lr"], img_per_sec)
                training_scalars = model.training_epoch(log_tensor)
                training_log_str += self.log_out(training_scalars)

                if hasattr(model, 'validation_step') and \
                   hasattr(model, 'validation_epoch') and \
                   hasattr(model, 'val_dataloader'):
                    log_tensor = []
                    for d in model.val_dataloader():
                        for k in d.keys():
                            d[k] = d[k].to(device, non_blocking=True)
                        log_tensor.append(model.validation_step(d))
                    eval_scalars = model.validation_epoch(log_tensor)
                    training_log_str += ' Eval:[{}]'.format(
                        self.log_out(eval_scalars))
                print(training_log_str)
                saver.save_checkpoint(
                    self.model_dir, model, step, optimizer, verbose=False)
                log_tensor = []
                time_now = time.time()

    def log_out(self, ret_dict):
        str_msg = ''
        for k, v in ret_dict.items():
            if len(str_msg) > 0:
                str_msg += ' '
            str_msg += '{}:{:.6f}'.format(k, v)
        return str_msg
