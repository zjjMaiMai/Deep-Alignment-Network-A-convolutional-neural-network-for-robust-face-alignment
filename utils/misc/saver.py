from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import torch
import pathlib


def save_checkpoint(path, model, epoch, optimizer=None, scheduler=None, max_keep=5, verbose=True):
    '''
    Fix me! 增加任意参数
    '''
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    file_path = "{:.0f}.pth".format(time.time())
    file_path = os.path.join(path, file_path)
    if verbose:
        print("Save to {}".format(file_path))
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }, file_path
    )

    if max_keep:
        file_path = os.path.join(path, "*.pth")
        file_list = glob.glob(file_path)
        file_list.sort()

        if len(file_list) <= max_keep:
            return

        for path in file_list[:-max_keep]:
            os.remove(path)
    return


def load_checkpoint(path, model, optimizer=None, scheduler=None, verbose=True):
    file_path = os.path.join(path, "*.pth")
    file_list = glob.glob(file_path)
    checkpoint = None
    if file_list:
        file_list.sort()
        checkpoint = torch.load(file_list[-1], map_location='cpu')
        if verbose:
            print("Load from {}".format(file_list[-1]))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint
