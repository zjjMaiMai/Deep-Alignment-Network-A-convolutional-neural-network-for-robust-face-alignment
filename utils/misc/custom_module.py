from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ELoss, self).__init__()
        self.reduction = reduction
        assert reduction in ['none', 'mean', 'sum']

    def forward(self, input, target):
        input = input.view(input.size(0), -1, 2)
        target = target.view(target.size(0), -1, 2)
        norm = torch.norm(target - input, 2, dim=-1)

        if self.reduction == 'none':
            return norm
        elif self.reduction == 'mean':
            return norm.mean()
        elif self.reduction == 'sum':
            return norm.sum()
        else:
            raise NotImplementedError()


def random_seed(seed=2019):
    import random
    import numpy as np
    import torch

    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def extract_glimpse(
        input,
        size,
        offsets,
        centered=False,
        normalized=False,
        mode='bilinear',
        padding_mode='zeros'):
    '''
    offsets: A 2-D tensor of shape [N, X, 2] containing the y, x locations of the center of each window.
    mode: Interpolation mode to calculate output values 'bilinear' | 'nearest'. Default: 'bilinear'
    padding_mode: padding mode for outside grid values 'zeros' | 'border' | 'reflection'. Default: 'zeros'

    The argument normalized and centered controls how the windows are built:

    If the coordinates are normalized but not centered, 0.0 and 1.0 correspond to the minimum and maximum of each height and width dimension.
    If the coordinates are both normalized and centered, they range from -1.0 to 1.0.
        The coordinates (-1.0, -1.0) correspond to the upper left corner, the lower right corner is located at (1.0, 1.0) and the center is at (0, 0).
    If the coordinates are not normalized they are interpreted as numbers of pixels.
    '''
    assert input.dtype == offsets.dtype

    n, x = offsets.size(0), offsets.size(1)
    p_w, p_h = size[0], size[1]
    h, w = input.size(2), input.size(3)

    input_shape = offsets.new_tensor([w, h])
    if normalized:
        offsets = offsets * input_shape

    if centered:
        offsets = (offsets + input_shape) / 2

    offsets = offsets - offsets.new_tensor([p_w, p_h]) / 2
    offsets = offsets.unsqueeze(2).unsqueeze(3)

    g_h = torch.arange(0, p_h, device=offsets.device)
    g_w = torch.arange(0, p_w, device=offsets.device)
    g_h, g_w = torch.meshgrid([g_h, g_w])

    mesh = torch.stack((g_w, g_h), dim=-1).unsqueeze(0).unsqueeze(0)
    mesh = mesh.repeat(n, x, 1, 1, 1)
    mesh = mesh.to(offsets.dtype) + offsets

    mesh = mesh + 0.5
    mesh = mesh / input_shape * 2 - 1
    mesh = mesh.view(n, x * p_h, p_w, 2)

    output = F.grid_sample(input, mesh, mode=mode, padding_mode=padding_mode,
                           align_corners=False)
    output = output.view(n, -1, x, p_h, p_w)
    output = output.permute(0, 2, 1, 3, 4)
    return output


class WarpAffineToAffineGrid(nn.Module):
    def __init__(self, w, h):
        super().__init__()
        left = torch.Tensor([
            [2.0 / w, 0.0, -1.0],
            [0.0, 2.0 / h, -1.0],
            [0.0, 0.0, 1.0]]).float()
        right = torch.inverse(left)
        self.register_buffer('left', left)
        self.register_buffer('right', right)

    def forward(self, x):
        return torch.matmul(self.left, torch.matmul(torch.inverse(x), self.right))
