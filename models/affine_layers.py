from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineParamsLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        last_row = torch.Tensor

    def forward(self, src, dst):
        src = src.view(src.size(0), -1, 2)
        src_mean = torch.mean(src, axis=1, keepdim=True)
        src_demean = src - src_mean

        dst = dst.view(dst.size(0), -1, 2)
        dst_mean = torch.mean(dst, axis=1, keepdim=True)
        dst_demean = dst - dst_mean

        dot_result = torch.sum(src_demean * dst_demean, dim=(1, 2))
        norm_pow_2 = torch.norm(src_demean, dim=(1, 2)) ** 2

        a = dot_result / norm_pow_2
        b = torch.sum(src_demean[:, :, 0] * dst_demean[:, :, 1] - src_demean[:, :, 1] * dst_demean[:, :, 0], dim=1) / norm_pow_2

        sr = torch.stack((a, -b, b, a), dim=1).view(-1, 2, 2)
        t = dst_mean - torch.bmm(src_mean, sr.permute(0, 2, 1))

        sr = F.pad(sr, pad=[0, 0, 0, 1], mode='constant', value=0)
        t = F.pad(t.permute(0, 2, 1), pad=[0, 0, 0, 1], mode='constant', value=1.0)
        return torch.cat((sr, t), dim=2)