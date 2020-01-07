from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dan_layers import *


class MobileBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, expand=6):
        super().__init__()
        assert stride in [1, 2]

        mid_c = int(in_c * expand)
        self.use_res_connect = (stride == 1 and in_c == out_c)

        self.block = []
        if expand != 1:
            self.block += [
                nn.Conv2d(in_c, mid_c, 1, bias=False),
                nn.BatchNorm2d(mid_c),
                nn.ReLU(True),
            ]
        self.block += [
            nn.Conv2d(mid_c, mid_c, 3, stride=stride,
                      padding=1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(True),
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        x = self.block(input)
        if self.use_res_connect:
            x = x + input
        return x


def make_block(c_in, c_out, n, s, e=1):
    blocks = []
    last_channel = c_in
    for count in range(n):
        stride = s if count == 0 else 1
        blocks.append(MobileBlock(last_channel, c_out, stride, e))
        last_channel = c_out
    return blocks


class FirstStageNetMBV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            *make_block(32, 16, 1, 1, 6),
            *make_block(16, 24, 2, 2, 6),
            *make_block(24, 32, 3, 2, 6),
            *make_block(32, 64, 4, 2, 6),
            *make_block(64, 96, 3, 2, 6),
            nn.Conv2d(96, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 7, stride=7, groups=256),
            nn.Conv2d(256, 136, 1)
        )

    def forward(self, input):
        x = self.feature(input)
        return x.view(x.size(0), -1, 2)


class SecondStageNetMBV2(nn.Module):
    def __init__(self, input_size, mean_shape):
        super().__init__()
        self.input_size = input_size
        self.register_buffer(
            'mean_shape', torch.from_numpy(mean_shape).float().unsqueeze(0))
        self.affine_params_layer = AffineParamsLayer()
        self.affine_image_layer = AffineImageLayer(
            input_size, input_size, input_size, input_size)
        self.affine_landmark_layer = AffineLandmarkLayer()
        self.gen_heatmap_layer = GenHeatmapLayer(
            input_size, input_size, input_size / 32)

        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            *make_block(32, 16, 1, 1, 6),
            *make_block(16, 24, 2, 2, 6),
            *make_block(24, 32, 3, 2, 6),
            *make_block(32, 64, 4, 2, 6),
            *make_block(64, 96, 3, 2, 6),
            nn.Conv2d(96, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 7, stride=7, groups=256),
            nn.Conv2d(256, 136, 1)
        )

    def forward(self, input, lmk):
        affine_params = self.affine_params_layer(
            lmk * self.input_size, self.mean_shape)
        inv_affine_params = torch.inverse(affine_params)
        lmk = self.affine_landmark_layer(lmk * self.input_size, affine_params)
        input = self.affine_image_layer(input, affine_params)
        heatmap = self.gen_heatmap_layer(lmk)

        x = torch.cat((input, heatmap), dim=1)
        x = self.feature(x)
        lmk = x.view(x.size(0), -1, 2) * self.input_size + lmk
        lmk = self.affine_landmark_layer(
            lmk, inv_affine_params) / self.input_size
        return lmk


class DAN_MBV2(nn.Module):
    def __init__(self, input_size, mean_shape, stage):
        super().__init__()
        self.first_stage = FirstStageNetMBV2()
        self.second_stage = SecondStageNetMBV2(input_size, mean_shape)
        self.stage = stage
        self.init_weight()

    def train(self, mode=True):
        self.training = mode
        if self.stage == 0:
            self.first_stage.train(mode)
            self.second_stage.train(False)
        elif self.stage == 1:
            self.first_stage.train(False)
            self.second_stage.train(mode)
        else:
            raise NotImplementedError
        return self

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        if self.stage == 0:
            lmk = self.first_stage(input)
        elif self.stage == 1:
            with torch.no_grad():
                lmk = self.first_stage(input)
            lmk = self.second_stage(input, lmk)
        else:
            raise NotImplementedError
        return lmk


if __name__ == "__main__":
    from thop import profile

    model = DAN_MBV2(224, np.random.rand(68, 2), 1)
    input = torch.randn(1, 3, 224, 224)

    total_ops, total_params = profile(model, (input,))
    print("{:.4f} MACs(G)\t{:.4f} Params(M)".format(
        total_ops / (1000 ** 3), total_params / (1000 ** 2)))
