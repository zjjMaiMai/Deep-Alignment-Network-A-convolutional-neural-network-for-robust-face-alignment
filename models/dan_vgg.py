from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dan_layers import *


class VGGBlock(nn.Module):
    def __init__(self, in_c, mid_c, reduce_size=True):
        super().__init__()
        self.block = [
            nn.Conv2d(in_c, mid_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(True),
            nn.Conv2d(mid_c, mid_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(True)]
        if reduce_size:
            self.block += [nn.MaxPool2d(2, 2)]
        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


class FirstStageNetVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = VGGBlock(3, 64)
        self.block_2 = VGGBlock(64, 128)
        self.block_3 = VGGBlock(128, 256)
        self.block_4 = VGGBlock(256, 512)
        self.fc_1 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Conv2d(7 * 7 * 512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.fc_2 = nn.Conv2d(256, 136, 1)

    def forward(self, input):
        x = self.block_1(input)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        fc_1 = self.fc_1(x)
        x = self.fc_2(fc_1)
        return x.view(x.size(0), -1, 2), fc_1


class SecondStageNetVGG(nn.Module):
    def __init__(self, input_size, mean_shape):
        super().__init__()
        self.input_size = input_size
        self.register_buffer(
            'mean_shape', torch.from_numpy(mean_shape).float().unsqueeze(0))
        self.affine_params_layer = AffineParamsLayer()
        self.affine_image_layer = AffineImageLayer(
            input_size, input_size, input_size, input_size)
        self.affine_landmark_layer = AffineLandmarkLayer()
        self.gen_heatmap_layer = GenHeatmapLayer(input_size, input_size, 2.0)
        self.fc_transform = nn.Sequential(
            nn.Conv2d(256, (input_size // 2) ** 2, 1, bias=False),
            nn.BatchNorm2d((input_size // 2) ** 2),
            nn.ReLU(True),
            ReshapeFixedSize(input_size // 2, input_size // 2),
            Upsample())

        self.block_1 = VGGBlock(5, 64)
        self.block_2 = VGGBlock(64, 128)
        self.block_3 = VGGBlock(128, 256)
        self.block_4 = VGGBlock(256, 512)
        self.fc_1 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Conv2d(7 * 7 * 512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.fc_2 = nn.Conv2d(256, 136, 1)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, lmk, fc_feature):
        affine_params = self.affine_params_layer(
            lmk * self.input_size, self.mean_shape)
        inv_affine_params = torch.inverse(affine_params)
        lmk = self.affine_landmark_layer(lmk * self.input_size, affine_params)
        input = self.affine_image_layer(input, affine_params)
        heatmap = self.gen_heatmap_layer(lmk)
        fc_feature = self.fc_transform(fc_feature)

        input = torch.cat((input, heatmap, fc_feature), dim=1)
        x = self.block_1(input)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.fc_1(x)
        x = self.fc_2(x)

        lmk = x.view(x.size(0), -1, 2) * self.input_size + lmk
        lmk = self.affine_landmark_layer(
            lmk, inv_affine_params) / self.input_size
        return lmk


class DAN_VGG(nn.Module):
    def __init__(self, input_size, mean_shape, stage):
        super().__init__()
        self.first_stage = FirstStageNetVGG()
        self.second_stage = SecondStageNetVGG(input_size, mean_shape)
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
            lmk, _ = self.first_stage(input)
        elif self.stage == 1:
            with torch.no_grad():
                lmk_s1, fc_feature = self.first_stage(input)
            lmk = self.second_stage(input, lmk_s1, fc_feature)
        else:
            raise NotImplementedError
        return lmk


if __name__ == "__main__":
    from thop import profile

    model = DAN_VGG(112, np.random.rand(68, 2), 1)
    input = torch.randn(1, 3, 112, 112)

    total_ops, total_params = profile(model, (input,))
    print("{:.4f} MACs(G)\t{:.4f} Params(M)".format(
        total_ops / (1000 ** 3), total_params / (1000 ** 2)))
