from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.affine_layers import *

__all__ = ['FirstStage']
INPUT_SIZE = 112


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)


class ReshapeFixedSize(nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h

    def forward(self, input):
        return input.view(input.size(0), -1, self.h, self.w)


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return F.interpolate(input, scale_factor=self.scale_factor)


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


class FirstStageNet(nn.Module):
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
        return x, fc_1


class SecondStageNet(nn.Module):
    def __init__(self, mean_shape):
        super().__init__()
        self.register_buffer('mean_shape', mean_shape * INPUT_SIZE)
        self.affine_params_layer = AffineParamsLayer()
        self.affine_image_layer = AffineImageLayer(
            INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
        self.affine_landmark_layer = AffineLandmarkLayer()
        self.gen_heatmap_layer = GenHeatmapLayer(INPUT_SIZE, INPUT_SIZE, 2.0)
        self.fc_transform = nn.Sequential(
            nn.Conv2d(256, (INPUT_SIZE // 2) ** 2, 1, bias=False),
            nn.BatchNorm2d((INPUT_SIZE // 2) ** 2),
            nn.ReLU(True),
            ReshapeFixedSize(INPUT_SIZE // 2, INPUT_SIZE // 2),
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
            lmk * INPUT_SIZE, self.mean_shape)
        inv_affine_params = torch.inverse(affine_params)
        lmk = self.affine_landmark_layer(lmk * INPUT_SIZE, affine_params)
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

        lmk = x * INPUT_SIZE + lmk
        lmk = self.affine_landmark_layer(lmk, inv_affine_params) / INPUT_SIZE
        return lmk


class DAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_stage = FirstStageNet()
        self.second_stage = SecondStageNet(
            torch.from_numpy(get_mean_shape_300w()).unsqueeze(0))
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, second_forward=True):
        lmk_s1, fc_feature = self.first_stage(input)
        if second_forward:
            lmk_s2 = self.second_stage(input, lmk_s1, fc_feature)
        else:
            lmk_s2 = lmk_s1
        return lmk_s1, lmk_s2


def get_mean_shape_300w():
    return np.array([
        [0., 0.19162136],
        [0.00342039, 0.32351251],
        [0.0180597, 0.45481963],
        [0.04585256, 0.5833164],
        [0.09756264, 0.70239563],
        [0.17664164, 0.80549044],
        [0.27326557, 0.89158001],
        [0.37968491, 0.96054857],
        [0.5, 0.98090376],
        [0.62031509, 0.96054857],
        [0.72673443, 0.89158001],
        [0.82335836, 0.80549044],
        [0.90243736, 0.70239563],
        [0.95414744, 0.5833164],
        [0.9819403, 0.45481963],
        [0.99657961, 0.32351251],
        [1., 0.19162136],
        [0.09431708, 0.09303366],
        [0.156603, 0.03609577],
        [0.24499027, 0.01909624],
        [0.33640482, 0.03208252],
        [0.4217254, 0.06810575],
        [0.5782746, 0.06810575],
        [0.66359518, 0.03208252],
        [0.75500973, 0.01909624],
        [0.843397, 0.03609577],
        [0.90568292, 0.09303366],
        [0.5, 0.17077372],
        [0.5, 0.25582131],
        [0.5, 0.34018057],
        [0.5, 0.42711253],
        [0.39750296, 0.48606286],
        [0.44679426, 0.50396655],
        [0.5, 0.51947823],
        [0.55320574, 0.50396655],
        [0.60249704, 0.48606286],
        [0.19547876, 0.18200029],
        [0.2495543, 0.15079845],
        [0.31450698, 0.151619],
        [0.37155765, 0.19479588],
        [0.31041216, 0.20588273],
        [0.2457658, 0.20499598],
        [0.62844235, 0.19479588],
        [0.68549302, 0.151619],
        [0.7504457, 0.15079845],
        [0.80452124, 0.18200029],
        [0.7542342, 0.20499598],
        [0.68958784, 0.20588273],
        [0.30589462, 0.64454894],
        [0.37753891, 0.6160173],
        [0.44988263, 0.60094314],
        [0.5, 0.61354581],
        [0.55011737, 0.60094314],
        [0.62246109, 0.6160173],
        [0.69410538, 0.64454894],
        [0.62477271, 0.71392546],
        [0.55454877, 0.74405802],
        [0.5, 0.74971382],
        [0.44545123, 0.74405802],
        [0.37522729, 0.71392546],
        [0.33558146, 0.6482952],
        [0.44915626, 0.64312397],
        [0.5, 0.64850295],
        [0.55084374, 0.64312397],
        [0.66441854, 0.6482952],
        [0.55181764, 0.67991149],
        [0.5, 0.68607805],
        [0.44818236, 0.67991149]], dtype=np.float32)


if __name__ == "__main__":
    from thop import profile

    model = DAN()
    input = torch.randn(1, 3, 112, 112)

    total_ops, total_params = profile(model, (input,))
    print("{:.4f} MACs(G)\t{:.4f} Params(M)".format(
        total_ops / (1000 ** 3), total_params / (1000 ** 2)))
