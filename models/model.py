from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['FirstStage']

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)


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
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.block_1(input)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        fc_1 = self.fc_1(x)
        x = self.fc_2(fc_1)
        return x, fc_1


if __name__ == "__main__":
    from thop import profile

    model = FirstStageNet()
    input = torch.randn(1, 3, 112, 112)

    total_ops, total_params = profile(model, (input,))
    print("{:.4f} MACs(G)\t{:.4f} Params(M)".format(
        total_ops / (1000 ** 3), total_params / (1000 ** 2)))
