from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineParamsLayer(nn.Module):
    def __init__(self):
        super().__init__()

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
        b = torch.sum(src_demean[:, :, 0] * dst_demean[:, :, 1] -
                      src_demean[:, :, 1] * dst_demean[:, :, 0], dim=1) / norm_pow_2

        sr = torch.stack((a, -b, b, a), dim=1).view(-1, 2, 2)
        t = dst_mean - torch.bmm(src_mean, sr.permute(0, 2, 1))

        sr = F.pad(sr, pad=[0, 0, 0, 1], mode='constant', value=0)
        t = F.pad(t.permute(0, 2, 1), pad=[
                  0, 0, 0, 1], mode='constant', value=1.0)
        return torch.cat((sr, t), dim=2)


class AffineLandmarkLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lmk, params):
        lmk = lmk.view(lmk.size(0), -1, 2)
        lmk = F.pad(lmk, pad=[0, 1], mode='constant', value=1.0)
        lmk = torch.bmm(lmk, params.permute(0, 2, 1))
        return lmk[:, :, :2]


class AffineImageLayer(nn.Module):
    def __init__(self, input_w, input_h, output_w, output_h):
        from utils.misc.custom_module import WarpAffineToAffineGrid
        super().__init__()
        self.output_w = output_w
        self.output_h = output_h
        self.cv_2_pytorch = WarpAffineToAffineGrid(
            input_w, input_h, output_w, output_h)

    def forward(self, img, params):
        params = self.cv_2_pytorch(params)
        grid = F.affine_grid(
            params[:, :2, :], [img.size(0), img.size(1), self.output_h, self.output_w], align_corners=False)
        return F.grid_sample(img, grid, align_corners=False)


class GenHeatmapLayer(nn.Module):
    def __init__(self, w, h, sigma, reduce_dim=True):
        super().__init__()
        self.register_buffer('const_pixels', torch.from_numpy(np.mgrid[0:w, 0:h].transpose(
            2, 1, 0).astype(np.float32)))
        self.sigma = sigma
        self.reduce_dim = reduce_dim

    def forward(self, shape: torch.Tensor):
        shape = shape.unsqueeze(2).unsqueeze(2)
        value = self.const_pixels - shape
        value = torch.norm(value, p=2, dim=-1)
        if self.reduce_dim:
            value, _ = torch.min(value, dim=1, keepdim=True)
        value = torch.exp(-(value / self.sigma) ** 2)
        return value


if __name__ == "__main__":
    import cv2
    from utils.misc.vis import draw_points
    from models.model import get_mean_shape_300w

    '''
    src image
    '''
    src_img = cv2.imread('./image/test/221458225_1.jpg',
                         cv2.IMREAD_COLOR).astype(np.float32) / 255
    src_lmk = np.genfromtxt('./image/test/221458225_1.pts',
                            skip_header=3, skip_footer=1) - 1.0
    draw_img = draw_points(src_img, src_lmk)
    cv2.imshow('draw_img', draw_img)

    '''
    test layer
    '''
    resize_size = 512
    src_img_tensor = torch.from_numpy(
        src_img).unsqueeze(0).float().permute(0, 3, 1, 2)
    src_lmk_tensor = torch.from_numpy(src_lmk).unsqueeze(0).float()
    mean_shape = torch.from_numpy(
        get_mean_shape_300w()).unsqueeze(0) * resize_size

    param = AffineParamsLayer()(src_lmk_tensor, mean_shape)
    trans_lmk_tensor = AffineLandmarkLayer()(src_lmk_tensor, param)
    trans_img_tensor = AffineImageLayer(
        src_img.shape[1], src_img.shape[0], resize_size, resize_size)(src_img_tensor, param)
    heatmap_tensor = GenHeatmapLayer(
        resize_size, resize_size, 3.0)(trans_lmk_tensor)

    trans_lmk = trans_lmk_tensor.detach().numpy()[0]
    trans_img = trans_img_tensor.detach().numpy()[0].transpose(1, 2, 0)
    heatmap = heatmap_tensor.detach().numpy()[0].transpose(1, 2, 0)

    trans_img = draw_points(trans_img, trans_lmk, color=(1.0, 0.0, 0.0))
    heatmap_lmk = draw_points(heatmap, trans_lmk, color=(0.0))

    cv2.imshow('trans_img', trans_img)
    cv2.imshow('heatmap_lmk', heatmap_lmk)
    cv2.waitKey(-1)
