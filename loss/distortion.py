import torch
import torch.nn as nn
from lpips.lpips import LPIPS
from pytorch_msssim import ms_ssim


class Distortion(torch.nn.Module):
    def __init__(self, distortion_type):
        super(Distortion, self).__init__()
        self.distortion_type = distortion_type
        if distortion_type == 'MSE':
            self.metric = nn.MSELoss()
        elif distortion_type == 'MS-SSIM':
            self.metric = ms_ssim
        elif distortion_type == 'LPIPS':
            self.metric = LPIPS()
        else:
            print("Unknown distortion type!")
            raise ValueError

    def forward(self, X, Y):
        if self.distortion_type == 'MS-SSIM':
            return 1 - self.metric(X, Y, data_range=1)
        elif self.distortion_type == 'LPIPS':
            return self.metric(X, Y, normalize=True)
        else:
            return self.metric(X, Y)
