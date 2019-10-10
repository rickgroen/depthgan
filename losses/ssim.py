import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim_gauss(img1, img2, window, window_size, channel):
    mu_x = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu_y = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    SSIM_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


# SSIM using the standard gaussian kernel. Taken from
# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
def ssim_gauss(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim_gauss(img1, img2, window, window_size, channel)


# SSIM from Godard's code
def ssim_godard(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.functional.avg_pool2d(x, 3, 1, padding=0)
    mu_y = nn.functional.avg_pool2d(y, 3, 1, padding=0)

    sigma_x = nn.functional.avg_pool2d(x ** 2, 3, 1, padding=0) - mu_x ** 2
    sigma_y = nn.functional.avg_pool2d(y ** 2, 3, 1, padding=0) - mu_y ** 2

    sigma_xy = nn.functional.avg_pool2d(x * y, 3, 1, padding=0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)
