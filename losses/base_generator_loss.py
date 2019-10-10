import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.bilinear_sampler import apply_disparity
from .ssim import ssim_gauss, ssim_godard


class BaseGeneratorLoss(nn.modules.Module):
    def __init__(self, args):
        super(BaseGeneratorLoss, self).__init__()
        self.which_ssim = args.which_ssim
        self.ssim_window_size = args.ssim_window_size

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]

        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img, [nh, nw], mode='bilinear', align_corners=False))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def generate_image_left(self, img, disp):
        return apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return apply_disparity(img, disp)

    def SSIM(self, x, y):
        if self.which_ssim == 'godard':
            return ssim_godard(x, y)
        elif self.which_ssim == 'gauss':
            return ssim_gauss(x, y, window_size=self.ssim_window_size)
        else:
            raise ValueError('{} version not implemented'.format(self.which_ssim))

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]

        return smoothness_x + smoothness_y

    def forward(self, input, target):
        pass
