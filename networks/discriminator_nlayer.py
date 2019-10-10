import torch.nn as nn
import functools


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 3
        stride = 2
        pad_width = 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=stride, padding=pad_width),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers - 2):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if norm_layer:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kernel_size, stride=stride, padding=pad_width, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kernel_size, stride=stride, padding=pad_width, bias=use_bias),
                    nn.LeakyReLU(0.2, True)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        if norm_layer:
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kernel_size, stride=1, padding=pad_width, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kernel_size, stride=1, padding=pad_width, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=pad_width)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
