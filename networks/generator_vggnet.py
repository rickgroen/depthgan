from networks.blocks import *


class VggNetMD(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=2, normalize=None):
        super(VggNetMD, self).__init__()
        # encoder
        self.conv1 = ConvBlock(num_in_layers, 32, 7, normalize=normalize)       # H/2
        self.conv2 = ConvBlock(32, 64, 5, normalize=normalize)                  # H/4
        self.conv3 = ConvBlock(64, 128, 3, normalize=normalize)                 # H/8
        self.conv4 = ConvBlock(128, 256, 3, normalize=normalize)                # H/16
        self.conv5 = ConvBlock(256, 512, 3, normalize=normalize)                # H/32
        self.conv6 = ConvBlock(512, 512, 3, normalize=normalize)                # H/64
        self.conv7 = ConvBlock(512, 512, 3, normalize=normalize)                # H/128

        # decoder
        self.upconv7 = Upconv(512, 512, 3, 2, normalize=normalize)
        self.iconv7 = Conv(512 + 512, 512, 3, 1, normalize=normalize)

        self.upconv6 = Upconv(512, 512, 3, 2, normalize=normalize)
        self.iconv6 = Conv(512 + 512, 512, 3, 1, normalize=normalize)

        self.upconv5 = Upconv(512, 256, 3, 2, normalize=normalize)
        self.iconv5 = Conv(256 + 256, 256, 3, 1, normalize=normalize)

        self.upconv4 = Upconv(256, 128, 3, 2, normalize=normalize)
        self.iconv4 = Conv(128 + 128, 128, 3, 1, normalize=normalize)
        self.disp4_layer = GetDisp(128, num_out_layers=num_out_layers)

        self.upconv3 = Upconv(128, 64, 3, 2, normalize=normalize)
        self.iconv3 = Conv(64 + 64 + num_out_layers, 64, 3, 1, normalize=normalize)
        self.disp3_layer = GetDisp(64, num_out_layers=num_out_layers)

        self.upconv2 = Upconv(64, 32, 3, 2, normalize=normalize)
        self.iconv2 = Conv(32 + 32 + num_out_layers, 32, 3, 1, normalize=normalize)
        self.disp2_layer = GetDisp(32, num_out_layers=num_out_layers)

        self.upconv1 = Upconv(32, 16, 3, 2, normalize=normalize)
        self.iconv1 = Conv(16 + num_out_layers, 16, 3, 1, normalize=normalize)
        self.disp1_layer = GetDisp(16, num_out_layers=num_out_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

        # skips
        skip1 = x1
        skip2 = x2
        skip3 = x3
        skip4 = x4
        skip5 = x5
        skip6 = x6

        # decoder
        upconv7 = self.upconv7(x7)
        concat7 = torch.cat((upconv7, skip6), 1)
        iconv7 = self.iconv7(concat7)

        upconv6 = self.upconv6(iconv7)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)

        return self.disp1, self.disp2, self.disp3, self.disp4


class VggNetSuper(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=2, normalize=None):
        super(VggNetSuper, self).__init__()
        # encoder
        self.conv1 = ConvBlock(num_in_layers, 32, 7, normalize=normalize)       # H/2
        self.conv2 = ConvBlock(32, 64, 5, normalize=normalize)                  # H/4
        self.conv3 = ConvBlock(64, 128, 3, normalize=normalize)                 # H/8
        self.conv4 = ConvBlock(128, 256, 3, normalize=normalize)                # H/16
        self.conv5 = ConvBlock(256, 512, 3, normalize=normalize)                # H/32
        self.conv6 = ConvBlock(512, 512, 3, normalize=normalize)                # H/64
        self.conv7 = ConvBlock(512, 512, 3, normalize=normalize)                # H/128

        # decoder
        self.upconv7 = Upconv(512, 512, 3, 2, normalize=normalize)
        self.iconv7 = Conv(512 + 512, 512, 3, 1, normalize=normalize)

        self.upconv6 = Upconv(512, 512, 3, 2, normalize=normalize)
        self.iconv6 = Conv(512 + 512, 512, 3, 1, normalize=normalize)

        self.upconv5 = Upconv(512, 256, 3, 2, normalize=normalize)
        self.iconv5 = Conv(256 + 256, 256, 3, 1, normalize=normalize)

        self.upconv4 = Upconv(256, 128, 3, 2, normalize=normalize)
        self.iconv4 = Conv(128 + 128, 128, 3, 1, normalize=normalize)

        self.upconv3 = Upconv(128, 64, 3, 2, normalize=normalize)
        self.iconv3 = Conv(64 + 64, 64, 3, 1, normalize=normalize)

        self.upconv2 = Upconv(64, 32, 3, 2, normalize=normalize)
        self.iconv2 = Conv(32 + 32, 32, 3, 1, normalize=normalize)

        self.upconv1 = Upconv(32, 16, 3, 2, normalize=normalize)
        self.iconv1 = Conv(16, 16, 3, 1, normalize=normalize)
        self.disp1_layer = GetDisp(16, num_out_layers=num_out_layers)

        self.upconv0 = Upconv(16, 8, 3, 2, normalize=normalize)
        self.iconv0 = Conv(8 + num_out_layers, 8, 3, 1, normalize=normalize)
        self.disp0_layer = GetDisp(8, num_out_layers=num_out_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

        # skips
        skip1 = x1
        skip2 = x2
        skip3 = x3
        skip4 = x4
        skip5 = x5
        skip6 = x6

        # decoder
        upconv7 = self.upconv7(x7)
        concat7 = torch.cat((upconv7, skip6), 1)
        iconv7 = self.iconv7(concat7)

        upconv6 = self.upconv6(iconv7)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2), 1)
        iconv3 = self.iconv3(concat3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1), 1)
        iconv2 = self.iconv2(concat2)

        upconv1 = self.upconv1(iconv2)
        iconv1 = self.iconv1(upconv1)
        self.disp1 = self.disp1_layer(iconv1)
        self.udisp1 = nn.functional.interpolate(self.disp1, scale_factor=2, mode='bilinear', align_corners=True)

        upconv0 = self.upconv0(iconv1)
        concat0 = torch.cat((upconv0, self.udisp1), 1)
        iconv0 = self.iconv0(concat0)
        self.disp0 = self.disp0_layer(iconv0)

        # Unlike the other generators, return the smallest disparity first,
        # so we can use the 256 x 512 disparities during test times without problems.
        return self.disp1, self.disp0
