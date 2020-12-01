import torch

from torch.nn import Module, Sequential
from torch.nn import Conv2d, ReLU, LeakyReLU, BatchNorm2d, InstanceNorm2d, Tanh
from torch.nn import UpsamplingNearest2d, ZeroPad2d, Sigmoid, Linear
from torchvision.ops import RoIAlign, RoIPool


class ConvBlock(Module):

    def __init__(self, in_ch, out_ch, stride=2, normalise=True):

        super(ConvBlock, self).__init__()

        self.pool = Sequential(
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(in_ch, out_ch, kernel_size=4, stride=stride),
            LeakyReLU(negative_slope=0.2, inplace=True))

        if normalise:
            self.pool.add_module('instance_norm', InstanceNorm2d(out_ch))

    def forward(self, x):
        x = self.pool(x)
        return x


class Upward(Module):

    def __init__(self, in_ch, out_ch):

        super(Upward, self).__init__()

        self.depool = Sequential(
            UpsamplingNearest2d(scale_factor=2),
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(in_ch, out_ch, kernel_size=4, stride=1),
            ReLU(inplace=True),
            InstanceNorm2d(out_ch))

    def forward(self, x1, x2):

        x = self.depool(x1)
        x = torch.cat((x, x2), dim=1)

        return x


class Generator(Module):

    def __init__(self, base_filters=32):

        super(Generator, self).__init__()

        self.in_conv = Sequential(ConvBlock(1, base_filters, stride=1),
                                  ConvBlock(base_filters, base_filters, stride=1))

        self.down1 = ConvBlock(base_filters, 2 * base_filters)
        self.down2 = ConvBlock(2 * base_filters, 4 * base_filters)
        self.down3 = ConvBlock(4 * base_filters, 8 * base_filters)
        self.down4 = ConvBlock(8 * base_filters, 8 * base_filters)

        self.up1 = Upward(8 * base_filters, 8 * base_filters)
        self.up2 = Upward(16 * base_filters, 4 * base_filters)
        self.up3 = Upward(8 * base_filters, 2 * base_filters)
        self.up4 = Upward(4 * base_filters, base_filters)

        self.out_conv = Sequential(
            ConvBlock(2 * base_filters, base_filters, stride=1),
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(base_filters, 1, kernel_size=4, stride=1),
            Tanh())

    def forward(self, x):

        x0 = self.in_conv(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        x = self.out_conv(x)

        return x


class PatchGAN(Module):

    def __init__(self, base_filters=64):

        super(PatchGAN, self).__init__()

        self.down1 = ConvBlock(1, base_filters, normalise=False)
        self.down2 = ConvBlock(base_filters, 2 * base_filters)
        self.down3 = ConvBlock(2 * base_filters, 4 * base_filters)
        self.down4 = ConvBlock(4 * base_filters, 8 * base_filters)
        
        self.validity = Sequential(
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(8 * base_filters, 1, kernel_size=4, stride=1))

    def forward(self, x):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.validity(x)

        return x


class DiscriminatorROI(Module):

    def __init__(self, base_filters=16):

        super(DiscriminatorROI, self).__init__()

        self.conv_layers = Sequential(
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(3, base_filters, kernel_size=4, stride=2, bias=False),
            LeakyReLU(0.2, inplace=True),
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(base_filters, 2 * base_filters, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(2 * base_filters, momentum=0.8),
            LeakyReLU(0.2, inplace=True),
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(2 * base_filters, 4 * base_filters, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(4 * base_filters, momentum=0.8),
            LeakyReLU(0.2, inplace=True))

        self.roi_pool = RoIAlign(output_size=(3, 3), spatial_scale=0.125, sampling_ratio=-1)

        self.classifier = Sequential(
            Conv2d(4 * base_filters, 1, kernel_size=3, padding=0, bias=False))

    def forward(self, inputs, condition, bboxes):

        bbox_batch = bboxes[:, :-1]

        x = torch.cat([inputs, condition], axis=1)
        x = self.conv_layers(x)

        pool = self.roi_pool(x, bbox_batch)
        outputs = self.classifier(pool)

        return outputs.squeeze()
