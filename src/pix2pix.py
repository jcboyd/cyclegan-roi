import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, ReLU, LeakyReLU, BatchNorm2d, Tanh
from torch.nn import MaxPool2d, UpsamplingNearest2d, ZeroPad2d, Sigmoid
from torchvision.ops import RoIAlign, RoIPool


class Downward(Module):

    def __init__(self, in_ch, out_ch, normalise=True):

        super(Downward, self).__init__()

        self.pool = Sequential(
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(in_ch, out_ch, kernel_size=4, stride=2),
            LeakyReLU(negative_slope=0.2, inplace=True))

        if normalise:
            self.pool.add_module('batch_norm', BatchNorm2d(out_ch, momentum=0.8))

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
            BatchNorm2d(out_ch, momentum=0.8),
            LeakyReLU(inplace=True),)

    def forward(self, x1, x2):

        x = self.depool(x1)
        x = torch.cat((x, x2), dim=1)

        return x


class Generator(Module):

    def __init__(self, nb_classes, nb_channels, base_filters=32):

        super(Generator, self).__init__()

        self.down1 = Downward(nb_classes, base_filters, normalise=False)
        self.down2 = Downward(base_filters, 2 * base_filters)
        self.down3 = Downward(2 * base_filters, 4 * base_filters)
        self.down4 = Downward(4 * base_filters, 8 * base_filters)

        self.up1 = Upward(8 * base_filters, 4 * base_filters)
        self.up2 = Upward(8 * base_filters, 2 * base_filters)
        self.up3 = Upward(4 * base_filters, base_filters)

        self.out_conv = Sequential(
            UpsamplingNearest2d(scale_factor=2),
            ZeroPad2d((1, 1, 1, 1)),
            Conv2d(2 * base_filters, nb_channels, kernel_size=3, stride=1),
            Tanh())

    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.out_conv(x)

        return x


class DiscriminatorROI(Module):

    def __init__(self, nb_classes, nb_channels, base_filters):

        super(DiscriminatorROI, self).__init__()

        self.conv_layers = Sequential(
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(nb_classes + nb_channels, base_filters, kernel_size=4, stride=2, bias=False),
            LeakyReLU(0.2, inplace=True),
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(base_filters, 2 * base_filters, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(2 * base_filters, momentum=0.8),
            LeakyReLU(0.2, inplace=True),
            ZeroPad2d((1, 2, 1, 2)),
            Conv2d(2 * base_filters, 4 * base_filters, kernel_size=4, stride=2, bias=False),
            BatchNorm2d(4 * base_filters, momentum=0.8),
            LeakyReLU(0.2, inplace=True))

        self.roi_pool = RoIAlign(output_size=(4, 4), spatial_scale=0.125, sampling_ratio=-1)

        self.classifier = Sequential(
            Conv2d(4 * base_filters, 1, kernel_size=4, padding=0, bias=False), Sigmoid())

    def forward(self, inputs, condition, bboxes):

        #bbox_batch = bboxes[:, :-1]

        x = torch.cat([inputs, condition], axis=1)
        x = self.conv_layers(x)

        pool = self.roi_pool(x, bboxes)
        outputs = self.classifier(pool)

        return outputs.squeeze()


class PatchGAN(Module):

    def __init__(self, nb_classes, nb_channels, base_filters=16):

        super(PatchGAN, self).__init__()

        self.down1 = Downward(nb_classes + nb_channels, base_filters, normalise=False)
        self.down2 = Downward(base_filters, 2 * base_filters)
        self.down3 = Downward(2 * base_filters, 4 * base_filters)

        self.padding = ZeroPad2d((1, 2, 1, 2))
        self.validity = Sequential(
            Conv2d(4 * base_filters, 1, kernel_size=4, stride=1), Sigmoid())

    def forward(self, x, y):

        x = torch.cat([x, y], axis=1)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.padding(x)
        x = self.validity(x)

        return x
