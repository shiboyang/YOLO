# @Time    : 2023/2/24 下午3:56
# @Author  : Boyang
# @Site    : 
# @File    : darknet.py
# @Software: PyCharm

import torch.nn as nn
from typing import List
from collections import OrderedDict
from detectron2.modeling.backbone import BACKBONE_REGISTRY


class DarkNet(nn.Module):

    def __init__(self, layers: List[int]):
        super(DarkNet, self).__init__()
        self.in_channel = 32
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu1 = nn.LeakyReLU(0.1)

        self.dark1 = self._make_layers(layers[0], [32, 64])

        self.dark2 = self._make_layers(layers[1], [64, 128])

        self.dark3 = self._make_layers(layers[2], [128, 256])

        self.dark4 = self._make_layers(layers[3], [256, 512])

        self.dark5 = self._make_layers(layers[4], [512, 102])

        self._out_features = ("d1", "d2", "d3", "d4", "d5")

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        res_dark1 = self.dark1(out)
        res_dark2 = self.dark2(res_dark1)
        res_dark3 = self.dark3(res_dark2)
        res_dark4 = self.dark4(res_dark3)
        res_dark5 = self.dark5(res_dark4)

        return {k: v for v, k in zip([res_dark1, res_dark2, res_dark3, res_dark4, res_dark5], self._out_features)}

    def _make_dark(self, blocks: int, out_channels: List[int]):
        """
        DownSample(downsample_conv -> downsample_bn -> downsample_relu) -> Residual
        :param blocks: int
        :param out_channels: List[int] the list of out_channels, len(out_channels) = 2
        :return:
        """
        layers = []
        layers.append(("DownSample", DownSample(self.in_channel, out_channels[-1])))
        self.in_channel = out_channels[-1]
        for i in range(blocks):
            residual = Residual(self.in_channel, out_channels)
            layers.append((f"Residual{i}", residual))

        return nn.Sequential(OrderedDict(layers))


class DownSample(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channels=out_channel, kernel_size=1, stride=2, padding=1,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class Residual(nn.Module):
    def __init__(self, in_channel: int, out_channels: List[int]):
        """
        Residual layer
        :param in_channel: int
        :param out_channels: lens=2 List[int]
        """
        super(Residual, self).__init__()
        assert in_channel == out_channels[-1]

        self.conv1 = nn.Conv2d(in_channel, out_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0], eps=1e-5, momentum=0.1)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1], eps=1e-5, momentum=0.1)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


@BACKBONE_REGISTRY.register()
def build_darknet53_backbone(cfg):
    backbone = DarkNet([1, 2, 8, 8, 4])
    return backbone
