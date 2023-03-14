# @Time    : 2023/2/24 下午3:56
# @Author  : Boyang
# @Site    : 
# @File    : darknet.py
# @Software: PyCharm
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from collections import OrderedDict
import fvcore.nn.weight_init as weight_init

from detectron2.layers import get_norm, Conv2d
from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
from detectron2.layers.blocks import CNNBlockBase


# todo Implement make_stage() function.
class DarkNet(Backbone):

    def __init__(
            self,
            stem: BasicStem,
            layers: List[int],
            norm: str
    ):
        super(DarkNet, self).__init__()
        self.stem = stem
        self.in_channel = self.stem.out_channels

        self.dark1 = self._make_dark_block(layers[0], [32, 64], norm)

        self.dark2 = self._make_dark_block(layers[1], [64, 128], norm)

        self.dark3 = self._make_dark_block(layers[2], [128, 256], norm)

        self.dark4 = self._make_dark_block(layers[3], [256, 512], norm)

        self.dark5 = self._make_dark_block(layers[4], [512, 1024], norm)

        self._out_features = ("res1", "res2", "res3", "res4", "res5")

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": self.stem.stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}
        for idx, blocks in enumerate([self.dark1, self.dark2, self.dark3, self.dark4, self.dark5]):
            name = self._out_features[idx]
            self._out_feature_strides[name] = current_stride = current_stride * np.prod([k.stride for k in blocks])
            self._out_feature_channels[name] = blocks[-1].out_channels

    def forward(self, x):
        out = self.stem(x)
        res_dark1 = self.dark1(out)
        res_dark2 = self.dark2(res_dark1)
        res_dark3 = self.dark3(res_dark2)
        res_dark4 = self.dark4(res_dark3)
        res_dark5 = self.dark5(res_dark4)

        return {k: v for v, k in zip([res_dark1, res_dark2, res_dark3, res_dark4, res_dark5], self._out_features)}

    def _make_dark_block(self, blocks: int, out_channels: List[int], norm: str):
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
            residual = Residual(self.in_channel, out_channels, norm)
            layers.append((f"Residual{i}", residual))

        return nn.Sequential(OrderedDict(layers))

    def darknet_modules(self):
        for name, conv in self.modules():
            if isinstance(conv, Conv2d):
                yield name, conv


class DownSample(CNNBlockBase):

    def __init__(self, in_channels, out_channels, stride=2, norm="BN"):
        super(DownSample, self).__init__(in_channels, out_channels, stride)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        weight_init.c2_msra_fill(self.conv)

    def forward(self, x):
        out = self.conv(x)
        out = F.leaky_relu(out, 0.1)
        return out


class Residual(CNNBlockBase):
    def __init__(self, in_channels: int, out_channels: List[int], norm: str):
        """
        Residual layer
        一个残差单元包含 conv 1x1 --> conv 3x3 这个模块并不会改变feature_map的大小
        :param in_channels: int
        :param out_channels: lens=2 List[int]
        """
        super(Residual, self).__init__(in_channels, out_channels[-1], 1)
        assert in_channels == out_channels[-1]

        use_bias = norm == ""
        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels[0],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
            norm=get_norm(norm, out_channels[0])
        )
        self.conv2 = Conv2d(
            in_channels=out_channels[0],
            out_channels=out_channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, out_channels[1])
        )
        for layer in [self.conv1, self.conv2]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        residual = x.clone()

        out = self.conv1(x)
        out = F.leaky_relu(out, 0.1)

        out = self.conv2(out)
        out = F.leaky_relu(out, 0.1)

        out += residual
        return out


class BasicStem(CNNBlockBase):
    """
    A standard Darknet53 stem
    """

    def __init__(self, in_channels, out_channels, norm="BN"):
        super(BasicStem, self).__init__(in_channels, out_channels, 1)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x):
        out = self.conv(x)
        out = F.leaky_relu(out, 0.1)
        return out


@BACKBONE_REGISTRY.register()
def build_darknet53_backbone(cfg, input_shape):
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.DARKNET.STEM_OUT_CHANNELS,
        norm=cfg.MODEL.DARKNET.NORM
    )
    backbone = DarkNet(
        stem,
        [1, 2, 8, 8, 4],
        norm=cfg.MODEL.DARKNET.NORM
    )
    return backbone
