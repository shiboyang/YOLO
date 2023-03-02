# @Time    : 2023/2/24 下午3:56
# @Author  : Boyang
# @Site    : 
# @File    : darknet.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from collections import OrderedDict

from detectron2.layers import get_norm, Conv2d, ShapeSpec

from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
from detectron2.layers.blocks import CNNBlockBase


class DarkNet(Backbone):

    def __init__(self, stem, layers: List[int]):
        super(DarkNet, self).__init__()
        self.in_channel = 32
        self.stem = stem

        self.dark1 = self._make_dark(layers[0], [32, 64])

        self.dark2 = self._make_dark(layers[1], [64, 128])

        self.dark3 = self._make_dark(layers[2], [128, 256])

        self.dark4 = self._make_dark(layers[3], [256, 512])

        self.dark5 = self._make_dark(layers[4], [512, 1024])

        self._out_features = ("d1", "d2", "d3", "d4", "d5")

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
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out


class Residual(CNNBlockBase):
    def __init__(self, in_channels: int, out_channels: List[int]):
        """
        Residual layer
        :param in_channels: int
        :param out_channels: lens=2 List[int]
        """
        super(Residual, self).__init__(in_channels, out_channels[-1], 1)
        assert in_channels == out_channels[-1]

        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
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


class BasicStem(CNNBlockBase):
    """
    A standard Darknet53 stem
    """

    def __init__(self, in_channels, out_channels, norm="BN"):
        super(BasicStem, self).__init__(in_channels, out_channels, 1)
        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.relu1 = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out


class DarkNetFPN(Backbone):
    def __init__(
            self,
            bottom_up,
            in_features,
            out_channels,
            num_classes,
            num_anchors,
            norm="",
    ):
        super(DarkNetFPN, self).__init__()
        input_shapes = bottom_up.output_shape()
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        self._head_out_channels = (num_classes + 4 + 1) * num_anchors
        lateral_convs = []
        output_convs = []
        head_convs = []
        for idx, in_channels in enumerate(in_channels_per_feature):
            # !!!在d5上lateral_conv的in_channels没有concat的过程，因此不需要调整in_channels
            if idx + 1 < len(in_channels_per_feature):
                in_channels += in_channels // 2  # todo ???
            lateral_conv, out_conv = self._make_lateral_conv_output_conv(in_channels, out_channels[idx], norm)
            head_conv = self._make_head_conv(out_channels[idx], self._head_out_channels, norm)

            # todo weight init

            self.add_module(f"fpn_lateral{idx}", lateral_conv)
            self.add_module(f"fpn_output{idx}", out_conv)
            self.add_module(f"fpn_head{idx}", head_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(out_conv)
            head_convs.append(head_conv)

        # reverser the order of convs
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.head_convs = head_convs[::-1]

        self.in_features = in_features
        self._out_features = ["p3", "p4", "p5"]
        self.bottom_up = bottom_up

        # 用于output_shape支持
        self._out_feature_channels = {k: input_shapes[f].channels for f, k in zip(self.in_features, self._out_features)}
        self._out_feature_strides = {k: input_shapes[f].stride for f, k in zip(self.in_features, self._out_features)}

    @property
    def size_divisibility(self):
        return 32

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        result = []

        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        result.append(self.head_convs[0](prev_features))
        prev_features = self.output_convs[0](prev_features)

        for idx, (lateral_conv, output_conv, head_conv) in enumerate(
                zip(self.lateral_convs, self.output_convs, self.head_convs)
        ):
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]

                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = torch.cat([top_down_features, features], dim=1)
                prev_features = lateral_conv(lateral_features)
                result.insert(0, head_conv(prev_features))

                prev_features = output_conv(prev_features)

        return {name: res for name, res in zip(self._out_features, result)}

    def _make_head_conv(self, in_channels, out_channels, norm):
        layers = []
        layers.append(
            ("head_conv1", Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=get_norm(norm, in_channels * 2)
            ))
        )
        layers.append(
            ("relu1", nn.LeakyReLU(0.1))
        )
        layers.append(
            ("head", Conv2d(
                in_channels=in_channels * 2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ))
        )
        return nn.Sequential(OrderedDict(layers))

    def _make_lateral_conv_output_conv(self, in_channels, out_channels, norm):
        use_bias = norm == ""
        layers = []
        out_channel_dict = {1: out_channels, 3: out_channels * 2}

        for i in range(5):
            k = 1 if i % 2 == 0 else 3
            p = (k - 1) // 2
            if i > 0:
                in_channels = out_channels
                out_channels = out_channel_dict[k]

            layers.append(
                (f"conv{i}", Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=p,
                    stride=1,
                    bias=use_bias,
                    norm=get_norm(norm, out_channels)
                ))
            )
            layers.append(
                (f"relu{i}", nn.LeakyReLU(0.1))
            )

        lateral_conv = nn.Sequential(OrderedDict(layers))
        out_conv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
            norm=get_norm(norm, out_channels // 2)
        )
        return lateral_conv, out_conv


@BACKBONE_REGISTRY.register()
def build_darknet53_backbone(cfg, input_shape):
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.DARKNET.STEM_OUT_CHANNELS,
        norm=cfg.MODEL.DARKNET.NORM
    )
    backbone = DarkNet(stem, [1, 2, 8, 8, 4])
    return backbone


@BACKBONE_REGISTRY.register()
def build_darknet53_fpn_backbone(cfg, input_shape):
    bottom_up = build_darknet53_backbone(cfg, input_shape)
    in_features = cfg.MODEL.DARKNET_FPN.IN_FEATURES
    out_channels = cfg.MODEL.DARKNET_FPN.OUT_CHANNELS
    num_classes = cfg.MODEL.YOLO.NUM_CLASSES
    num_anchors = len(cfg.MODEL.ANCHOR_GENERATOR.SIZES)
    norm = cfg.MODEL.DARKNET_FPN.NORM
    backbone = DarkNetFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_classes=num_classes,
        num_anchors=num_anchors,
        norm=norm
    )
    return backbone
