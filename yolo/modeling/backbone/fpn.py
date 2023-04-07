# @Time    : 2023/3/6 下午2:12
# @Author  : Boyang
# @Site    : 
# @File    : fpn.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from itertools import zip_longest

from detectron2.modeling import BACKBONE_REGISTRY, build_anchor_generator
from detectron2.modeling.backbone import Backbone
from detectron2.layers import Conv2d, get_norm, CNNBlockBase
from . import build_darknet53_backbone
from .darknet import build_darknet53_backbone2

__all__ = ["DarkNetFPN"]


class UpSampleBlock(CNNBlockBase):
    def __init__(self, in_channels, out_channels, *, norm):
        super(UpSampleBlock, self).__init__(in_channels, out_channels, stride=1)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.upsample_conv = nn.Upsample(scale_factor=2.0, mode="nearest")

    def forward(self, x):
        out = self.conv(x)
        out = F.leaky_relu(out, 0.1)
        out = self.upsample_conv(out)
        return out


class LateralBlock(CNNBlockBase):
    def __init__(self, in_channels, out_channels, *, norm):
        super(LateralBlock, self).__init__(in_channels, out_channels, stride=1)
        self.blocks = nn.ModuleList()
        use_bias = norm == ""
        out_channel_dict = {1: out_channels, 3: out_channels * 2}
        for i in range(5):
            k = 1 if i % 2 == 0 else 3
            p = (k - 1) // 2
            if i > 0:
                in_channels = out_channels
                out_channels = out_channel_dict[k]
            conv = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=k,
                padding=p,
                stride=1,
                bias=use_bias,
                norm=get_norm(norm, out_channels)
            )
            self.blocks.append(conv)

        for conv in self.blocks:
            weight_init.c2_msra_fill(conv)

    def forward(self, x):
        for conv in self.blocks:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)

        return x


class OutputBlock(CNNBlockBase):
    def __init__(self, in_channels, out_channels, *, norm):
        super(OutputBlock, self).__init__(in_channels, out_channels, stride=1)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, in_channels * 2)
        )
        self.head = Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True
        )
        weight_init.c2_msra_fill(self.conv)
        weight_init.c2_msra_fill(self.head)

    def forward(self, x):
        out = self.conv(x)
        out = F.leaky_relu(out, 0.1)
        out = self.head(out)
        return out


class DarkNetFPN(Backbone):
    def __init__(
            self,
            bottom_up,
            in_features,
            out_channels,
            norm="",
    ):
        super(DarkNetFPN, self).__init__()
        self.bottom_up = bottom_up
        self.in_features = in_features

        input_shapes = bottom_up.output_shape()
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        self.output_channels = out_channels
        lateral_convs = []
        output_convs = []
        upsample_convs = []
        # reverse order, 从p5 -> p3
        for idx, in_channels in enumerate(reversed(in_channels_per_feature)):
            lateral_conv_out_channels = in_channels // 2
            # 循环p3 p4 p5 的in_channels 链接最后一层的conv的通道数是没有融合过程的
            # !!!在d5上lateral_conv的in_channels没有concat的过程，因此不需要调整in_channels
            if idx > 0:
                # concat layer,通过concat方式将两层融合,网络中规定上一层的通道数为这一层的1/2
                in_channels += in_channels // 2

            lateral_conv = LateralBlock(in_channels, lateral_conv_out_channels, norm=norm)
            lateral_convs.append(lateral_conv)
            self.add_module(f"fpn_lateral{idx}", lateral_conv)

            output_conv = OutputBlock(lateral_conv_out_channels, out_channels, norm=norm)
            output_convs.append(output_conv)
            self.add_module(f"fpn_output{idx}", output_conv)

            if idx + 1 < len(in_channels_per_feature):
                upsample_conv = UpSampleBlock(lateral_conv_out_channels, lateral_conv_out_channels // 2, norm=norm)
                upsample_convs.append(upsample_conv)
                self.add_module(f"fpn_upsample{idx}", upsample_conv)

        """
        重新调整了DarkNetFPN模型初始化结构，FPN模型结构分为：lateral_conv, output_conv, upsample_conv, 三个部分。
        lateral接收从backbone输出的特征 --> output, lateral和in_features的特征图名称是顺序对应的
        output中包含分类和回归head
        upsample中包含：conv1x1 - nn.upsample, 在p5上的计算不包含upsample
        """

        # reverser the order of convs
        self.lateral_convs = lateral_convs
        self.output_convs = output_convs
        self.upsample_convs = upsample_convs

        self._out_features = ["p3", "p4", "p5"]

        # 用于output_shape支持
        self._out_feature_channels = {k: input_shapes[f].channels for f, k in zip(self.in_features, self._out_features)}
        self._out_feature_strides = {k: input_shapes[f].stride for f, k in zip(self.in_features, self._out_features)}

    @property
    def size_divisibility(self):
        return 32

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        result = []

        topdown_feature = None
        for idx, (lateral_conv, output_conv, upsample_conv) in enumerate(zip_longest(
                self.lateral_convs, self.output_convs, self.upsample_convs
        )):
            features = self.in_features[-idx - 1]
            features = bottom_up_features[features]
            if topdown_feature is None:
                pre_features = lateral_conv(features)
                result.insert(0, output_conv(pre_features))
            else:
                features = torch.cat([topdown_feature, features], dim=1)
                pre_features = lateral_conv(features)
                result.insert(0, output_conv(pre_features))
            if upsample_conv:
                topdown_feature = upsample_conv(pre_features)

        return dict(zip(self._out_features, result))

    def darknet_modules(self):
        for name, conv in self.named_modules():
            if isinstance(conv, Conv2d):
                yield name, conv


@BACKBONE_REGISTRY.register()
def build_darknet53_fpn_backbone(cfg, input_shape):
    bottom_up = build_darknet53_backbone2(cfg, input_shape)
    in_features = cfg.MODEL.DARKNET.OUT_FEATURES
    backbone_shape = bottom_up.output_shape()
    feature_shape = [backbone_shape[f] for f in in_features]
    num_classes = cfg.MODEL.YOLOV3.NUM_CLASSES
    num_anchors = build_anchor_generator(cfg, feature_shape).num_cell_anchors
    assert (
            len(set(num_anchors)) == 1
    ), "Using different number of anchors between levels is not currently supported!"
    num_anchors = num_anchors[0]

    out_channels = (4 + 1 + num_classes) * num_anchors

    norm = cfg.MODEL.DARKNET.NORM
    backbone = DarkNetFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=norm
    )
    return backbone
