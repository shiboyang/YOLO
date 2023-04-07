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

from detectron2.layers import get_norm, Conv2d, ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
from detectron2.layers.blocks import CNNBlockBase


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


class ResidualBlock(CNNBlockBase):
    def __init__(self, in_channels: int, out_channels: int, norm: str, downsample: bool = False):
        """
        Residual layer
        一个残差单元包含: downsample - conv 1x1 - conv 3x3
        :param in_channels: 输入特征图的通道数
        :param out_channels: 经过这个单元处理后输出特征图的通道数
        :param norm
        :param downsample: 是否使用 k=3x3 s=2 pad=1 进行下采样
        """
        super(ResidualBlock, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                            stride=2 if downsample else 1)
        if downsample:
            self.downsample = Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=2,
                bias=False,
                padding=1,
                norm=get_norm(norm, self.out_channels)
            )
        else:
            self.downsample = None
        # bottleneck_channels: 对1x1卷积输出通道数
        bottleneck_channels = self.out_channels // 2
        self.conv1 = Conv2d(
            in_channels=self.out_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=get_norm(norm, bottleneck_channels)
        )
        self.conv2 = Conv2d(
            in_channels=bottleneck_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, self.out_channels)
        )

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        if downsample:
            weight_init.c2_msra_fill(self.downsample)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
            x = F.leaky_relu(x, 0.1)

        residual = x.clone()
        out = self.conv1(x)
        out = F.leaky_relu(out, 0.1)

        out = self.conv2(out)
        out = F.leaky_relu(out, 0.1)

        out += residual
        return out


class DarkNet2(Backbone):
    """
    Implement :paper:`DarkNet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"res1": 1, "res2": 2, "res3": 3, "res4": 4, "res5": 5}.get(f, 0) for f in out_features]
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 1)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, norm):
        """
        Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.

        Returns:
            list[CNNBlockBase]: a list of block module.

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        for i in range(num_blocks):
            downsample = (i == 0)
            blocks.append(
                block_class(in_channels=in_channels, out_channels=out_channels, downsample=downsample, norm=norm)
            )
            in_channels = out_channels
        return blocks

    # @staticmethod
    # def make_default_stages(depth, block_class=None, **kwargs):
    #     """
    #     Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
    #     If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
    #     instead for fine-grained customization.
    #
    #     Args:
    #         depth (int): depth of ResNet
    #         block_class (type): the CNN block class. Has to accept
    #             `bottleneck_channels` argument for depth > 50.
    #             By default it is BasicBlock or BottleneckBlock, based on the
    #             depth.
    #         kwargs:
    #             other arguments to pass to `make_stage`. Should not contain
    #             stride and channels, as they are predefined for each depth.
    #
    #     Returns:
    #         list[list[CNNBlockBase]]: modules in all stages; see arguments of
    #             :class:`ResNet.__init__`.
    #     """
    #     num_blocks_per_stage = {
    #         18: [2, 2, 2, 2],
    #         34: [3, 4, 6, 3],
    #         50: [3, 4, 6, 3],
    #         101: [3, 4, 23, 3],
    #         152: [3, 8, 36, 3],
    #     }[depth]
    #     if block_class is None:
    #         block_class = BasicBlock if depth < 50 else BottleneckBlock
    #     if depth < 50:
    #         in_channels = [64, 64, 128, 256]
    #         out_channels = [64, 128, 256, 512]
    #     else:
    #         in_channels = [64, 256, 512, 1024]
    #         out_channels = [256, 512, 1024, 2048]
    #     ret = []
    #     for (n, s, i, o) in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
    #         if depth >= 50:
    #             kwargs["bottleneck_channels"] = o // 4
    #         ret.append(
    #             ResNet.make_stage(
    #                 block_class=block_class,
    #                 num_blocks=n,
    #                 stride_per_block=[s] + [1] * (n - 1),
    #                 in_channels=i,
    #                 out_channels=o,
    #                 **kwargs,
    #             )
    #         )
    #     return ret


def make_stage(*args, **kwargs):
    return


@BACKBONE_REGISTRY.register()
def build_darknet53_backbone2(cfg, input_shape):
    """
       Create a DarkNet instance from config.

       Returns:
           DarkNet: a :class:`DarkNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.DARKNET.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.DARKNET.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.DARKNET.OUT_FEATURES
    depth = cfg.MODEL.DARKNET.DEPTH
    in_channels = cfg.MODEL.DARKNET.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.DARKNET.RES1_OUT_CHANNELS
    # fmt: on

    num_blocks_per_stage = {
        53: [1, 2, 8, 8, 4],
    }[depth]

    stages = []

    for idx, stage_idx in enumerate(range(1, 6)):
        stage_kargs = {
            "block_class": ResidualBlock,
            "num_blocks": num_blocks_per_stage[idx],
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        blocks = DarkNet2.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        stages.append(blocks)

    return DarkNet2(stem, stages, out_features=out_features, freeze_at=freeze_at)
