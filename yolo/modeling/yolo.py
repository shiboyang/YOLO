# @Time    : 2023/2/24 下午2:11
# @Author  : Boyang
# @Site    : 
# @File    : yolo.py
# @Software: PyCharm
from typing import List
import torch.nn as nn

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.modeling.backbone import build_backbone


@META_ARCH_REGISTRY.register()
class YoloV3(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            backbone,
            out_features: List[str]
    ):
        super(YoloV3, self).__init__()
        self.backbone = backbone
        self._out_features = out_features

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()

        out_features = cfg.MODEL.DARKNET_FPN.OUT_FEATURES

        return {
            "backbone": backbone,
            "out_features": out_features
        }

    def forward(self, x):
        out_features = self.backbone(x)
        out_features = [out_features[f] for f in self._out_features]
