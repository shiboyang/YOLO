# @Time    : 2023/2/24 下午2:11
# @Author  : Boyang
# @Site    : 
# @File    : yolo.py
# @Software: PyCharm
from typing import List

import torch
import torch.nn as nn

from detectron2.modeling import META_ARCH_REGISTRY, build_anchor_generator
from detectron2.config import configurable
from detectron2.modeling.backbone import build_backbone


@META_ARCH_REGISTRY.register()
class YoloV3(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            in_features,
            backbone,
            anchor_generator,
            box2box_transform,
            anchor_matcher,
            num_classes,
    ):
        super(YoloV3, self).__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.anchor_generator = anchor_generator

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shape = [backbone_shape[f] for f in cfg.MODEL.HEAD.IN_FEATURES]

        anchor_generator = build_anchor_generator(cfg, feature_shape)
        # 在feature_map上生成anchor
        anchor_generator.strides = [1, 1, 1]

        return {
            "backbone": backbone,
            "anchor_generator": anchor_generator,
            "box2box_transform": ...,
            "anchor_matcher": ...,
            "num_classes": cfg.MODEL.HEAD.NUM_CLASSES,
            # LOSS PARAMETER
        }

    def forward(self, batched_inputs):
        features = self.backbone(batched_inputs)
        features = [features[f] for f in self.in_features]  # List[[B,(C+5)*A,H,W]]
        anchors = self.anchor_generator(features)
        features = [
            # [B, C+5, A, H, W]
            f.view(batched_inputs.shape[0], -1, anchors.box_dim, batched_inputs.shape[-2], batched_inputs.shape[-1])
            # [B, H, W, A, C+5]
            .permute(0, 3, 4, 2, 1) for f in features
        ]
        features = torch.cat()
        pred_logits = ...

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.label_anchors(anchors, gt_instances)

    def label_anchors(self, anchors, gt_instances):
        ...

    def losses(self):
        ...
