# @Time    : 2023/2/24 下午2:11
# @Author  : Boyang
# @Site    : 
# @File    : yolo.py
# @Software: PyCharm
from typing import List, Dict

import torch
import torch.nn as nn
from torch import Tensor

from detectron2.modeling import META_ARCH_REGISTRY, build_anchor_generator
from detectron2.config import configurable
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import Boxes, ImageList, Instances


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
            device,
            pixel_mean,
            pixel_std
    ):
        super(YoloV3, self).__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.num_classes = num_classes
        self.device = device

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        in_features = cfg.MODEL.YOLO.IN_FEATURES
        feature_shape = [backbone_shape[f] for f in in_features]

        anchor_generator = build_anchor_generator(cfg, feature_shape)

        return {
            "in_features": in_features,
            "backbone": backbone,
            "anchor_generator": anchor_generator,
            "box2box_transform": ...,
            "anchor_matcher": ...,
            "num_classes": cfg.MODEL.YOLO.NUM_CLASSES,
            "device": cfg.MODEL.DEVICE,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD
            # LOSS PARAMETER
        }

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]  # List[[B,(C+5)*A,H,W]]
        anchors = self.anchor_generator(features)
        features = [
            # [B, C+4, A, H, W]
            f.view(f.shape[0], -1, self.anchor_generator.box_dim + self.num_classes, f.shape[-2], f.shape[-1])
            # [B, H, W, A, C+4]
            .permute(0, 3, 4, 1, 2) for f in features
        ]

        pred_logits = []
        pred_anchor_delta = []
        for f in features:
            pred_logits.append(f[..., :self.num_classes])
            pred_anchor_delta.append(f[..., self.num_classes:])

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.label_anchors(anchors, gt_instances)

    @torch.no_grad()
    def label_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        anchors = Boxes.cat(anchors)
        gt_labels = []
        matched_gt_boxes = []

    def losses(self):
        ...

    def preprocess_image(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    def _move_to_current_device(self, x):
        return x.to(self.device)


class YoloV3_Head(nn.Module):
    @configurable
    def __init__(self, *, num_class, num_anchors):
        super(YoloV3_Head, self).__init__()
        self.num_class = num_class
        self.num_anchors = num_anchors

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_class": cfg.MODEL.YOLO.NUM_CLASSES,
            "num_anchors": len(cfg.MODEL.ANCHOR_GENERATOR.SIZES)
        }

    def forward(self, features: List[Tensor]):
        pred_confs = []
        pred_anchor_deltas = []
        pred_logits = []

        for feature in features:
            ...

        return pred_confs, pred_anchor_deltas, pred_logits
