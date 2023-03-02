# @Time    : 2023/2/24 下午2:11
# @Author  : Boyang
# @Site    : 
# @File    : yolo.py
# @Software: PyCharm
from typing import List, Dict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_anchor_generator
from detectron2.config import configurable
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.meta_arch import DenseDetector
import matplotlib.pyplot as plt

from .matcher import Matcher
from .utils import visualize_image


@META_ARCH_REGISTRY.register()
class YoloV3(DenseDetector):

    @configurable
    def __init__(
            self,
            *,
            in_features,
            backbone,
            head,
            anchor_generator,
            box2box_transform,
            anchor_matcher,
            num_classes,
            pixel_mean,
            pixel_std
    ):
        super(YoloV3, self).__init__(backbone, head, in_features, pixel_std=pixel_std, pixel_mean=pixel_mean)
        self.backbone = backbone
        self.in_features = in_features
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        in_features = cfg.MODEL.YOLO.IN_FEATURES
        feature_shape = [backbone_shape[f] for f in in_features]
        head = YoloV3Head(cfg, feature_shape)
        anchor_generator = build_anchor_generator(cfg, feature_shape)

        return {
            "in_features": in_features,
            "backbone": backbone,
            "head": head,
            "anchor_generator": anchor_generator,
            "box2box_transform": ...,
            "anchor_matcher": Matcher(
                threshold=cfg.MODEL.YOLO.IOU_THRESHOLD,
                labels=cfg.MODEL.YOLO.IOU_LABELS,
                allow_low_quality_matches=False
            ),
            "num_classes": cfg.MODEL.YOLO.NUM_CLASSES,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD
            # LOSS PARAMETER
        }

    def forward_training(self, images, features, predictions, gt_instances):
        pred_logits, pred_confs, pred_anchor_deltas, = self._transpose_dense_predictions(
            predictions, [self.num_classes, 1, 4]
        )
        anchors = self.anchor_generator(features)
        # debug
        visualize_image(images[0], anchors, self.pixel_mean, self.pixel_std)

        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)

        # debug
        visualize_image(images[0], gt_boxes[0])

        backbone_shapes = [backbone_shape for backbone_shape in self.backbone.output_shape()]
        stride = [backbone_shapes[f].stride for f in self.head_in_features]

        # return self.losses()

    def losses(self, anchors, pred_logits, pred_confs, pred_anchor_deltas, gt_labels, gt_boxes):
        """
        计算三个loss: confidence_loss, logits_loss, anchor_deltas_loss
        """

        return {}

    @torch.no_grad()
    def label_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        """
        anchors: generate A anchor for each point on each feature map.
        gt_instances: a list, one instance contain all gt instance on a image.
        """
        anchors = Boxes.cat(anchors)
        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_gt_idx, anchor_labels = self.anchor_matcher(match_quality_matrix, anchors, gt_per_image.gt_boxes)
            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_gt_idx]
                gt_labels_i = gt_per_image.gt_classes[matched_gt_idx]
                gt_labels_i[anchor_labels == 0] = self.num_classes
                gt_labels_i[anchor_labels == -1] = -1
            else:
                gt_labels_i = torch.zeros_like(anchors.tensor) + self.num_classes
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes


class YoloV3Head(nn.Module):
    @configurable
    def __init__(self, *, num_classes, num_anchors):
        super(YoloV3Head, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        return {
            "num_classes": cfg.MODEL.YOLO.NUM_CLASSES,
            "num_anchors": len(cfg.MODEL.ANCHOR_GENERATOR.SIZES)
        }

    def forward(self, features: List[Tensor]):
        pred_confs = []
        pred_anchor_deltas = []
        pred_logits = []

        for feature in features:
            N, _, H, W = feature.shape
            feature = feature.view(feature.shape[0], -1, self.num_anchors, feature.shape[-2],
                                   feature.shape[-1])  # [B, 4+1+C, H, W]
            pred_anchor_delta = feature[:, :4].view(N, -1, H, W)
            # delta_xy = F.sigmoid(pred_anchor_delta[:, :2])
            pred_conf = feature[:, 4:5].view(N, -1, H, W)
            pred_logit = feature[:, 5:].view(N, -1, H, W)
            pred_anchor_deltas.append(pred_anchor_delta)  # [B, 4*A, H, W]
            pred_confs.append(pred_conf)  # [B, 1*A, H, W]
            pred_logits.append(pred_logit)  # [B, C*A, H, W]

        return pred_logits, pred_confs, pred_anchor_deltas
