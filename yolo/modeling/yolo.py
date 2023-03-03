# @Time    : 2023/2/24 下午2:11
# @Author  : Boyang
# @Site    : 
# @File    : yolo.py
# @Software: PyCharm
import os
from typing import List
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
from detectron2.utils.events import get_event_storage
from .box_regression import Box2BoxTransform

from .matcher import Matcher
from .utils import visualize_image

__all__ = ["YoloV3"]


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
            box2box_transform: Box2BoxTransform,
            anchor_matcher: Matcher,
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
            "box2box_transform": Box2BoxTransform((1., 1., 1., 1.), 0.5),
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
        # DEBUG:
        if os.environ.get("DEBUG"):
            self.images = images

        pred_logits, pred_confs, pred_anchor_deltas, = self._transpose_dense_predictions(
            predictions, [self.num_classes, 1, 4]
        )
        anchors = self.anchor_generator(features)
        # DEBUG
        if os.environ.get("DEBUG"):
            visualize_image(images[0], anchors, self.pixel_mean, self.pixel_std, separate_show=True)

        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)

        backbone_shapes = self.backbone.output_shape()
        strides = [backbone_shapes[f].stride for f in self.head_in_features]

        return self.losses(anchors, pred_logits, pred_confs, pred_anchor_deltas, gt_labels, gt_boxes, strides)

    def losses(self, anchors: List[Boxes], pred_logits, pred_confs, pred_anchor_deltas, gt_labels, gt_boxes, strides):
        """
        计算三个loss: confidence_loss, logits_loss, regression_loss
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # [N, num_anchors]
        gt_boxes = torch.stack(gt_boxes)

        valid_mask = gt_labels >= 0
        positive_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        negative_mask = gt_labels == self.num_classes

        num_positive_anchors = positive_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_positive_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_positive_anchors, 1), 100)

        # 计算confidence loss
        pred_confs = torch.cat(pred_confs, dim=1)
        confidence_target = torch.zeros_like(pred_confs)
        confidence_target[positive_mask] = 1
        confidence_loss = self._dense_confidence_loss(
            pred_confs,
            confidence_target,
            positive_mask,
            negative_mask
        )

        # 计算logits loss
        pred_logits = torch.cat(pred_logits, dim=1)
        target = F.one_hot(gt_labels[positive_mask], self.num_classes + 1)[:, :-1].to(pred_logits.dtype)
        logits_loss = self._dense_logits_loss(
            pred_logits[positive_mask],
            target
        )

        # 计算regression loss
        strides = torch.tensor(strides, device=self.device)
        strides = strides.repeat_interleave(
            torch.tensor([deltas.shape[0] * deltas.shape[1] for deltas in pred_anchor_deltas], device=self.device)
        )
        pred_anchor_deltas = torch.cat(pred_anchor_deltas, dim=1)
        regression_loss = self._dense_box_regression_loss(
            pred_anchor_deltas,
            gt_boxes,
            anchors,
            positive_mask,
            strides,
        )

        return {
            "confs_loss": confidence_loss / valid_mask.sum().item(),
            "logits_loss": logits_loss / normalizer,
            "regression_loss": regression_loss / normalizer
        }

    def _dense_box_regression_loss(self, pred_anchor_deltas, gt_boxes, anchors, positive_mask, strides):
        target = self.box2box_transform.get_deltas(gt_boxes, anchors, strides)
        regression_loss = F.smooth_l1_loss(
            pred_anchor_deltas[positive_mask],
            target[positive_mask],
            reduction="sum"
        )

        return regression_loss

    def _dense_confidence_loss(self, pred_confs, target, objectness_mask, noobjectness_mask):
        objectness_loss = F.binary_cross_entropy(
            pred_confs[objectness_mask],
            target[objectness_mask],
            reduction="sum"
        )
        noobjectness_loss = F.binary_cross_entropy(
            pred_confs[noobjectness_mask],
            target[noobjectness_mask],
            reduction="sum"
        )
        confidence_loss = objectness_loss + noobjectness_loss
        return confidence_loss

    def _dense_logits_loss(self, pred_logits, target):
        logits_loss = F.binary_cross_entropy(pred_logits, target, reduction="sum")
        return logits_loss

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
            # DEBUG visualize matched anchors and gt_anchors
            if os.environ.get("DEBUG"):
                img = self.images[0].clone()
                visualize_image(
                    img,
                    [gt_per_image.gt_boxes, Boxes(anchors.tensor[anchor_labels == 1])],
                    self.pixel_mean,
                    self.pixel_std,
                )

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
        """
        pred_anchor_delta 中的 txy 是相对与中心点的偏移量 sigmoid(txy)
        pred_confs 是sigmoid(tc)
        pred_logits 就是原始的 tclass_score
        """
        pred_confs = []
        pred_anchor_deltas = []
        pred_logits = []

        for feature in features:
            N, _, H, W = feature.shape
            feature = feature.view(feature.shape[0], -1, self.num_anchors, feature.shape[-2],
                                   feature.shape[-1])  # [B, 4+1+C, H, W]
            pred_anchor_delta = feature[:, :4].view(N, -1, H, W)  # [B, 4, H, W]
            delta_cxy = torch.sigmoid(pred_anchor_delta[:, :2]) - 0.5
            pred_anchor_delta[:, :2] = delta_cxy

            pred_conf = feature[:, 4:5].view(N, -1, H, W)
            pred_conf = torch.sigmoid(pred_conf)

            pred_logit = feature[:, 5:].view(N, -1, H, W)
            pred_logit = torch.sigmoid(pred_logit)

            pred_anchor_deltas.append(pred_anchor_delta)  # [B, 4*A, H, W]
            pred_confs.append(pred_conf)  # [B, 1*A, H, W]
            pred_logits.append(pred_logit)  # [B, C*A, H, W]

        return pred_logits, pred_confs, pred_anchor_deltas
