# @Time    : 2023/2/24 下午2:11
# @Author  : Boyang
# @Site    : 
# @File    : yolo.py
# @Software: PyCharm
import os
from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_anchor_generator
from detectron2.config import configurable
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList
from detectron2.modeling.meta_arch import DenseDetector
from detectron2.utils.events import get_event_storage

from .box_regression import Box2BoxTransform
from .matcher import Matcher, Matcher2
from .utils import visualize_image, draw_point, pairwise_iou_with_wh

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
            anchor_matcher: Matcher2,
            num_classes,
            # loss parameters
            # ...
            test_score_thresh=0.25,
            test_topk_candidates=1000,
            test_nms_thresh=0.5,
            max_detections_per_image=100,
            pixel_mean,
            pixel_std,
    ):
        super(YoloV3, self).__init__(backbone, head, in_features, pixel_std=pixel_std, pixel_mean=pixel_mean)
        self.backbone = backbone
        self.in_features = in_features
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.num_classes = num_classes

        # inference parameters
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_pre_image = max_detections_per_image

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
            "anchor_matcher": Matcher2(
                threshold=cfg.MODEL.YOLO.IOU_THRESHOLD,
                labels=cfg.MODEL.YOLO.IOU_LABELS,
                allow_low_quality_matches=False
            ),
            "num_classes": cfg.MODEL.YOLO.NUM_CLASSES,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def forward_training(self, images, features, predictions, gt_instances):
        # del images
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

        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        # DEBUG
        visualize_image(images[0], gt_instances[0].gt_boxes, 0, 0)
        draw_point(images[0], grid_sizes[2], 32)

        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances, grid_sizes)

        return self.losses(anchors, pred_logits, pred_confs, pred_anchor_deltas, gt_labels, gt_boxes,
                           self._get_strides())

    def _get_strides(self) -> List[int]:
        backbone_shapes = self.backbone.output_shape()
        strides = [backbone_shapes[f].stride for f in self.head_in_features]
        return strides

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
        # normalizer = self._ema_update("loss_normalizer", max(num_positive_anchors, 1), 100)

        # 计算confidence loss
        pred_confs = torch.cat(pred_confs, dim=1)
        confidence_target = torch.zeros_like(pred_confs)
        confidence_target[positive_mask] = 1
        confidence_loss = self._compute_confidence_loss(
            pred_confs,
            confidence_target,
            positive_mask,
            negative_mask
        )

        # 计算logits loss
        pred_logits = torch.cat(pred_logits, dim=1)
        target = F.one_hot(gt_labels[positive_mask], self.num_classes + 1)[:, :-1].to(pred_logits.dtype)
        logits_loss = self._compute_logits_loss(
            pred_logits[positive_mask],
            target
        )

        # 计算regression loss
        strides = torch.tensor(strides, device=self.device)
        strides = strides.repeat_interleave(
            torch.tensor([deltas.shape[1] for deltas in pred_anchor_deltas], device=self.device)
        )
        pred_anchor_deltas = torch.cat(pred_anchor_deltas, dim=1)
        regression_loss = self._compute_box_regression_loss(
            pred_anchor_deltas,
            gt_boxes,
            anchors,
            positive_mask,
            strides,
        )

        return {
            "confs_loss": confidence_loss,
            "logits_loss": logits_loss,
            "regression_loss": regression_loss
        }

    def _compute_box_regression_loss(self, pred_anchor_deltas, gt_boxes, anchors, positive_mask, strides):
        target = self.box2box_transform.get_deltas(gt_boxes, anchors, strides)
        regression_loss = F.smooth_l1_loss(
            pred_anchor_deltas[positive_mask],
            target[positive_mask],
            reduction="sum"
        )

        return regression_loss

    def _compute_confidence_loss(self, pred_confs, target, objectness_mask, noobjectness_mask):
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

    def _compute_logits_loss(self, pred_logits, target):
        logits_loss = F.binary_cross_entropy(pred_logits, target, reduction="sum")
        return logits_loss

    @torch.no_grad()
    def label_anchors(self, anchors: List[Boxes], gt_instances: List[Instances], grid_sizes):
        """
        anchors: generate a anchor for each point on each feature map.
        gt_instances: a list, one instances contain all gt instance on a image.
        """
        # anchors = Boxes.cat(anchors)
        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            # match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            # # v1
            # # # 对所有的点分类 正样本(1) 负样本(0) 忽略样本(-1)
            # matched_gt_idx, anchor_labels = self.anchor_matcher(match_quality_matrix, anchors, gt_per_image.gt_boxes)

            # v2
            matched_gt_idx, anchor_labels = [], []
            for anchor, num_anchor, feature_map_size in zip(anchors, self.anchor_generator.num_anchors, grid_sizes):
                anchor = anchor.tensor
                anchor = anchor.reshape([feature_map_size[0], feature_map_size[1], num_anchor, 4])
                gt_boxes = gt_per_image.gt_boxes.tensor
                anchor_i = torch.stack([anchor[0, 0, i, :] for i in range(num_anchor)])
                anchor_wh = anchor_i[:, 2:] - anchor_i[:, :2]
                gt_boxes_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
                iou_quality_matrix = pairwise_iou_with_wh(anchor_wh, gt_boxes_wh)

                matched_gt_idx_i, anchor_labels_i = self.anchor_matcher(match_quality_matrix, gt_per_image.gt_boxes,
                                                                        self.anchor_generator.strides, grid_sizes)

                matched_gt_idx.append(matched_gt_idx_i)
                anchor_labels.append(anchor_labels_i)

            # DEBUG visualize matched anchors and gt_anchors
            if os.environ.get("DEBUG"):
                img = self.images[0].clone()
                visualize_image(
                    img,
                    [gt_per_image.gt_boxes, Boxes(anchors.tensor[anchor_labels == 1])],
                    self.pixel_mean,
                    self.pixel_std,
                )

            matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_gt_idx]
            gt_labels_i = gt_per_image.gt_classes[matched_gt_idx]
            gt_labels_i[anchor_labels == 0] = self.num_classes
            gt_labels_i[anchor_labels == -1] = -1

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def forward_inference(self, images: ImageList, features: List[Tensor], predictions: List[List[Tensor]]):
        """
        推理函数，在test阶段DenseDetector.forward会调用此函数
        Return bounding-box detected results by thresholding on scores and applying non-maximum suppression
        Arguments:
            images:
            features
            predictions:

        """
        pred_logits, pred_confs, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes, 1, 4]
        )
        anchors = self.anchor_generator(features)
        results: List[Instances] = []
        for img_idx, image_size in enumerate(images.image_sizes):
            logits_per_image = [x[img_idx] for x in pred_logits]
            confs_per_image = [x[img_idx] for x in pred_confs]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, logits_per_image, confs_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)

        return results

    def inference_single_image(self, anchors, logits, confidences, deltas, image_size):
        """

        """
        pred = self.__decode_multi_level_predictions(
            anchors=anchors,
            pred_logits=logits,
            pred_confs=confidences,
            pred_deltas=deltas,
            score_thresh=self.test_score_thresh,
            topk_candidates=self.test_topk_candidates,
            image_size=image_size
        )

        keep = batched_nms(pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.test_nms_thresh)

        return pred[keep[:self.max_detections_pre_image]]

    def __decode_multi_level_predictions(
            self,
            anchors: List[Boxes],
            pred_logits: List[Tensor],
            pred_confs: List[Tensor],
            pred_deltas: List[Tensor],
            score_thresh: float,
            topk_candidates: int,
            image_size: Tuple[int, int],
    ) -> Instances:
        predictions: List[Instances] = []
        strides = self._get_strides()
        assert len(strides) == len(anchors)

        for box_cls_i, box_reg_i, confs_i, anchors_i, stride_i in zip(pred_logits, pred_deltas, pred_confs, anchors,
                                                                      strides):
            predictions.append(
                self.__decode_per_level_predictions(
                    anchors=anchors_i,
                    pred_scores=box_cls_i,
                    pred_confs=confs_i,
                    pred_deltas=box_reg_i,
                    score_thresh=score_thresh,
                    topk_candidates=topk_candidates,
                    image_size=image_size,
                    stride=stride_i
                )
            )

        return predictions[0].cat(predictions)

    def __decode_per_level_predictions(
            self,
            anchors: Boxes,
            pred_scores: Tensor,
            pred_confs: Tensor,
            pred_deltas: Tensor,
            score_thresh: float,
            topk_candidates: int,
            image_size: Tuple[int, int],
            stride: int
    ) -> Instances:
        """
        在一个feature map上计算每一个点的 分类 回归 和 置信度
        Args:
            anchors
            pred_scores: score for each point of feature map. shape: [HW, num_classes]
            pred_confs: confidence. shape: [HW, 1]
            pred_deltas: (dx, dy, dw, dh). offset. shape [HW, 4]

        """
        # the first filter. filtering the object confidence.
        confs_mask = pred_confs > score_thresh
        pred_confs = pred_confs[confs_mask]  # [num_mask, 1]
        pred_deltas = pred_deltas[confs_mask]  # [num_mask, 4]
        pred_scores = pred_scores[confs_mask]  # [num_mask, 1]
        # the second filter, filtering classes scores.
        # P(cls)  = P(cls) * P(cls|obj)
        pred_scores = pred_scores * pred_confs[:, None]
        if self.num_classes > 1:
            # multiple class
            # i: HW's index. j: classes index.
            i, j = torch.nonzero(pred_scores > score_thresh, as_tuple=True)
            pred_confs = pred_scores[j]
            pred_cls = j
            pred_deltas = pred_deltas[i]
        else:
            # single class.
            pred_confs = ...
            pred_cls = ...
            pred_deltas = ...

        # # 再次通过topk个点过滤
        num_topk = min(topk_candidates, pred_confs.size(0))
        pred_confs, topk_idxs = pred_confs.topk(num_topk)
        pred_cls = pred_cls[topk_idxs]
        #
        pred_boxes = self.box2box_transform.apply_deltas(pred_deltas[topk_idxs], anchors.tensor[topk_idxs], stride)

        return Instances(
            image_size=image_size, pred_boxes=Boxes(pred_boxes), scores=pred_scores, pred_classes=pred_cls
        )


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
            feature = feature.view(N, self.num_anchors, -1, H, W)  # [B, A, 4+1+C, H, W]

            delta_i = feature[:, :, :4].reshape(N, -1, H, W)
            confs_i = feature[:, :, 4:5].reshape(N, -1, H, W)
            logits_i = feature[:, :, 5:].reshape(N, -1, H, W)
            #
            delta_i[:, 0::4] = torch.sigmoid(delta_i[:, 0::4]) - 0.5
            delta_i[:, 1::4] = torch.sigmoid(delta_i[:, 1::4]) - 0.5

            confs_i = torch.sigmoid(confs_i)
            logits_i = torch.sigmoid(logits_i)

            pred_anchor_deltas.append(delta_i)  # [B, A*4, H, W]
            pred_confs.append(confs_i)  # [B, A*1, H, W]
            pred_logits.append(logits_i)  # [B, A*C, H, W]

        return pred_logits, pred_confs, pred_anchor_deltas
