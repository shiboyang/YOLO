# @Time    : 2023/3/3 上午9:48
# @Author  : Boyang
# @Site    : 
# @File    : box_regression.py
# @Software: PyCharm
from typing import Tuple
import torch

from detectron2.structures import Boxes


class Box2BoxTransform:

    def __init__(self, weights: Tuple[float, float, float, float], scale_clamp: float):
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes, strides):
        # calculate the center point of gt_boxes and weight height
        gt_boxes = src_boxes
        anchors = target_boxes

        gt_wh = gt_boxes[..., 2:] - gt_boxes[..., :2]
        gt_cxy = gt_boxes[..., :2] + gt_wh / 2
        # for anchor calculate
        anchors = Boxes.cat(anchors).tensor
        anchors_wh = anchors[..., 2:] - anchors[..., :2]

        gt_delta_cxy = gt_cxy / strides[..., None]
        delta_cxy = gt_delta_cxy - torch.floor(gt_delta_cxy) - 0.5

        delta_wh = torch.log(gt_wh / anchors_wh)

        wx, wy, ww, wh = self.weights
        dx = wx * delta_cxy[..., 0]
        dy = wy * delta_cxy[..., 1]
        dw = wh * delta_wh[..., 0]
        dh = ww * delta_wh[..., 1]

        deltas = torch.stack((dx, dy, dw, dh), dim=-1)

        return deltas

    def apply_deltas(self, deltas, boxes):
        # todo scale_clamp
        pass
