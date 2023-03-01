# @Time    : 2023/2/27 下午2:25
# @Author  : Boyang
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import torch

from detectron2.structures import Boxes


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes):
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[..., 2:], boxes2[..., 2:]) - torch.max(boxes1[..., :2], boxes2[..., :2])
    width_height.clamp_(min=0)
    intersection = width_height.prod(dim=-1)
    return intersection


def pairwise_iou(boxes1: Boxes, boxes2: Boxes):
    area1 = boxes1.area()
    area2 = boxes2.area()
    intersection = pairwise_intersection(boxes1, boxes2)
    iou = torch.where(
        intersection > 0,
        intersection / (area1 + area2 - intersection),
        torch.zeros(1, dtype=intersection.dtype, device=intersection.device)
    )
    return iou
