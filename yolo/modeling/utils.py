# @Time    : 2023/2/27 下午2:25
# @Author  : Boyang
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import os

import torch
import matplotlib.pyplot as plt
import cv2
from torch import Tensor
from detectron2.structures.boxes import Boxes
from typing import Optional, List, Dict, Tuple, Union, Iterator
import numpy as np


def pairwise_iou_with_wh(box1_wh: Tensor, box2_wh: Tensor):
    """

    """
    min_w = torch.minimum(box1_wh[:, 0][:, None], box2_wh[:, 0][None, :])
    min_h = torch.minimum(box1_wh[:, 1][:, None], box2_wh[:, 1][None, :])
    inter_area = min_w * min_h
    box1_area = box1_wh[:, 0] * box1_wh[:, 1]
    box2_area = box2_wh[:, 0] * box2_wh[:, 1]
    union_area = box1_area[:, None] + box2_area[None, :] - inter_area
    iou = inter_area / union_area
    return iou


def tensor_to_np(tensor, pixel_mean=None):
    # torch.tensor([103.530, 116.280, 123.675], device=tensor.device)[:, None, None]
    img = tensor.clone()
    if pixel_mean:
        img = img + pixel_mean
    img = img.permute(1, 2, 0)
    img = img.to("cpu", torch.uint8).numpy()
    ndarr = np.ascontiguousarray(img)
    return ndarr


def visualize_image(
        image: Tensor,
        boxes: [Optional[Boxes], Optional[List[Boxes]]],
        pixel_mean,
        pixel_std,
        show_original_image=False,
        separate_show=False

):
    img = tensor_to_np(image)
    colors = [(255, 0, 0), (255, 255, 0), (255, 255, 255), (255, 0, 255), (0, 255, 255)]
    if show_original_image:
        plt.imshow(img[..., ::-1])
        plt.show()

    copy_img = img.copy()
    if isinstance(boxes, list):
        for i, boxes_i in enumerate(boxes):
            boxes = boxes_i.tensor
            copy_img = img.copy() if separate_show else copy_img
            i = i % len(colors)
            for (x0, y0, x1, y1) in boxes:
                copy_img = cv2.rectangle(copy_img, (int(x0), int(y0)), (int(x1), int(y1)), color=colors[i], thickness=1)
            if separate_show:
                plt.imshow(copy_img[..., ::-1])
                plt.show()

        if not separate_show:
            plt.imshow(copy_img[..., ::-1])
            plt.show()

    elif isinstance(boxes, (Boxes, Tensor)):
        if isinstance(boxes, Boxes):
            boxes = boxes.tensor
        for (x0, y0, x1, y1) in boxes:
            img = cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 0, 0),
                                thickness=1)
        plt.imshow(img[..., ::-1])
        plt.show()
    else:
        print(f"Unsupported type {type(boxes)}")
        pass


def draw_point(image, grid_size, stride=1):
    img = tensor_to_np(image) if isinstance(image, torch.Tensor) else image
    grid_height, grid_width = grid_size
    x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32)
    y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32)
    x, y = torch.meshgrid(x, y)
    x, y = x.reshape(-1), y.reshape(-1)
    point1 = torch.stack([x, y], dim=-1).numpy()

    plt.scatter(point1[:, 0], point1[:, 1], marker=".")
    plt.imshow(img)
    plt.show()


def visualize_predictions(image, boxes: torch.Tensor, classes: torch.Tensor, scores: torch.Tensor,
                          cls_map=None):
    if isinstance(image, str):
        img = cv2.imread(os.path.expanduser(image))
    elif isinstance(image, torch.Tensor):
        img = image.clone()
        img = img * 255
        img = img.permute(1, 2, 0)
        img = img.to("cpu", torch.uint8).numpy()
        img = np.ascontiguousarray(img)
    else:
        raise NotImplementedError

    for (x1, y1, x2, y2), cls_id, score in zip(boxes, classes, scores):
        pt1 = int(x1), int(y1)
        pt2 = int(x2), int(y2)
        cv2.rectangle(img, pt1, pt2, color=(0, 0, 255), thickness=1)
        cls = cls_map[cls_id] if cls_map else cls_id
        cls = f"{cls} {score:0.3}"
        cv2.putText(img, cls, (pt1[0], pt1[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    plt.imshow(img)
    plt.show()




if __name__ == '__main__':
    from matplotlib import image

    data = image.imread("./1.png")
    draw_point(data, (13, 13), 52)
