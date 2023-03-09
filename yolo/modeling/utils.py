# @Time    : 2023/2/27 下午2:25
# @Author  : Boyang
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import torch
import matplotlib.pyplot as plt
import cv2
from torch import Tensor
from detectron2.structures.boxes import Boxes
from typing import Optional, List
import numpy as np


def img_to_np(image):
    img = image.clone()
    img = img + torch.tensor([103.530, 116.280, 123.675])
    img = img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    img = np.ascontiguousarray(img)
    return img


def visualize_image(
        image: Tensor,
        boxes: [Optional[Boxes], Optional[List[Boxes]]],
        pixel_mean,
        pixel_std,
        show_original_image=False,
        separate_show=False

):
    img = img_to_np(image)
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
        boxes = Boxes(boxes).tensor
        for (x0, y0, x1, y1) in boxes:
            img = cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 0, 0),
                                thickness=1)
        plt.imshow(img[..., ::-1])
        plt.show()
    else:
        print(f"Unsupported type {type(boxes)}")
        pass


def draw_grid_lines(image, grid_size, stride=1):
    img = img_to_np(image) if isinstance(image, torch.Tensor) else image
    grid_height, grid_width = grid_size
    x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32)
    y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32)
    x, y = torch.meshgrid(x, y)
    x, y = x.reshape(-1), y.reshape(-1)
    point1 = torch.stack([x, y], dim=-1).numpy()
    point2 = torch.stack([x, y], dim=-1).numpy()

    for p1, p2 in zip(point1, point2):
        # plt.plot(p1, p2)
        ...

    plt.plot([0, 0], [0, 100])
    plt.plot([100, 100], [100, 100])
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    from matplotlib import image

    data = image.imread("/home/sparkai/PycharmProjects/YOLO/1.png")
    draw_grid_lines(data, (13, 13), 52)
