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


def visualize_image(
        image: Tensor,
        boxes: [Optional[Boxes], Optional[List[Boxes]]],
        pixel_mean,
        pixel_std,
        show_original_image=False,
        separate_show=False

):
    img = image.clone()
    img = img * pixel_std + pixel_mean
    img = img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    img = np.ascontiguousarray(img)
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


if __name__ == '__main__':
    image = cv2.imread("/home/sparkai/Pictures/2.png")
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).to("cuda")
    visualize_image(image, Boxes(Tensor([[-10, -10, 200, 200]])))
