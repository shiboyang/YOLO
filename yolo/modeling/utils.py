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


def visualize_image(image: Tensor, boxes: [Optional[Boxes], Optional[List[Boxes]]], pixel_mean, pixel_std):
    img = image.clone()
    img = img * pixel_std + pixel_mean
    img = img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    img = np.ascontiguousarray(img)

    plt.imshow(img[..., ::-1])
    plt.show()

    if isinstance(boxes, list):
        for boxes_i in boxes:
            boxes = boxes_i.tensor
            copy_img = img.copy()
            for (x0, y0, x1, y1) in boxes:
                copy_img = cv2.rectangle(copy_img, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 0, 0),
                                         thickness=1)

            plt.imshow(copy_img[..., ::-1])
            plt.show()
    else:
        boxes = boxes.tensor
        for (x0, y0, x1, y1) in boxes:
            img = cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 0, 0),
                                thickness=1)
        plt.imshow(img[..., ::-1])
        plt.show()


if __name__ == '__main__':
    image = cv2.imread("/home/sparkai/Pictures/2.png")
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).to("cuda")
    visualize_image(image, Boxes(Tensor([[-10, -10, 200, 200]])))
