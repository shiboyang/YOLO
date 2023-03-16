# @Time    : 2023/3/16 下午12:57
# @Author  : Boyang
# @Site    : 
# @File    : visualizer.py
# @Software: PyCharm
from typing import Iterator, Union, Tuple

import cv2
import numpy as np


def draw_box(img: np.ndarray, boxes: Iterator[Union[Tuple[float, float, float, float], Tuple[int, int, int, int]]]):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
    return img
