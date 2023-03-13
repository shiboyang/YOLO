# @Time    : 2023/3/10 下午3:46
# @Author  : Boyang
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import os
from typing import Dict

import cv2
import torch
from matplotlib import pyplot as plt


def diff(x):
    y = torch.load(r"../PyTorch-YOLOv3/test.pt")["x"]
    return (x - y).abs().sum()
