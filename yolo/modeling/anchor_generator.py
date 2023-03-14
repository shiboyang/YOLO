# @Time    : 2023/2/28 下午12:07
# @Author  : Boyang
# @Site    : 
# @File    : anchor_generator.py
# @Software: PyCharm
import math
from typing import List

import torch

from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, ANCHOR_GENERATOR_REGISTRY


@ANCHOR_GENERATOR_REGISTRY.register()
class YoloAnchorGenerator(DefaultAnchorGenerator):

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        anchors = cfg.MODEL.ANCHOR_GENERATOR.ANCHORS
        sizes = [[(w * h) ** (1.0 / 2) for w, h in item] for item in anchors]
        aspect_ratios = [[h / w for w, h in item] for item in anchors]

        return {
            "sizes": sizes,
            "aspect_ratios": aspect_ratios,
            "strides": [x.stride for x in input_shape],
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
        }

    def generate_cell_anchors(self, sizes, aspect_ratios):
        anchors = []
        for size, ratio in zip(sizes, aspect_ratios):
            area = size ** 2
            w = math.sqrt(area / ratio)
            h = ratio * w
            x0, y0, x1, y1 = (w / -2, h / -2, w / 2, h / 2)
            anchors.append([x0, y0, x1, y1])

        return torch.tensor(anchors)
