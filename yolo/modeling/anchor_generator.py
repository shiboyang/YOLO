# @Time    : 2023/2/28 下午12:07
# @Author  : Boyang
# @Site    : 
# @File    : anchor_generator.py
# @Software: PyCharm
import torch
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, ANCHOR_GENERATOR_REGISTRY


@ANCHOR_GENERATOR_REGISTRY.register()
class YoloAnchorGenerator(DefaultAnchorGenerator):

    def generate_cell_anchors(self, sizes, aspect_ratios):
        anchors = []
        for (w, h) in sizes:
            x0, y0, x1, y1 = w / -2, h / -2, w / 2, h / 2
            anchors.append([x0, y0, x1, y1])

        return torch.tensor(anchors)
