# @Time    : 2023/2/24 下午5:00
# @Author  : Boyang
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model

import argparse
import cv2

import yolo
from detectron2.structures import Instances, Boxes


def setup(args):
    cfg = get_cfg()
    # cfg.merge_from_list(args)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


def main():
    args = argparse.ArgumentParser().parse_args()
    args.config_file = "./configs/darknet53.yaml"
    cfg = setup(args)

    model = build_model(cfg)
    print(model)

    img_path = r"./1.png"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (416, 416))
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)

    img = img.type(torch.float).to(cfg.MODEL.DEVICE)
    fields = {'gt_boxes': Boxes(torch.tensor([[0.0, 0.0, 25.0, 25.0]])),
              'gt_classes': torch.tensor([6])}
    batched_input = [{
        "image": img,
        "instances": Instances(image_size=(416, 416), **fields)
    }]
    model(batched_input)


if __name__ == '__main__':
    main()
