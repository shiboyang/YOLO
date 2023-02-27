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
    img = img.permute(2, 0, 1).unsqueeze(0)

    img = img.type(torch.float).to(cfg.MODEL.DEVICE)

    model(img)


if __name__ == '__main__':
    main()
