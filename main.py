# @Time    : 2023/2/24 下午5:00
# @Author  : Boyang
# @Site    : 
# @File    : main.py
# @Software: PyCharm
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import argparse


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    return cfg


def main():
    args = argparse.ArgumentParser().parse_args()
    args.config_file = "./configs/darknet53.yaml"
    cfg = setup(args)

    model = build_model(cfg)


if __name__ == '__main__':
    main()
