# @Time    : 2023/3/1 下午2:14
# @Author  : Boyang
# @Site    : 
# @File    : train_net.py
# @Software: PyCharm
import argparse

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

from tools.train_net import build_evaluator
import yolo


def setup(args):
    cfg = get_cfg()
    # cfg.merge_from_list(args)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)


def main():
    args = argparse.ArgumentParser().parse_args()
    args.config_file = "./configs/darknet53.yaml"
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
