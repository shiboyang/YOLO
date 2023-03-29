# @Time    : 2023/3/28 下午2:08
# @Author  : Boyang
# @Site    : 
# @File    : test_mapper.py
# @Software: PyCharm
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader

from yolo.data.dataset_mapper import YOLOV3DatasetMapper
from yolo.utils.visualizer import draw_box


def setup():
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("configs/yolov3.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.DATASETS.TRAIN = ('coco_2017_val',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def show(image):
    cv2.namedWindow("Image")
    image = cv2.resize(image, (960, 960), cv2.INTER_LINEAR)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def invert_tensor(tensor):
    tensor = tensor.permute(1, 2, 0)
    return tensor.numpy()


def test_yolov3_dataset_mapper(cfg):
    mapper = YOLOV3DatasetMapper(cfg, True)
    dataloader = build_detection_train_loader(cfg, mapper=mapper, num_workers=0)
    for data_dict in dataloader:
        for d in data_dict:
            img = invert_tensor(d["image"])

            # draw_box(img, data_dict["boxes"])
            show(img)


def main():
    cfg = setup()
    test_yolov3_dataset_mapper(cfg)


if __name__ == "__main__":
    main()
