# @Time    : 2023/3/10 上午10:52
# @Author  : Boyang
# @Site    : 
# @File    : detect.py
# @Software: PyCharm
from typing import List

import torch
from torchvision import transforms

from detectron2.data.detection_utils import read_image
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.config import get_cfg

from yolo.checkpoint.checkpoint import YOLOV3Checkpointer
from yolo.modeling.utils import visualize_predictions, load_classes

from pytorchyolo.detect import _create_data_loader


class Predictor:
    def __init__(self, cfg):
        self.model = build_model(cfg)
        self.model.eval()
        checkpointer = YOLOV3Checkpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        # self.transforms = transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])
        self.input_format = cfg.INPUT.FORMAT

        assert self.input_format in ["RGB", "BGR"], self.input_format

    @torch.no_grad()
    def __call__(self, original_image):
        if self.input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        # image = self.augmentation(original_image)
        image = original_image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = dict(height=height, width=width, image=image)
        predictions = self.model([inputs])
        return predictions

    @torch.no_grad()
    def detect_batched_image(self, batched_inputs: torch.Tensor, classes: List[str]):
        height, width = batched_inputs.shape[-2:]
        batched_inputs = [dict(height=height, width=width, image=image) for image in batched_inputs]
        predictions = self.model(batched_inputs)

        for img_dict, prediction in zip(batched_inputs, predictions):
            img = img_dict["image"]
            instances = prediction["instances"]
            visualize_predictions(img, boxes=instances.pred_boxes.tensor, classes=instances.pred_classes,
                                  scores=instances.scores, cls_map=classes)

        return predictions


def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    img_path = r"./datasets/samples/"
    batch_size = 1
    img_size = 416
    n_cpu = 0
    data_loader = _create_data_loader(img_path, batch_size, img_size, n_cpu)
    classes_name = load_classes("./datasets/coco.names")

    cfg = setup(args)
    predictor = Predictor(cfg)

    for img_paths, images in data_loader:
        pred = predictor.detect_batched_image(images, classes_name)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
