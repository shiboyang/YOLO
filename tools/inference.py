# @Time    : 2023/3/10 上午10:52
# @Author  : Boyang
# @Site    : 
# @File    : detect.py
# @Software: PyCharm
import torch
from torchvision import transforms

from detectron2.data.detection_utils import read_image
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.config import get_cfg

from yolo.checkpoint.checkpoint import YOLOV3Checkpointer


class Predictor:
    def __init__(self, cfg):
        self.model = build_model(cfg)
        self.model.eval()
        checkpointer = YOLOV3Checkpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        # self.transforms = transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])
        self.input_format = cfg.INPUT.FORMAT

        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # image = self.augmentation(original_image)
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = dict(height=height, width=width, image=image)
            predictions = self.model([inputs])[0]
            return predictions


def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    file = r"./datasets/samples/dog.jpg"
    image = read_image(file)
    cfg = setup(args)
    predictor = Predictor(cfg)
    pred = predictor(image)
    print(pred)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
