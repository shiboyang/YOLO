# @Time    : 2023/3/10 上午10:52
# @Author  : Boyang
# @Site    : 
# @File    : detect.py
# @Software: PyCharm
import argparse
import glob
import os

import cv2
import torch
import tqdm

import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer, ColorMode
from yolo.checkpoint.checkpoint import YOLOV3Checkpointer


class Predictor:
    def __init__(self, cfg):
        self.model = build_model(cfg)
        self.model.eval()
        checkpointer = YOLOV3Checkpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT

        assert self.input_format in ["RGB", "BGR"], self.input_format

    @torch.no_grad()
    def __call__(self, original_image):
        if self.input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = dict(height=height, width=width, image=image)
        prediction = self.model([inputs])[0]
        return prediction


class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        self.predictor = Predictor(cfg)
        self.image_format = cfg.INPUT.FORMAT
        self.instance_mode = instance_mode
        self.cpu_device = torch.device("cpu")
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        assert self.image_format in ["BGR", "RGB"], self.image_format

    def run_on_image(self, image):
        """resume_or_load
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="YOLO V3 Inference")
    parser.add_argument("--config-file", metavar="FILE", default="configs/yolov3_inference.yaml")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", default="./output")
    parser.add_argument("--score-confs", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.4)
    return parser


def main():
    args = get_parser().parse_args()
    cfg = setup(args)
    demo = VisualizationDemo(cfg)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        # 默认的cv的读入图片格式为BGR
        img = read_image(path, format="BGR")
        predictions, visualized_output = demo.run_on_image(image=img)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", visualized_output.get_image()[:, :, ::-1])
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
