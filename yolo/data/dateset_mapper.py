# @Time    : 2023/3/21 上午10:26
# @Author  : Boyang
# @Site    : 
# @File    : dateset_mapper.py
# @Software: PyCharm
import copy
import logging
from collections import deque
from typing import List, Union, Optional

import numpy as np

from detectron2.config import CfgNode, configurable
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from .detection_utils import build_augmentation, read_image


class YOLOV3DatasetMapper:
    @configurable
    def __init__(
            self,
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            mosaic_transform: CfgNode,
            use_instance_mask: bool = False,
            instance_mask_format: str = "polygon",
            recompute_boxes: bool = False,
    ):
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"

        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        self.mosaic_transform = mosaic_transform
        if self.mosaic_transform.ENABLE:
            self.mosaic_pool = deque(maxlen=self.mosaic_transform.POOL_CAPACITY)

    @classmethod
    def from_config(cls, cfg, is_tran: bool = True):
        """
        Only for bounding box and instance mask. the instance mask is used to recompute boxes.
        not support keypoint.
        """

        augs = build_augmentation(cfg, is_tran)
        if cfg.INPUT.CROP.ENABLED and is_tran:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_tran,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "recompute_boxes": recompute_boxes
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        mosaic_flag = 0
        mosaic_samples = None
        if self.mosaic_transform.ENABLE and self.is_train:
            if len(self.mosaic_pool) > self.mosaic_transform.NUM_IMAGES:
                mosaic_flag = np.random.randint(2)
                if mosaic_flag == 1:
                    mosaic_samples = np.random.choice(self.mosaic_pool, self.mosaic_transform.NUM_IMAGES - 1)

            # add image info to mosaic pool
            self.mosaic_pool.append(copy.deepcopy(dataset_dict))

        image, annos = self._load_image_with_anns(dataset_dict)

        if self.is_train and mosaic_flag == 1 and mosaic_samples:
            mosaic_width = self.mosaic_transform.MOSAIC_WIDTH
            mosaic_height = self.mosaic_transform.MOAIC_HEIGHT
            out_image = np.zeros([mosaic_height, mosaic_width, 3], dtype=image.dtype)
            out_annos = []
            mosaic_border = (-mosaic_height // 2, -mosaic_width // 2)  # H,W
            mosaic_cy, mosaic_cx = [int(np.random.uniform(-x, 2 * s + x))
                                    for x, s in zip(mosaic_border, [mosaic_height, mosaic_width])]

            for m_idx in range(self.mosaic_transform.NUM_IMAGES):
                if m_idx != 0:
                    dataset_dict = copy.deepcopy(mosaic_samples[m_idx - 1])
                    image, annos = self._load_image_with_anns(dataset_dict)

                out_image, annos_i = self._blend_mosaic(
                    m_idx, image, annos, out_image, mosaic_cx, mosaic_cy, mosaic_width, mosaic_height
                )
                out_annos.append(annos_i)

        aug_input = T.AugInput(image)
        self.augmentations()

        if annos:
            annos = utils.annotations_to_instances(
                annos, image.shape[:2], mask_format=self.instance_mask_format
            )

        return dataset_dict

    def _blend_mosaic(self, idx, image, annos, out_image, xc, yc, mosaic_width, mosaic_height):
        h, w = image.shape[:2]
        if idx == 0:
            out_image.full_(114)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif idx == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, mosaic_width * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif idx == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(mosaic_height * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif idx == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, mosaic_width * 2), min(mosaic_height * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        out_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        if len(annos):

            ...

        return

    def _load_image_with_anns(self, dataset_dict):
        """
        Combined read image, ResizeShortestEdge, transform annotation.
        """
        image = read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        T.ResizeShortestEdge().get_transform(image)

        aug_input = T.AugInput(image=image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape

        # USER: Modify this if you want to keep them for some reason.
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return image, None

        # USER: Implement additional transformations if you have other types of data
        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

        else:
            annos = None

        return image, annos
