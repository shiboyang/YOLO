# @Time    : 2023/3/21 上午10:26
# @Author  : Boyang
# @Site    : 
# @File    : dateset_mapper.py
# @Software: PyCharm
from typing import List, Union, Optional

import numpy as np

from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.transforms as T
from detectron2.config import configurable
import detection_utils as utils


class YOLOV3DatasetMapper:
    @configurable
    def __init__(
            self,
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            use_instance_mask: bool = False,
            use_keypoint: bool = False,
            instance_mask_format: str = "polygon",
            keypoint_hflip_indices: Optional[np.ndarray] = None,
            precomputed_proposal_topk: Optional[int] = None,
            recompute_boxes: bool = False
    ):
        ...

    @classmethod
    def from_config(cls, cfg, is_tran: bool = True):
        aug = utils.build_augmentation(cfg, is_tran)
