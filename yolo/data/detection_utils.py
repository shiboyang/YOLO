import os
import cv2

import detectron2.data.transforms as T


def read_image(file_name, format=None):
    assert os.path.exists(file_name), FileNotFoundError(f"{file_name}")
    if format:
        assert format in ["BGR", "RGB"], ValueError("Only support `BGR` or `RGB` format")
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if format and format == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def build_augmentation(cfg, is_train: bool = True):
    if cfg.INPUT.MOSACAL.ENABLE:
        augmentation = build_yolov3_augmentation(cfg, is_train)

    else:
        augmentation = build_normal_augmentation(cfg, is_train)

    return augmentation


def build_normal_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation


def build_yolov3_augmentation(cfg, is_train):
    return []
