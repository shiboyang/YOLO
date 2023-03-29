import os
import cv2

import detectron2.data.transforms as T
from .augmentation_impl import RandomAffine, RandomColorJitter, LetterBox


def read_image(file_name, format=None):
    assert os.path.exists(file_name), FileNotFoundError(f"{file_name}")
    if format:
        assert format in ["BGR", "RGB"], ValueError("Only support `BGR` or `RGB` format")
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if format and format == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def build_augmentation(cfg, is_train: bool = True):
    if cfg.INPUT.MOSAIC.ENABLE:
        augmentation = build_yolov3_augmentation(cfg, is_train)
    else:
        augmentation = build_normal_augmentation(cfg, is_train)

    return augmentation


def build_normal_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.

    is_train: letterbox(resize_shortest_edge) - random_perspective - random_color_jitter - random_flip.
    is_valid: resize_shortest_edge.

    Returns:
        list[Augmentation]
    """
    augmentation = []
    # letterbox
    if is_train and cfg.INPUT.LETTERBOX.ENABLE:
        augmentation.append(
            LetterBox(
                new_height=cfg.INPUT.LETTERBOX.HEIGHT,
                new_width=cfg.INPUT.LETTERBOX.WIDTH
            )
        )
    else:
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    # random_perspective
    if is_train and cfg.INPUT.RANDOM_AFFIEN.ENABLE:
        augmentation.append(
            RandomAffine(
                translate=cfg.INPUT.RANDOM_AFFIEN.TRANSLATE,
                scale=cfg.INPUT.RANDOM_AFFIEN.SCALE,
                degree=cfg.INPUT.RANDOM_AFFIEN.DEGREE,
                shear=cfg.INPUT.RANDOM_AFFIEN.SHEAR,
                perspective=cfg.INPUT.RANDOM_AFFIEN.PERSPECTIVE
            )
        )
    # hsv aug
    if is_train and cfg.INPUT.COLOR_JITTER.ENABLE:
        augmentation.append(
            RandomColorJitter(
                image_format=cfg.INPUT.FORMAT,
                hue_gain=cfg.INPUT.COLOR_JITTER.HUE,
                saturation_gain=cfg.INPUT.COLOR_JITTER.SATURATION,
                exposure_gain=cfg.INPUT.COLOR_JITTER.EXPOSURE
            )
        )
    # flip up-down
    # flip left-right
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation


def build_yolov3_augmentation(cfg, is_train):
    """
    yolov3 augmentation
    is_train: resize_shortest_edge - random_affine - color_jitter - random_flip.
    is_valid: resize_shortest_edge.
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

    # random perspective
    if is_train and cfg.INPUT.RANDOM_AFFINE.ENABLE:
        augmentation.append(
            RandomAffine(
                translate=cfg.INPUT.RANDOM_AFFINE.TRANSLATE,
                scale=cfg.INPUT.RANDOM_AFFINE.SCALE,
                degree=cfg.INPUT.RANDOM_AFFINE.DEGREE,
                shear=cfg.INPUT.RANDOM_AFFINE.SHEAR,
                perspective=cfg.INPUT.RANDOM_AFFINE.PERSPECTIVE
            )
        )
    # hsv aug
    if is_train and cfg.INPUT.COLOR_JITTER.ENABLE:
        augmentation.append(
            RandomColorJitter(
                image_format=cfg.INPUT.FORMAT,
                hue_gain=cfg.INPUT.COLOR_JITTER.HUE,
                saturation_gain=cfg.INPUT.COLOR_JITTER.SATURATION,
                exposure_gain=cfg.INPUT.COLOR_JITTER.EXPOSURE
            )
        )
    # flip up-down
    # flip left-right
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    return augmentation
