import random
from typing import List, Optional

import numpy as np
from fvcore.transforms import Transform

from detectron2.data.transforms import Augmentation
from .transforms import (
    AffineTransform,
    BlurTransform,
    GaussianBlurTransform,
    PixelDropoutTransform,
    BilateralFilterTransform,
    MedianBlurTransform
)


class RandomAffine(Augmentation):
    """
    It will execute perspective transform, rotate transform, shear transform, translate transform
    in order on image center.
    """

    def __init__(self, translate: float = 0.1, scale: float = 0.1, degree: int = 0, shear: float = 10.0,
                 perspective: float = 0.0):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        height, width = image.shape[:2]
        angle = np.random.uniform(-self.degree, self.degree)
        scale = np.random.uniform(1 - self.scale, 1 + self.scale)
        translate_x = np.random.uniform(0.5 - self.translate, 0.5 + self.translate)
        translate_y = np.random.uniform(0.5 - self.translate, 0.5 + self.translate)
        shear_x = np.tan(np.random.uniform(-self.shear, self.shear) * np.pi / 180)
        shear_y = np.tan(np.random.uniform(-self.shear, self.shear) * np.pi / 180)

        perspective_x = random.uniform(-self.perspective, self.perspective)
        perspective_y = random.uniform(-self.perspective, self.perspective)

        return AffineTransform(height=height, width=width, angle=angle, scale=scale,
                               translate=(translate_x, translate_y), shear=(shear_x, shear_y),
                               perspective=(perspective_x, perspective_y))


class RandomBlur(Augmentation):
    """
    Random blur using box filter
    """

    def __init__(self, kernel_sizes: Optional[List[int]] = None):
        super(RandomBlur, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        self._init(locals())

    def get_transform(self, *args) -> Transform:
        ksize = random.choice(self.kernel_sizes)
        ksize = (ksize, ksize)
        return BlurTransform(ksize)


class RandomGaussianBlur(Augmentation):
    def __init__(self, kernel_sizes=None):
        super(RandomGaussianBlur, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [9]
        sigma_x = 0
        self._init(locals())

    def get_transform(self, *args) -> Transform:
        ksize = random.choice(self.kernel_sizes)
        ksize = (ksize, ksize)

        return GaussianBlurTransform(ksize, self.sigma_x)


class RandomBilateralFilter(Augmentation):
    def __init__(self, sigma_color: float = 10.0, sigma_space: float = 2.5, diameter: int = 5):
        super(RandomBilateralFilter, self).__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        diameter = random.randint(1, self.diameter)
        sigma_color = 1000
        sigma_space = diameter / 2
        return BilateralFilterTransform(sigma_color, sigma_space, diameter)


class RandomMedianBlur(Augmentation):
    def __init__(self, ksizes: Optional[List[int]] = None):
        super(RandomMedianBlur, self).__init__()
        if ksizes is None:
            ksizes = [i for i in range(1, 31, 2)]
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        ksize = random.choice(self.ksizes)
        return MedianBlurTransform(ksize)


class RandomPixelDropout(Augmentation):
    def __init__(self, dropout_prob=0.01, per_channel: bool = False, drop_value: int = 0):
        super(RandomPixelDropout, self).__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        shape = image.shape if self.per_channel else image.shape[:2]
        rnd = np.random.RandomState(random.randint(0, 1 << 31))
        drop_mask = rnd.choice([True, False], shape, p=(self.dropout_prob, 1 - self.dropout_prob))
        if drop_mask.ndim != image.ndim:
            drop_mask = np.expand_dims(drop_mask, -1)
        if isinstance(drop_mask, (int, float)) and self.drop_value == 0:
            drop_value = np.zeros_like(image)
        else:
            drop_value = np.full_like(image, self.drop_value)

        return PixelDropoutTransform(drop_mask, self.dropout_prob, drop_value)
