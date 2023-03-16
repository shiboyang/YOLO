import random
from typing import List, Optional

import numpy as np
import torch.nn
from fvcore.transforms import Transform

from detectron2.data.transforms import Augmentation
from .transforms import AffineTransform, BlurTransform, GaussianBlurTransform


class RandomAffine(Augmentation):
    def __init__(self, translate: float = 0.1, scale: float = 0.1, degree: int = 0, shear: float = 10.0,
                 perspective: float = 0.0):
        super().__init__()
        self._init(locals())

    def get_transform(self, image) -> Transform:
        angle = np.random.uniform(-self.degree, self.degree)
        scale = np.random.uniform(1 - self.scale, 1 + self.scale)
        translate_x = np.random.uniform(0.5 - self.translate, 0.5 + self.translate)
        translate_y = np.random.uniform(0.5 - self.translate, 0.5 + self.translate)
        shear_x = np.tan(np.random.uniform(-self.shear, self.shear) * np.pi / 180)
        shear_y = np.tan(np.random.uniform(-self.shear, self.shear) * np.pi / 180)

        perspective_x = random.uniform(-self.perspective, self.perspective)
        perspective_y = random.uniform(-self.perspective, self.perspective)

        return AffineTransform(angle=angle, scale=scale, translate=(translate_x, translate_y), shear=(shear_x, shear_y),
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
        return BlurTransform(ksize)


class RandomGaussianBlur(Augmentation):
    def __init__(self, kernel_sizes=None):
        super(RandomGaussianBlur, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [9]
        sigma_x = 1000
        self._init(locals())

    def get_transform(self, *args) -> Transform:
        ksize = random.choice(self.kernel_sizes)

        return GaussianBlurTransform(ksize, self.sigma_x)
