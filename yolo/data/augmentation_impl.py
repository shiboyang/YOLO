import random

import numpy as np
from fvcore.transforms import Transform

from detectron2.data.transforms import Augmentation
from .transforms import AffineTransform


class RandomAffine(Augmentation):
    def __init__(self, translate: float = 0.1, scale: float = 0.0, angle: int = 0, shear: float = 0.0,
                 perspective: float = 0.0):
        super().__init__()
        self._init(locals())

    def get_transform(self, image) -> Transform:
        angle = np.random.uniform(-self.angle, self.angle)
        scale = np.random.uniform(1 - self.scale, 1 + self.scale)
        translate_x = np.random.uniform(0.5 - self.translate, 0.5 + self.translate)
        translate_y = np.random.uniform(0.5 - self.translate, 0.5 + self.translate)
        shear_x = np.tan(np.random.uniform(-self.shear, self.shear) * np.pi / 180)
        shear_y = np.tan(np.random.uniform(-self.shear, self.shear) * np.pi / 180)

        perspective_x = random.uniform(-self.perspective, self.perspective)
        perspective_y = random.uniform(-self.perspective, self.perspective)

        return AffineTransform(angle=angle, scale=scale, translate=(translate_x, translate_y), shear=(shear_x, shear_y),
                               perspective=(perspective_x, perspective_y))
