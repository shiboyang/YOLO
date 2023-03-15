from typing import Tuple

import numpy as np
import cv2

from detectron2.data.transforms import Transform


class AffineTransform(Transform):
    def __init__(
            self,
            angle: int = 0,
            scale: float = 1.0,
            translate: Tuple[float, float] = (0.0, 0.0),
            shear: Tuple[float, float] = (0.0, 0.0),
            perspective: Tuple[float, float] = (0.0, 0.0),
            is_inverse: bool = False
    ):
        super(AffineTransform, self).__init__()
        translate_M = np.eye(3)
        translate_M[0, 2] = translate[0]
        translate_M[1, 2] = translate[1]
        shear_M = np.eye(3)
        shear_M[0, 1] = shear[0]
        shear_M[1, 0] = shear[1]
        rotate_scale_M = np.eye(3)
        rotate_scale_M[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=scale)
        perspective_M = np.eye(3)
        perspective_M[2, 0] = perspective[0]
        perspective_M[2, 1] = perspective[1]

        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        height, width = img.shape[:2]
        center_M = np.eye(3)
        center_M[0, 2] = -width / 2
        center_M[1, 2] = -height / 2
        self.translate_M[0, 2] *= width
        self.translate_M[1, 2] *= height
        M = self.translate_M @ self.shear_M @ self.rotate_scale_M @ self.perspective_M @ center_M

        if self.is_inverse:
            M = M.T
        if any(self.perspective):
            ret = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            ret = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        return ret

    def apply_coords(self, coords: np.ndarray):
        pass

    def inverse(self) -> "Transform":
        return AffineTransform(
            angle=self.angle,
            scale=self.scale,
            translate=self.translate,
            shear=self.shear,
            perspective=self.perspective,
            is_inverse=True
        )
