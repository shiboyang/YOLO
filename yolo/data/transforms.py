from typing import Tuple, Optional

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
        M = shear_M @ rotate_scale_M @ perspective_M

        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        height, width = img.shape[:2]
        center_M = np.eye(3)
        center_M[0, 2] = -width / 2
        center_M[1, 2] = -height / 2
        self.translate_M[0, 2] *= width
        self.translate_M[1, 2] *= height
        self.M = self.translate_M @ self.M @ center_M

        if self.is_inverse:
            self.M = np.matrix(self.M).I
        if any(self.perspective):
            ret = cv2.warpPerspective(img, self.M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            ret = cv2.warpAffine(img, self.M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        return ret

    def apply_coords(self, coords: np.ndarray):
        if any(self.perspective):
            coords = cv2.perspectiveTransform(coords[:, np.newaxis, :], self.M)
        else:
            coords = cv2.transform(coords[:, np.newaxis, :], self.M[:2])

        return coords

    def inverse(self) -> "Transform":
        return AffineTransform(
            angle=self.angle,
            scale=self.scale,
            translate=self.translate,
            shear=self.shear,
            perspective=self.perspective,
            is_inverse=True
        )


class BoxFilterTransform(Transform):
    """
    Box filter transform
    """

    def __init__(self, ksize: int, ddepth: int = -1, anchor=(-1, -1), normalize: bool = True,
                 border_type=cv2.BORDER_DEFAULT):
        super(BoxFilterTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        ret = cv2.boxFilter(img, self.ddepth, (self.ksize, self.ksize), anchor=self.anchor, normalize=self.normalize,
                            borderType=self.border_type)
        return ret

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        pass


class BlurTransform(Transform):
    def __init__(self, ksize: int, anchor: Optional[Tuple[int, int]] = None, board_type=cv2.BORDER_DEFAULT):
        super(BlurTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        ret = cv2.blur(img, (self.ksize, self.ksize), anchor=self.anchor, borderType=self.board_type)
        return ret

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        return BoxFilterTransform(self.ksize)


class GaussianBlurTransform(Transform):

    def __init__(self, ksize: int, sigma_x, sigma_y: int = 0, border_type=cv2.BORDER_DEFAULT):
        super(GaussianBlurTransform, self).__init__()
        kernel = cv2.getGaussianKernel(ksize, sigma_x)

        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        # cv2.GaussianBlur(img, (self.ksize, self.ksize), sigmaX=self.sigma_x, sigmaY=self.sigma_y,
        #                  borderType=self.border_type)
        return cv2.filter2D(img, ddepth=-1, kernel=self.kernel)

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        transform = GaussianBlurTransform(self.ksize, self.sigma_x)
        transform.kernel = np.matrix(self.kernel).I
        return transform
