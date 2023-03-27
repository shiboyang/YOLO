from typing import Tuple, Optional

import cv2
import numpy as np
from detectron2.data.transforms import Transform, NoOpTransform


class AffineTransform(Transform):
    """
    affine transform
    """

    def __init__(
            self,
            height: int,
            width: int,
            angle: int = 0,
            scale: float = 1.0,
            translate: Tuple[float, float] = (0.0, 0.0),
            shear: Tuple[float, float] = (0.0, 0.0),
            perspective: Tuple[float, float] = (0.0, 0.0)
    ):
        super(AffineTransform, self).__init__()
        translate_M = np.eye(3)
        translate_M[0, 2] = translate[0] * width
        translate_M[1, 2] = translate[1] * height
        shear_M = np.eye(3)
        shear_M[0, 1] = shear[0]
        shear_M[1, 0] = shear[1]
        rotate_scale_M = np.eye(3)
        rotate_scale_M[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=scale)
        perspective_M = np.eye(3)
        perspective_M[2, 0] = perspective[0]
        perspective_M[2, 1] = perspective[1]
        center_M = np.eye(3)
        center_M[0, 2] = -width / 2
        center_M[1, 2] = -height / 2

        M = translate_M @ shear_M @ rotate_scale_M @ perspective_M @ center_M

        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        height, width = img.shape[:2]
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

        coords = np.squeeze(coords, 1)
        coords[:, 0] = np.clip(coords[:, 0], 0, self.width)
        coords[:, 1] = np.clip(coords[:, 1], 0, self.height)

        return coords

    def inverse(self) -> "Transform":
        transform = AffineTransform(height=self.height, width=self.width, angle=self.angle, scale=self.scale,
                                    translate=self.translate, shear=self.shear, perspective=self.perspective)
        transform.M = np.matrix(transform.M).I
        return transform


class BoxFilterTransform(Transform):
    """
    Box filter transform
    """

    def __init__(self, ksize: int, ddepth: int = -1, anchor=(-1, -1), normalize: bool = True,
                 border_type=cv2.BORDER_DEFAULT):
        super(BoxFilterTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        ret = cv2.boxFilter(img, self.ddepth, self.ksize, anchor=self.anchor, normalize=self.normalize,
                            borderType=self.border_type)
        return ret

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        return NoOpTransform()


class BlurTransform(Transform):
    def __init__(self, ksize: int, anchor: Optional[Tuple[int, int]] = None, board_type=cv2.BORDER_DEFAULT):
        super(BlurTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        return cv2.blur(img, self.ksize, anchor=self.anchor, borderType=self.board_type)

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        return NoOpTransform()


class GaussianBlurTransform(Transform):

    def __init__(self, ksize: Tuple[int, int] = (3, 3), sigma_x: float = 1.4, sigma_y=None,
                 border_type=cv2.BORDER_DEFAULT):
        super(GaussianBlurTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        return cv2.GaussianBlur(img, self.ksize, sigmaX=self.sigma_x, sigmaY=self.sigma_y,
                                borderType=self.border_type)

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        return NoOpTransform()


class MedianBlurTransform(Transform):
    def __init__(self, ksize):
        super(MedianBlurTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        return cv2.medianBlur(img, ksize=self.ksize)

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        return NoOpTransform()


class BilateralFilterTransform(Transform):
    def __init__(self, sigma_color, sigma_space, diameter):
        super(BilateralFilterTransform, self).__init__()
        self.diameter = 5
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        return cv2.bilateralFilter(img, self.diameter, self.sigma_color, self.sigma_space)

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        return NoOpTransform()


class PixelDropoutTransform(Transform):
    def __init__(self, drop_mask, drop_prob, drop_value: int = 0):
        super(PixelDropoutTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        return np.where(self.drop_mask, self.drop_value, img)

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        return NoOpTransform()


class ColorJitterTransform(Transform):
    def __init__(self, hue, saturation, exposure, cvt_format: Tuple[int, int] = (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR)):
        super(ColorJitterTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        img = cv2.cvtColor(img, self.cvt_format[0])
        hue, saturation, exposure = cv2.split(img)
        x = np.arange(0, 256)
        lut_hue = (x * self.hue % 180).astype(img.dtype)
        lut_saturation = np.clip(x * self.saturation, 0, 255).astype(img.dtype)
        lut_exposure = np.clip(x * self.exposure, 0, 255).astype(img.dtype)
        img = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(saturation, lut_saturation), cv2.LUT(exposure, lut_exposure)))
        img = cv2.cvtColor(img, self.cvt_format[1])
        return img

    def apply_coords(self, coords: np.ndarray):
        return coords

    def inverse(self) -> "Transform":
        return NoOpTransform()
