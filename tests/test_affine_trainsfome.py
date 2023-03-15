import cv2
import numpy as np

from detectron2.data.transforms import AugInput
from yolo.data.augmentation_impl import RandomAffine


def show(img: np.ndarray):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    aug = RandomAffine(
        angle=30
    )
    file = r"/home/shiby/Pictures/1.png"
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    aug_input = AugInput(img)
    transform = aug(aug_input)
    show(aug_input.image)
    inv_transform = transform.inverse()
    n_img = inv_transform.apply_image(aug_input.image)
    show(n_img)


if __name__ == '__main__':
    main()
