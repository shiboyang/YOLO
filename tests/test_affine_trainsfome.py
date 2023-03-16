from typing import List

import cv2
import numpy as np

from detectron2.data.transforms import AugInput
from detectron2.utils.visualizer import Visualizer
from yolo.data.augmentation_impl import RandomAffine, RandomBlur, RandomGaussianBlur
from yolo.utils import visualizer as utils_visualizer


def show(images: List[np.ndarray]):
    for i, img in enumerate(images):
        cv2.namedWindow(f"Image{i}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Image{i}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_rando_affine(aug_input):
    aug = RandomAffine(
        translate=0.1,
        degree=0,
        perspective=0.001,
        shear=0,
        scale=0
    )

    transform = aug(aug_input)
    vis = Visualizer(cv2.cvtColor(aug_input.image, cv2.COLOR_BGR2RGB))
    for box in aug_input.boxes:
        vis_output = vis.draw_box(box)
    show(vis_output.get_image()[..., ::-1])
    inv_transform = transform.inverse()
    n_img = inv_transform.apply_image(aug_input.image)
    boxes = inv_transform.apply_box(aug_input.boxes)
    n_img = utils_visualizer.draw_box(n_img, boxes)
    show(n_img)


def test_rando_blur(aug_input):
    original_img = aug_input.image.copy()
    aug = RandomBlur()
    transform = aug(aug_input)
    inv_transform = transform.inverse()
    inv_img = inv_transform.apply_image(aug_input.image.copy())
    show([original_img, aug_input.image, inv_img])


def test_gaussian_blur(aug_input):
    original_img = aug_input.image.copy()
    aug = RandomGaussianBlur()
    transform = aug(aug_input)
    inv_transform = transform.inverse()
    inv_img = inv_transform.apply_image(aug_input.image.copy())
    show([original_img, aug_input.image, inv_img])


def main():
    file = r"./datasets/samples/dog.jpg"
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    boxes = np.array([[114.2162, 110.7084, 578.8490, 438.1197],
                      [133.1878, 212.1925, 313.0800, 551.4429],
                      [462.2363, 85.1510, 683.7464, 166.9331],
                      [473.3467, 80.5246, 690.3759, 162.6522]])

    aug_input = AugInput(img, boxes=boxes)
    test_gaussian_blur(aug_input)


if __name__ == '__main__':
    main()
