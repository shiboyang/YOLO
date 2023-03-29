from typing import List, Optional, Union

import cv2
import numpy as np

from detectron2.data.transforms import AugInput
from detectron2.utils.visualizer import Visualizer
from yolo.data.augmentation_impl import RandomAffine, RandomBlur, RandomGaussianBlur, RandomPixelDropout, \
    RandomBilateralFilter, RandomMedianBlur, RandomColorJitter, LetterBox
from yolo.utils import visualizer as utils_visualizer


def show(image: Union[List[np.ndarray], np.ndarray]):
    if isinstance(image, list):
        for i, img in enumerate(image):
            cv2.namedWindow(f"Image{i}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Image{i}", img)
    else:
        cv2.namedWindow(f"Image", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_rando_affine(aug_input):
    aug = RandomAffine(
        translate=0.1,
        degree=10,
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


def test_random_blur(aug_input):
    original_img = aug_input.image.copy()
    aug = RandomBlur()
    transform = aug(aug_input)
    inv_transform = transform.inverse()
    inv_img = inv_transform.apply_image(aug_input.image.copy())
    show([original_img, aug_input.image, inv_img])


def test_gaussian_blur(aug_input, mean=0, sigma=100 ** 0.5):
    original_img = aug_input.image.copy()
    # gaussian noise
    gaussian_noise = np.random.normal(mean, sigma, original_img.shape)
    noised_img = original_img + gaussian_noise
    cv2.normalize(noised_img, noised_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noised_img = noised_img.astype(np.uint8)

    aug = RandomGaussianBlur([3], sigma)
    aug(aug_input)

    new_img = cv2.fastNlMeansDenoisingColored(noised_img.copy(), None, 10, 10, 7, 21)

    show([original_img, noised_img, aug_input.image, new_img])


def test_random_pixel_dropout(aug_input):
    original_img = aug_input.image.copy()
    aug = RandomPixelDropout(drop_value=255, dropout_prob=0.05)
    aug(aug_input)
    pixel_dropout_img = aug_input.image.copy()

    images = []
    for ksize in [3, 5, 7, 9, 11, 31]:
        box_filter = RandomBlur([ksize])
        aug_input = AugInput(image=pixel_dropout_img)
        box_filter(aug_input)
        transformed_img2 = aug_input.image.copy()

        gaussian_filter = RandomGaussianBlur([ksize])
        aug_input = AugInput(image=pixel_dropout_img)
        gaussian_filter(aug_input)
        transformed_img3 = aug_input.image.copy()

        bilateral_filter = RandomBilateralFilter(ksize * 1000, ksize * 100, ksize)
        aug_input = AugInput(image=pixel_dropout_img)
        bilateral_filter(aug_input)
        transformed_img4 = aug_input.image.copy()

        median_blur = RandomMedianBlur([ksize])
        aug_input = AugInput(image=pixel_dropout_img)
        median_blur(aug_input)
        transformed_img5 = aug_input.image.copy()

        image = np.hstack(
            [original_img, pixel_dropout_img, transformed_img2, transformed_img3, transformed_img4, transformed_img5])
        images.append(image)

    show(np.vstack(images))


def test_distortion(aug_input):
    original_img = aug_input.image.copy()
    distortion = RandomColorJitter("BGR", 0.09, 0.09, 0.01)
    distortion(aug_input)
    image = np.hstack([original_img, aug_input.image])
    show(image)


def test_letterbox(aug_input):
    original_img = aug_input.image.copy()
    letterbox = LetterBox(608, 640)
    letterbox(aug_input)
    # image = np.hstack([original_img, aug_input.image])
    image = utils_visualizer.draw_box(aug_input.image, aug_input.boxes)
    show(image)


def main():
    file = r"./datasets/samples/dog.jpg"
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    boxes = np.array([[114.2162, 110.7084, 578.8490, 438.1197],
                      [133.1878, 212.1925, 313.0800, 551.4429],
                      [462.2363, 85.1510, 683.7464, 166.9331],
                      [473.3467, 80.5246, 690.3759, 162.6522]])

    aug_input = AugInput(image=img, boxes=boxes)
    test_letterbox(aug_input)


if __name__ == '__main__':
    main()
