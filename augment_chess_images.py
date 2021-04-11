import augmentor as ag
import os
import numpy as np
from tqdm import tqdm
import cv2


def get_image_paths(directory_path, extensions=None):
    """
    Gets the absolute path of all files in a directory.

    :param directory_path: Path to a directory.
    :param extensions: The allowed extensions to get the path to.
    :return: List of strings of absolute paths to images.
    """
    directory_contents = os.listdir(directory_path)
    paths = [os.path.abspath(directory_path + "/" + file) for file in directory_contents]

    return paths


def random_corner_warp(image, strength=0.2):
    """
    Randomly warps the corners of an image.

    :param image: The image to warp.
    :param strength: The amount of distortion.
    :return: The image with corners warped.
    """
    choice = np.random.rand()
    x_scale_factors = ag.get_rand_points(low=0, high=strength, shape=[4])
    y_scale_factors = ag.get_rand_points(low=0, high=strength, shape=[4])

    if choice >= 0.66:
        warped_image = ag.warp_corners(image, x_scale_factors, y_scale_factors)
    elif choice >= 0.33:
        warped_image = ag.horizontal_squeeze(image, x_scale_factors)
    else:
        warped_image = ag.vertical_squeeze(image, y_scale_factors)

    return warped_image


def random_translation(image, strength=0.1):
    """
    Randomly shifts an image.

    :param image: The image to translate.
    :param strength: The amount of distortion.
    :return: The image with translations applied.
    """
    rows, cols = image.shape[0:2]

    distance_scale_factor = (np.random.rand() - 0.5) * strength
    translated = ag.translate(image, rows * distance_scale_factor, cols * distance_scale_factor)

    return translated


def random_rotation(image, strength=1, resize=True):
    """
    Rotates an image around the center.

    :param image: The image to rotate
    :param strength: The amount of distortion
    :param resize: resize the output image to not crop the image.
    :return: The rotated image.
    """
    angle = (2 * np.random.rand() - 1) * 360 * strength

    rotated = ag.rotate(image, angle, resize)

    return rotated


def random_blur(image, standard_deviation=3, strength=3):
    """
    Randomly apply a blur to an image.

    :param image: The image to blur
    :param standard_deviation: How many standard deviations from the mean a gaussian blur should use.
    :param strength: The amount of blur.
    :return: The blurred image.
    """
    choice = np.random.rand()

    if choice >= 0.66:
        blurred_image = ag.average_blur(image, strength)
    elif choice >= 0.33:
        blurred_image = ag.median_blur(image, strength)
    else:
        blurred_image = ag.gaussian_blur(image, standard_deviation, strength)

    return blurred_image


def random_noise(image, standard_deviation=3, strength=0.1):
    """
    Adds random noise to an image.

    :param image: The image to add noise to.
    :param standard_deviation: How many standard deviations from the mean the noise function should use.
    :param strength: The amount of noise.
    :return: The noisy image.
    """
    choice = np.random.rand()

    if choice >= 0.5:
        noisy_image = ag.gaussian_noise(image, standard_deviation, strength)
    else:
        noisy_image = ag.speckle_noise(image, standard_deviation, strength * 5)

    return noisy_image


def random_distortion(image, standard_deviation=5, strength=0.4):
    """
    Randomly distorts an image with either 1D or 2D distortions.

    :param image: The image to distort.
    :param standard_deviation: How many standard deviations from the mean the warping functions should use.
    :param strength: The amount of distortion.
    :return: The distorted image.
    """
    choice = np.random.rand()

    strength *= 10
    if choice >= 0.5:
        distorted_image = ag.gaussian_noise_distortion_1d(image, standard_deviation, strength)
    else:
        distorted_image = ag.gaussian_noise_distortion_2d(image, standard_deviation, strength)

    return distorted_image


def add_random_transformations(image, number_transforms, functions):
    """
    Randomly applies a number of transformations to an image.

    :param image: The image to transform.
    :param number_transforms: The number of transformations to apply.
    :param functions: The functions to randomly apply from.
    :return: The image with random transformations applied.
    """
    choices = np.random.choice(functions, size=number_transforms, replace=False)  # chooses random functions to use
    for choice in choices:
        image = choice(image)

    return image


def resize_to_match_dimensions(base_image, new_image):
    rows, cols = base_image.shape[0:2]

    resized_image = cv2.resize(new_image, (cols, rows), interpolation=cv2.INTER_CUBIC)

    return resized_image


def concatenate_images(left_image, right_image):
    return cv2.hconcat([left_image, right_image])


def main():
    paths = get_image_paths("./data/sub_set/train")
    save_to = "./data/augmented/train"
    functions = [
        random_corner_warp,
        random_translation,
        random_rotation,
        random_blur,
        random_noise,
        random_distortion
    ]

    for image_path in tqdm(paths):
        name = os.path.basename(image_path)
        image = ag.load(image_path)
        transformed_image = add_random_transformations(image, 2, functions)
        transformed_image = resize_to_match_dimensions(image, transformed_image)
        concatenated_images = concatenate_images(image, transformed_image)
        ag.save_image(concatenated_images, save_to + "./" + name)


if __name__ == '__main__':
    main()
