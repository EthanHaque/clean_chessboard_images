import augmentor as ag
import os
import matplotlib.pyplot as plt
import numpy as np


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
    rows, cols = image.shape[0:2]

    distance_scale_factor = (np.random.rand() - 0.5) * strength
    translated = ag.translate(image, rows * distance_scale_factor, cols * distance_scale_factor)

    return translated


def random_rotation(image, strength=1, resize=True):
    angle = (2 * np.random.rand() - 1) * 360 * strength

    rotated = ag.rotate(image, angle, resize)

    return rotated


def random_blur(image, standard_deviation=3, strength=3):
    choice = np.random.rand()

    if choice >= 0.66:
        blurred_image = ag.average_blur(image, strength)
    elif choice >= 0.33:
        blurred_image = ag.median_blur(image, strength)
    else:
        blurred_image = ag.gaussian_blur(image, standard_deviation, strength)

    return blurred_image


def random_noise(image, standard_deviation=3, strength=0.1):
    choice = np.random.rand()

    if choice >= 0.5:
        noisy_image = ag.gaussian_noise(image, standard_deviation, strength)
    else:
        noisy_image = ag.speckle_noise(image, standard_deviation, strength * 5)

    return noisy_image


def main():
    pass


if __name__ == '__main__':
    paths = get_image_paths("./data/sub_set/train")
    image = random_noise(ag.load(paths[-10]))
    plt.imshow(image)
    plt.show()
