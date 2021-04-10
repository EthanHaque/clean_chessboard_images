import augmentor
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
    x_scale_factors = augmentor.get_rand_points(low=0, high=strength, shape=[4])
    y_scale_factors = augmentor.get_rand_points(low=0, high=strength, shape=[4])

    if choice >= 0.66:
        warped_image = augmentor.warp_corners(image, x_scale_factors, y_scale_factors)
    elif choice >= 0.33:
        warped_image = augmentor.horizontal_squeeze(image, x_scale_factors)
    else:
        warped_image = augmentor.vertical_squeeze(image, y_scale_factors)

    return warped_image


def random_translation(image, strength=0.1):
    rows, cols = image.shape[0:2]

    distance_scale_factor = (np.random.rand() - 0.5) * strength
    translated = augmentor.translate(image, rows * distance_scale_factor, cols * distance_scale_factor)

    return translated


def main():
    pass


if __name__ == '__main__':
    paths = get_image_paths("./data/sub_set/train")
    image = random_translation(augmentor.load(paths[-10]))
    plt.imshow(image)
    plt.show()
