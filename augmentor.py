import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.ndimage as ndi


def load(image_file, color=cv2.IMREAD_COLOR):
    """
    Loads an image from a given filepath.

    :param image_file: String of path to an image file.
    :param color: Can be one of three values:

        cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of image will be
        neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.

        cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode. Alternatively,
        we can pass integer value 0 for this flag.

        cv2.IMREAD_UNCHANGED: It specifies to load an image as such including alpha channel.
        Alternatively, we can pass integer value -1 for this flag.

    :return: An image object.
    """
    image = cv2.imread(image_file, color)

    return image


def _get_corners(input_image):
    """
    Returns the locations of the corners of an image as a numpy float32 array with dimensions (4,2).

    :param input_image: The image to find the corners of.
    :return: A numpy float32 array with dimensions (4,2).
    """
    rows, cols = input_image.shape[0:2]

    return np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])


def _get_rand_points(low=0.0, high=1.0, shape=None):
    """
    Creates a numpy array of random floats in a range.

    :param low: The lower bound to generate numbers over.
    :param high: The upper bound to generate numbers over.
    :param shape: List or Tuple with the desired shape for the array of random numbers.
    :return: A numpy array of random numbers over [low, high) in the given shape.
    """
    if shape is None:
        shape = [4]
    return (high - low) * np.random.rand(*shape) + low


def perspective_transformation(input_image, matrix):
    """
    Applies a perspective transformation to a given image.

    :param input_image: The image to apply a perspective transformation to.
    :param matrix: 3x3 matrix that defines the transformation.
    :return: The image with the perspective transformation applied
    """
    rows, cols = input_image.shape[0:2]
    transformed_image = cv2.warpPerspective(input_image, matrix, (cols, rows))

    return transformed_image


def warp_corners(input_image, x_scale_factors, y_scale_factors):
    """
    Applies a perspective transformation to a given image which results in the corners being shifted some scale factor.

        The scale factors are applied to the respective x and y component of the coordinate pair in the order:
            1. top left
            2. top right
            3. bottom left
            4. bottom right

    :param input_image: The image to scale the corners of.
    :param x_scale_factors: Array with shape (4,) with the scale factors to apply to each corner's x-component.
    :param y_scale_factors: Array with shape (4,) with the scale factors to apply to each corner's y-component.
    :return: The image with the scaling applied.
    """
    rows, cols = input_image.shape[0:2]
    src_points = _get_corners(input_image)

    p1x = int(x_scale_factors[0] * (cols - 1))
    p1y = int(y_scale_factors[0] * (rows - 1))
    p1 = [p1x, p1y]

    p2x = cols - int(x_scale_factors[1] * (cols - 1)) - 1
    p2y = int(y_scale_factors[1] * (rows - 1))
    p2 = [p2x, p2y]

    p3x = int(x_scale_factors[2] * (cols - 1))
    p3y = rows - int(y_scale_factors[2] * (rows - 1)) - 1
    p3 = [p3x, p3y]

    p4x = cols - int(x_scale_factors[3] * (cols - 1)) - 1
    p4y = rows - int(y_scale_factors[3] * (rows - 1)) - 1
    p4 = [p4x, p4y]

    dst_points = np.float32([p1, p2, p3, p4])

    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = perspective_transformation(input_image, projective_matrix)

    return transformed_image


def horizontal_squeeze(input_image, x_scale_factors):
    """
    Applies a perspective transformation to a given image which results in the corners being shifted some scale factor
    in the x-direction.

        The scale factors are applied to the x component of the coordinate pair in the order:
            1. top left
            2. top right
            3. bottom left
            4. bottom right

    :param input_image: The image to scale the corners of in the x-direction.
    :param x_scale_factors: Array with shape (4,) with the scale factors to apply to each corner's x-component.
    :return: The image with the scaling applied.
    """
    transformed_image = warp_corners(input_image, x_scale_factors, np.zeros(4))

    return transformed_image


def vertical_squeeze(input_image, y_scale_factors):
    """
    Applies a perspective transformation to a given image which results in the corners being shifted some scale factor
    in the y-direction.

        The scale factors are applied to the y component of the coordinate pair in the order:
            1. top left
            2. top right
            3. bottom left
            4. bottom right

    :param input_image: The image to scale the corners of in the x-direction.
    :param y_scale_factors: Array with shape (4,) with the scale factors to apply to each corner's y-component.
    :return: The image with the scaling applied.
    """
    transformed_image = warp_corners(input_image, np.zeros(4), y_scale_factors)

    return transformed_image


def affine_transformation(input_image, matrix):
    """
    Applies a affine transformation to a given image.

    :param input_image: The image to apply a affine transformation to.
    :param matrix: 2x3 matrix that defines the transformation.
    :return: The image with the affine transformation applied:
    """
    rows, cols = input_image.shape[0:2]
    transformed_image = cv2.warpAffine(input_image, matrix, (cols, rows))

    return transformed_image


def translate(input_image, dx, dy):
    """
    Translates an image by a given offset.

    :param input_image: The image to translate.
    :param dx: The change in x. Positive dx values result in a shift to the right.
    :param dy: The change in y. Positive y values result in a shift downwards.
    :return: The image with the translation applied.
    """
    transformed = affine_transformation(input_image, np.float32([[1, 0, dx],
                                                                 [0, 1, dy]]))

    return transformed


def rotate(input_image, angle, resize=True):
    """
    Rotates an image about the center by a given degree measure.

    :param input_image: The image to rotate.
    :param angle: The desired angle to rotate by. Positive angles result in a anti-clockwise rotation.
    :param resize: If True, increases the size of the resulting image to include portions of the image that
    ended outside of the original bounds of the image.
    :return: The image with the rotation applied.
    """
    rows, cols = input_image.shape[0:2]
    (cX, cY) = (cols // 2, rows // 2)
    matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])

    if resize:
        new_width = int((rows * sin) + (cols * cos))
        hew_height = int((rows * cos) + (cols * sin))
        matrix[0, 2] += (new_width / 2) - cX
        matrix[1, 2] += (hew_height / 2) - cY
    else:
        new_width = cols
        hew_height = rows

    return cv2.warpAffine(input_image, matrix, (new_width, hew_height))


def scale(input_image, width_scale_factor, height_scale_factor):
    """
    Scales an image by a given width and height scale factor.

    :param input_image: The image to scale.
    :param width_scale_factor: The scale factor to increase the width by. Values over the interval (0, 1) result in a
    reduction, and values over the interval (1, ∞) result in a dilation.
    :param height_scale_factor: The scale factor to increase the height by. Values over the interval (0, 1) result in a
    reduction, and values over the interval (1, ∞) result in a dilation.
    :return: The image with the scaling applied.
    """
    rows, cols = input_image.shape[0:2]
    width = rows * width_scale_factor
    height = cols * height_scale_factor

    dims = (int(width), int(height))

    scaled = cv2.resize(input_image, dims)

    return scaled


def average_blur(input_image, strength):
    """
    Blurs an image by averaging the pixel values in a radius of strength around every pixel.

    :param input_image: The image to blur.-
    :param strength: The size of the burring filter to apply to the image. The larger the matrix, the larger the
    area sampled to average over is.
    :return: The image with the average blur applied.
    """
    kernel = np.ones((strength, strength), np.float32) / (strength * strength)
    blurred = cv2.filter2D(input_image, -1, kernel)

    return blurred


def median_blur(input_image, strength):
    """
    Blurs an image by taking the median pixel values in a radius of strength around every pixel.

    :param input_image: The image to blur.
    :param strength: The size of the burring filter to apply to the image. The larger the matrix, the larger the
    area sampled to take the median over is. Must be odd
    :return: The image with the median blur applied.
    """
    median = cv2.medianBlur(input_image, strength)

    return median


def gaussian_blur(input_image, standard_deviation, strength):
    """
    Applies a gaussian blur to an image.

    :param input_image: The image to blur.
    :param standard_deviation: The standard deviation for the normal distribution.
    :param strength: The size of the burring filter to apply to the image. The larger the matrix, the larger the
    area sampled to blur per pixel. Must be odd.
    :return: The image with the gaussian blur applied.
    """
    blurred = cv2.GaussianBlur(input_image, (strength, strength), standard_deviation)

    return blurred


def gaussian_noise(input_image, standard_deviation, strength):
    """
    Overlays gaussian noise over an image.

    :param input_image: The image to add gaussian noise to.
    :param strength: How strong the noise to add is.
    :param standard_deviation: How many standard deviations for the gaussian noise.
    :return: The image with noise overlaid.
    """
    rows, cols, chans = input_image.shape

    mu = 0  # mean
    sigma = standard_deviation  # standard deviation
    gauss = np.random.normal(mu, sigma, input_image.size) * strength * 50
    gauss = gauss.reshape(rows, cols, chans).astype('uint8')

    noisy = input_image + gauss

    return noisy


def speckle_noise(input_image, standard_deviation, strength):
    """
    Overlays speckle noise over an image.

    :param input_image: The image to add speckle noise to.
    :param strength: How strong the noise to add is.
    :param standard_deviation: How many standard deviations for the gaussian noise.
    :return: The image with noise overlaid.
    """
    rows, cols, chans = input_image.shape

    mu = 0  # mean
    sigma = standard_deviation  # standard deviation
    gauss = np.random.normal(mu, sigma, input_image.size) * strength
    gauss = gauss.reshape(rows, cols, chans).astype('uint8')

    noisy = input_image + input_image * gauss

    return noisy


def _create_x_y_mesh(input_image):
    """
    Creates a mesh for x-values and y-values with the same size as the input image.
    In other words, two arrays of shape (rows, cols) are generated that look like the following:

        xs = [
                [0.0, 1.0, 2.0, ... (cols - 2), (cols - 1)],
                [0.0, 1.0, 2.0, ... (cols - 2), (cols - 1)],
                [0.0, 1.0, 2.0, ... (cols - 2), (cols - 1)],
                .
                .
                .
                [0.0, 1.0, 2.0, ... (cols - 2), (cols - 1)],
                [0.0, 1.0, 2.0, ... (cols - 2), (cols - 1)]

                    ] shape=(rows, cols)

        ys = [
                [0.0, 0.0, 0.0, ... 0.0, 0.0],
                [1.0, 1.0, 1.0, ... 1.0, 1.0],
                [2.0, 2.0, 2.0, ... 2.0, 2.0],
                .
                .
                .
                [(rows - 2), (rows - 2), (rows - 2), ... (rows - 2), (rows - 2)]
                [(rows - 1), (rows - 1), (rows - 1), ... (rows - 1), (rows - 1)]

                    ] shape=(rows, cols)


    :param input_image: The image to generate a mesh for.
    :return: The mesh for the x-values and the mesh for the y-values.
    """

    rows, cols = input_image.shape[0:2]
    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)

    xs, ys = np.array(np.meshgrid(x, y), dtype=np.float32)

    return xs, ys


def distort_with_noise(input_image, noise):
    """
    Distorts an image with a given noise mesh.

    :param input_image: The image to apply the noise to.
    :param noise: A mesh of shape (rows, cols) to apply to the image.
    :return: The image remapped using the given noise.
    """
    xv, yv = _create_x_y_mesh(input_image)

    xv += noise
    yv += noise

    remapped = cv2.remap(input_image, xv, yv, cv2.INTER_LINEAR)

    return remapped


def gaussian_noise_distortion_2d(input_image, standard_deviation, strength):
    """
    Distorts an image with gaussian noise in two dimensions.

    :param input_image: The image to add gaussian noise to.
    :param strength: How strong the noise to add is. Higher strength values make the sampling more dispersed.
    :param standard_deviation: How many standard deviations for the gaussian noise. For this distortion, higher
    standard deviation values (sigmas) make the image less wavy.
    :return: The image remapped to the given mesh.
    """
    rows, cols = input_image.shape[0:2]
    input_shape = np.random.rand(rows, cols)

    gauss = ndi.gaussian_filter(input_shape, standard_deviation)
    gauss -= np.amin(gauss)
    gauss /= np.amax(gauss)
    gauss = (2 * gauss - 1) * strength  # translate correctly

    remapped = distort_with_noise(input_image, gauss)

    return remapped


def gaussian_noise_distortion_1d(input_image, standard_deviation, strength):
    """
    Distorts an image with gaussian noise in one dimension.

    :param input_image: The image to add gaussian noise to.
    :param strength: How strong the noise to add is. Higher strength values make the sampling more dispersed.
    :param standard_deviation: How many standard deviations for the gaussian noise. For this distortion, higher
    standard deviation values (sigmas) make the image less wavy.
    :return: The image remapped to the given mesh.
    """
    rows, cols = input_image.shape[0:2]
    input_shape = np.random.rand(cols)

    gauss = ndi.gaussian_filter(input_shape, standard_deviation)
    gauss = np.tile(gauss, (rows, 1))
    gauss -= np.amin(gauss)
    gauss /= np.amax(gauss)
    gauss = (2 * gauss - 1) * strength  # translate correctly

    remapped = distort_with_noise(input_image, gauss)

    return remapped


def threshold(input_image, standard_deviation, threshold):
    """
    Applies a gaussian filter to an image with thresholding. Deep fries image.

    :param input_image: The image to apply the thresholding to.
    :param standard_deviation: How many standard deviations for the gaussian noise.
    :param threshold: The value to cut thresholding off at.
    :return: The image with thresholding applied.
    """

    gauss = ndi.gaussian_filter(input_image, standard_deviation)
    thresholded = 1.0 * (gauss > threshold)

    return thresholded


def save_image(input_image, filepath):
    """
    Saves an image to the given filepath

    :param input_image: The image to save.
    :param filepath: The path to save the image to. Includes the name of the image.
    :return:
    """
    cv2.imwrite(filepath, input_image)


if __name__ == '__main__':
    PATH = "./data/sub_set/"

    image = load(PATH + 'train/bb6-8-8-1pK1N1pP-1b6-7k-8-1q6.jpeg')

    transformed1 = threshold(image, 1, 150)

    plt.subplot(1, 2, 1)
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.imshow(transformed1)

    plt.show()
