from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import os

GRADIENT_KERNEL = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
x = cv2.getGaussianKernel(5, 1)
GAUSSIAN_KERNEL = np.dot(x,x.T)

BOTH_KERNELS = convolve(GAUSSIAN_KERNEL, GRADIENT_KERNEL)


def load_images(size, *paths):
    return [load_image(path, size) for path in paths]


def load_image(path, size):
    image = Image.open(path).convert('L')
    w, h = image.size
    w_0, h_0 = 0, 0
    if w > h:
        w_0 = (w - h) // 2
        w -= w_0
    elif h > w:
        h_0 = (h - w) // 2
        h -= h_0
    image = image.crop((w_0, h_0, w, h)).resize((size, size), Image.ANTIALIAS)
    return np.array(image) / 255


def squared_error(mat_a, mat_b):
    return np.square(mat_a - mat_b).sum()


def show_image(image):
    image = np.clip(image, 0, 1)
    if image.shape[0] == 1:
        image.shape = image.shape[1:]
    if len(image.shape) < 3:
        image = np.expand_dims(image,2)
        image = np.repeat(image, 3, 2)
    plt.imshow(image)
    plt.show(block=True)


def save_image(image, path):
    image = np.clip(image, 0, 1)
    img = Image.fromarray(np.uint8(image * 255), 'L')
    img.save(path)


def apply_filter(kernel, images):
    filtered =  [convolve(image, kernel) for image in images]
    return [np.clip(image, 0, 1) for image in filtered]

def save_bw(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.png':
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, file_name + '.jpg')
                image = load_image(input_file, 200)
                save_image(image, output_file)


def save_image(image, path):
    image = np.clip(image, 0, 1)
    if image.shape[0] == 1:
        image.shape = image.shape[1:]
    if len(image.shape) < 3:
        image = np.expand_dims(image,2)
        image = np.repeat(image, 3, 2)

    plt.imsave(path, image)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # TESTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
    save_bw('./images', './images_bw')

