from utils import *
import numpy as np

def calculate_row_heights(dir_a_values, dir_b_values, s):
    return None, None

def local_method(images_paths, image_size, light_angle):
    image_a, image_b, image_c = load_images(image_size, *images_paths)
    light_angle *= np.pi / 180
    s = 1 / np.tan(light_angle)
    # r = np.zeros([image_size, image_size])
    u = np.zeros([image_size, image_size + 1])
    # v = np.zeros([image_size, image_size])
    # side constrains
    for i in range(image_size):
        u[:, i + 1] = u[:, i] + s * (image_b[:, i] - image_a[:, i])
    u += (s * image_a[:, 0])[:, np.newaxis]
    # height constrains
    images_constrains = s * (-image_a[:image_size - 1, :] + image_a[1:, :] - image_c[1:, :])
    u[0, :] -= min(0, np.min(u[0, :]))
    for j in range(image_size - 1):
        height_constrain = u[j + 1, 1:] - u[j, 1:] + images_constrains[j, :]
        u[j + 1, :] += max(np.max(height_constrain), 0)
    r = u[:, : image_size] - s * image_a
    v = r + s * image_c
    print('done')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # TESTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if  __name__ == '__main__':
    path_a = './images/hopper_gas.jpg'
    path_b = './images/hockney.jpg'
    path_c = './images/hockney.jpg'
    paths = [path_a, path_b, path_c]
    local_method(paths, 200, 30)