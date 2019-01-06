from image_utils import *
from mesh_utils import *
import numpy as np

epsilon = 1e-10


def check_constrains(r, u, v):
    constrains = [r <= u[:, :-1] +  epsilon, r <= u[:, 1:] + + epsilon, r[: -1, :] <= v[1:, :] + epsilon]
    for idx, constrain in enumerate(constrains):
        if not np.all(constrain):
            print('fail on constrain %d' % (idx + 1))


def local_method(images_paths, image_size, light_angle, output_file):
    image_a, image_b, image_c = load_images(image_size, *images_paths)
    image_a, image_b, image_c = 1 - image_a, 1 - image_b, 1 - image_c
    light_angle *= np.pi / 180
    s = 1 / np.tan(light_angle)
    u = np.zeros([image_size, image_size + 1])
    # side constrains
    for i in range(image_size):
        u[:, i + 1] = u[:, i] + s * (image_b[:, i] - image_a[:, i])
    u += (s * image_a[:, 0])[:, np.newaxis]
    # height constrains
    images_constrains = s * (-image_a[:image_size - 1, :] + image_a[1:, :] - image_c[1:, :])
    u[0, :] -= min(0, np.min(u[0, :]))
    for j in range(image_size - 1):
        height_constrain = -u[j + 1, :-1] + u[j, :-1] + images_constrains[j, :]
        u[j + 1, :] += max(np.max(height_constrain), 0)
    r = u[:, : image_size] - s * image_a
    v = r + s * image_c
    check_constrains(r, u, v)
    ds_to_mesh(r, u, v, 0.1, output_file)
    print('done')

