from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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
    return 1- np.array(image).astype(np.float) / 255


def show_image(image):
    if len(image.shape) < 3:
        image = np.expand_dims(image,2)
        image = np.repeat(image, 3, 2)
    plt.imshow(image)
    plt.show(block=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # TESTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if  __name__ == '__main__':
    path_a = './images/hopper_gas.jpg'
    im_a = load_image(path_a, 100)
    show_image(im_a)
