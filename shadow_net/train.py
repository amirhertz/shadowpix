from shadow_net.networks import *
from image_utils import *
import torch


def load():
    path_a = '../images/hopper_gas.jpg'
    path_b = '../images/hockney_chairs.jpg'
    path_c = '../images/modigliani_women.jpg'
    path_d = '../images/miro.jpg'
    paths = [path_a, path_b, path_c, path_d]
    raw_images = load_images(64, *paths)
    return [torch.from_numpy(image.astype(np.float32)).unsqueeze(0) for image in raw_images]


def run():
    encoder = Encoder()
    decoder = Decoder(4)
    epochs = 10
    images = load()
    loss_f = HFLoss(64, 10)
    images = torch.stack(images, 0)
    for _ in range(epochs):
        code = encoder(images)
        code = code.view(1, -1)
        heightfield = decoder(code)
        heightfield = heightfield.squeeze(0).squeeze(0)
        loss = loss_f(images, heightfield)
        loss.backward()






if __name__ == '__main__':
    run()