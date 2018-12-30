from shadow_net.networks import *
from image_utils import *
from shadow_net.heightfield_loss import *
import torch


def load():
    path_a = '../images/hopper_gas.jpg'
    path_b = '../images/hockney_chairs.jpg'
    path_c = '../images/modigliani_women.jpg'
    path_d = '../images/miro.jpg'
    paths = [path_a, path_b, path_c, path_d, path_a, path_b, path_c, path_d]
    raw_images = load_images(64, *paths)
    return [image.astype(np.float32) for image in raw_images]


def run():
    net = EncoderDecoder(Encoder, Decoder, 4)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epochs = 10
    loss_func = HFLoss(64, 10, 0)
    raw_images = load()
    for _ in range(epochs):
        images = torch.stack([torch.from_numpy(image).unsqueeze(0) for image in raw_images], 0)
        heightfield = net(images)
        loss = loss_func(images, heightfield)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())


def view_heightfield(heightfield):
    heightfield_image = H2I(64,10)(heightfield).cpu().detach().numpy()
    show_image(heightfield_image[0])


if __name__ == '__main__':
    run()