from shadow_net.networks import *
from image_utils import *
from shadow_net.heightfield_loss import *
import torch
import torchvision
from shadow_net.data_loader import DataLoader
import sys
import os


def load():
    path_a = '../images/hopper_gas.jpg'
    path_b = '../images/hockney_chairs.jpg'
    path_c = '../images/modigliani_women.jpg'
    path_d = '../images/miro.jpg'
    paths = [path_a, path_b, path_c, path_d, path_a, path_b, path_c, path_d]
    raw_images = load_images(64, *paths)
    return [image.astype(np.float32) for image in raw_images]


def run(running_path):
    k = 4
    data = DataLoader(os.path.join(running_path, '../../', 'gray_celebA/'), k)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)


    net = EncoderDecoder(Encoder, Decoder, k)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epochs = 10
    loss_func = HFLoss(64, 10, 0)
    # raw_images = load()
    for epoch in range(epochs):
        batch_ind = 0
        for batch in data_loader:
            batch_ind += 1
            # images = torch.stack([torch.from_numpy(image).unsqueeze(0) for image in raw_images], 0)
            heightfield = net(batch.squeeze(0))
            loss = loss_func(batch.squeeze(0), heightfield)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('{}/{}: {}'.format(epoch, batch_ind, loss.item()))

            if batch_ind % 100:
                save_heightfield(heightfield, os.path.join(running_path, '..', 'heightfields/heightfield_latest.png'))

        save_heightfield(heightfield, os.path.join(running_path, '..', 'heightfields/heightfield_.png'.format(epoch)))


def save_heightfield(heightfield, path):
    heightfield_image = H2I(64, 10)(heightfield).cpu().detach().numpy()
    save_image(heightfield_image[0], path)

def view_heightfield(heightfield):
    heightfield_image = H2I(64,10)(heightfield).cpu().detach().numpy()
    show_image(heightfield_image[0])


if __name__ == '__main__':
    running_path = os.path.dirname(os.path.abspath(__file__))
    run(running_path)