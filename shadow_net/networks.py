import torch
from torch import nn
from torch.nn import functional as f
from torchvision.transforms import RandomRotation


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 12, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 24, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Linear(24 * 16 * 16, 1568)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 24 * 16 * 16)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_images):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(1568 * num_images, 24 * 16 * 16)
        self.conv1 = nn.Conv2d(24, 12, 3, 1, 1)
        self.unconv1 = nn.ConvTranspose2d(12, 12, 2, 2)
        self.conv2 = nn.Conv2d(12, 1, 3, 1, 1)
        self.unconv2 = nn.ConvTranspose2d(1, 1, 2, 2)


    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 24, 16, 16)
        x = f.relu(self.conv1(x))
        x = self.unconv1(x)
        x = f.relu(self.conv2(x))
        x = self.unconv2(x)
        return x


class H2I(nn.Module):
    def __init__(self,im_size, radius, device):
        super(H2I, self).__init__()
        self.radius = radius
        self.im_size = im_size
        self.padding = nn.ConstantPad2d((0, radius, 0, 0), -1000)
        buffer_vector = torch.arange(1, radius + 1)
        selecting_matrix = torch.arange(0, im_size).unsqueeze(1) + buffer_vector
        buffer_vector = buffer_vector.float()
        self.buffer_vector , self.selecting_matrix = buffer_vector.to(device), selecting_matrix.to(device)

    def __call__(self, height_field):
        return self.forward(height_field)

    def forward(self, height_field):
        height_field_padded = self.padding(height_field)
        unroll_heights = height_field_padded[:, self.selecting_matrix] - self.buffer_vector
        compare_heights, _ = unroll_heights.max(dim=2)
        image = nn.ReLU()(compare_heights - height_field)
        return image


class HFLoss:
    def __init__(self, im_size, radius, device=torch.device('cpu')):
        self.h2i = H2I(im_size, radius, device)
        self.l2 = nn.MSELoss()

    def __call__(self, real_images, heightfield):
        loss = 0
        for i in range(len(real_images)):
            if i == 0:
                rot_img = real_images[i]
                rot_heightfield = heightfield
            elif i == 1:
                rot_img = real_images[i].transpose(1,2).flip(2)
                rot_heightfield = heightfield.transpose(0,1).flip(1)
            elif i == 2:
                rot_img = real_images[i].flip(2)
                rot_heightfield = heightfield.flip(1)
            else:
                rot_img = real_images[i].transpose(1, 2).flip(1)
                rot_heightfield = heightfield.transpose(0, 1).flip(0)
            heightfield_result = self.h2i(rot_heightfield)
            loss += self.l2(heightfield_result.unsqueeze(0), rot_img)
        return loss


if __name__ == '__main__':
    e = Encoder()
    d = Decoder()
    x = torch.rand(3, 3, 64, 64)
    x = e(x)
    d(x)

