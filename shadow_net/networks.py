import torch
from torch import nn


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
        x = torch.relu(self.conv1(x))
        x = self.unconv1(x)
        x = torch.relu(self.conv2(x))
        x = self.unconv2(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, encoder_class, decoder_class, num_images):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder_class()
        self.decoder = decoder_class(num_images)
        self.num_images = num_images

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.shape[0] // self.num_images, -1)
        return self.decoder(encoded)


if __name__ == '__main__':
    e = Encoder()
    d = Decoder(4)
    x = torch.rand(3, 3, 64, 64)
    x = e(x)
    d(x)

