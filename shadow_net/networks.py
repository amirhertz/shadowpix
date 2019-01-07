import torch
from torch import nn
from torch.nn import init


def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


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
        batch_size = x.size()[0]
        x = x.view(batch_size * self.num_images, 1, 64, 64)
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, self.num_images, -1)
        encoded = encoded.view(batch_size, encoded.shape[1] // self.num_images, -1)
        return self.decoder(encoded)


if __name__ == '__main__':
    e = Encoder()
    d = Decoder(4)
    x = torch.rand(3, 3, 64, 64)
    x = e(x)
    d(x)

