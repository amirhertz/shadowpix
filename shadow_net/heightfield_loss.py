import torch
import torch.nn as nn
from torch.autograd import Variable

class H2I(nn.Module):
    
    def __init__(self,im_size, radius, device=torch.device('cpu')):
        super(H2I, self).__init__()
        self.radius = radius
        self.im_size = im_size
        self.padding = nn.ConstantPad2d((0, radius, 0, 0), -1000)
        buffer_vector = torch.arange(1, radius + 1)
        selecting_matrix = torch.arange(0, im_size).unsqueeze(1) + buffer_vector
        buffer_vector = buffer_vector.float() / 10
        self.buffer_vector , self.selecting_matrix = buffer_vector.to(device), selecting_matrix.to(device)

    def __call__(self, heightfield):
        return self.forward(heightfield)

    def forward(self, heightfield):
        heightfield_padded = self.padding(heightfield)
        unroll_heights = heightfield_padded[:, :, :, self.selecting_matrix] - self.buffer_vector
        unroll_heights2 = heightfield_padded.index_select(3, self.selecting_matrix.view(-1)).view(*heightfield.size(), self.radius) - self.buffer_vector
        assert (unroll_heights == unroll_heights2).all()
        compare_heights, _ = unroll_heights2.max(dim=4)
        # image = 1 - torch.clamp(compare_heights - heightfield, 0, 1)
        image = compare_heights - heightfield
        return image

    # def naive(self, heightfield):
    #     heightfield_padded = self.padding(heightfield)
    #     for row_id in range(64):
    #         for col_id in range(64):
    #             cur_block = heightfield.index_select(heightfield, torch.arange(col_id, min(col_id + 10, 64)))
    #     unroll_heights = heightfield_padded[:, :, :, self.selecting_matrix] - self.buffer_vector
    #     compare_heights, _ = unroll_heights.max(dim=4)
    #     # image = 1 - torch.relu(torch.tanh(compare_heights - heightfield))
    #     image = 1 - torch.clamp(compare_heights - heightfield, 0, 1)
    #     return image


class HFLoss:
    def __init__(self, im_size, radius, gradient_weight, device=torch.device('cpu')):
        self.h2i = H2I(im_size, radius, device)
        self.mse = nn.MSELoss()
        self.gradient_conv = self.init_grad_conv()
        self.gamma = gradient_weight
        self.zeros = torch.zeros(1, 1, im_size, im_size)

    def __call__(self, real_images, heightfield):
        loss = Variable(torch.zeros(1,1, requires_grad=True))
        shape = real_images.shape
        real_images = real_images.view(shape[0] // heightfield.shape[0], -1, *shape[1:])
        for i in range(len(real_images)):
            rot_img = real_images[i]
            rot_heightfield = heightfield
            if i == 1:
                rot_img = rot_img.transpose(2,3).flip(3)
                rot_heightfield = rot_heightfield.transpose(2,3).flip(3)
            elif i == 2:
                rot_img = rot_img.flip(3)
                rot_heightfield = rot_heightfield.flip(3)
            elif i==3:
                rot_img = rot_img.transpose(2, 3).flip(2)
                rot_heightfield = rot_heightfield.transpose(2, 3).flip(2)
            assert i < 4, 'Each heightfield holds up to 4 images but %d were given' % len(real_images)
            heightfield_result = self.h2i(rot_heightfield)
            # heightfield_result = heightfield
            loss += self.mse(heightfield_result, heightfield_result*2)
        if self.gamma:
            zeros = self.zeros.expand(heightfield.shape)
            loss += self.gamma * self.mse(self.gradient_conv(heightfield), zeros)
        return loss

    @staticmethod
    def init_grad_conv():
        filter_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        filter_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            conv_x.weight = nn.Parameter(filter_x.unsqueeze(0).unsqueeze(0))
            conv_y.weight = nn.Parameter(filter_y.unsqueeze(0).unsqueeze(0))

        def apply_gradient_filter(images):
            gradient_x = conv_x(images)
            gradient_y = conv_y(images)
            g = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
            return g
        return apply_gradient_filter


if __name__ == '__main__':
    d = torch.device('cpu')
    m = H2I(5, 3, d)
    h = torch.rand(2, 1, 5, 5).to(d)
    m(h)