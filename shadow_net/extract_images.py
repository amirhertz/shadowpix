import torch
import torch.nn as nn


class H2I(nn.Module):
    def __init__(self, radius, im_size, device=torch.device('cpu')):
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
        image = nn.ReLU(compare_heights - height_field)
        return image


if __name__ == '__main__':
    m = H2I(3,5)
    h = torch.rand(5,5)
    m(h)