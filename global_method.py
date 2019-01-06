import sys
sys.path.append('../')
from image_utils import *
import numpy as np
import pickle
import os
from mesh_utils import heightfield_to_mesh
import torch
from torch import nn


class GlobalMethod:
    def __init__(self, images_paths, image_size, w_g, w_s, radius=20, device=torch.device('cuda:0')):
        self.num_images = len(images_paths)
        if self.num_images > 4:
            raise ValueError("can't handle more than 4 images")
        raw_images = load_images(image_size, *images_paths)
        self.raw_images = torch.stack([torch.from_numpy(image).to(device) for image in raw_images]).float().unsqueeze(1).detach()
        self.grad_conv = self.init_grad_conv(device)
        self.smooth_conv = self.init_smooth_conv(device)
        self.gp_images = self.grad_conv(self.smooth_conv(self.raw_images))
        self.size = image_size
        self.heightfield = torch.zeros([1, 1, image_size, image_size], requires_grad=False).to(device)
        self.heightfield_images = torch.ones([self.num_images,1, image_size, image_size], requires_grad=False).to(device)  # all white
        self.T = 0
        self.alpha = 0
        self.w_g = w_g
        self.w_s = w_s
        self.radius = radius
        self.device = device
        self.value = self.objective(self.heightfield_images)
        self.pad = nn.ConstantPad1d((0, radius), -1000)
        buffer_vector = torch.arange(1, self.radius +1, requires_grad=False).to(device)
        selecting_matrix = torch.arange(0, self.size, requires_grad=False).unsqueeze(1).to(device)
        self.selecting_matrix = selecting_matrix + buffer_vector
        self.buffer_vector = buffer_vector.float()

        self.shadow_vec = torch.arange(1, self.radius + 2, dtype=torch.float, requires_grad=False).to(device)
        self.shadow_vec_flipped = self.shadow_vec.float().flip(0)

    def calculate_update(self, row, col):
        update_images = self.heightfield_images.clone()
        for i in range(self.num_images):
            if i == 0:
                to_compare = self.heightfield[0, 0, row, :]
            elif i == 1:
                to_compare = self.heightfield[0, 0, row, :].flip(0)
            elif i == 2:
                to_compare = self.heightfield[0, 0, :, col]
            else:
                to_compare = self.heightfield[0, 0, :, col].flip(0)
            height_select = self.pad(to_compare)

            compare_values = height_select[self.selecting_matrix] - self.buffer_vector
            compare_values, _ = compare_values.max(dim=1)
            updated_pixels = torch.clamp(to_compare - compare_values, 0,1)
            if i % 2 == 1:
                updated_pixels = updated_pixels.flip(0)
            if i < 2:
                update_images[i,  0, row, :] = updated_pixels
            else:
                update_images[i, 0, :, col] = updated_pixels
        return update_images

    def mse(self, x, y):
        loss = x - y
        loss = (loss * loss).sum()
        loss /= (self.size * self.size * self.num_images)
        return loss

    def objective(self, height_field_images):
        height_field_image_smooth = self.smooth_conv(height_field_images)
        height_field_gradient = self.grad_conv(self.heightfield)
        height_field_image_both = self.grad_conv(self.smooth_conv(self.heightfield_images))
        value = (1 - self.w_g - self.w_s) * self.mse(height_field_image_smooth, self.raw_images)
        if self.w_g:
            value += self.w_g * self.mse(height_field_image_both, self.gp_images)
        if self.w_s:
            value += self.w_s * self.mse(height_field_gradient, 0)
        return value

    def export_data(self, name):
        untype = ["<class 'builtin_function_or_method'>", "<class 'method'>", "<class 'method-wrapper'>",
                  "<class 'type'>", "<class 'NoneType'>", "<class 'dict'>", "<class 'str'>", "<class 'function'>"]
        d = {}
        for att in self.__dir__():
            att_type = str(type(eval("self." + att)))
            if att_type not in untype:
                x = eval("self." + att)
                if hasattr(x, 'device'):
                    x = x.cpu()
                d[att] = x
        d_path = './data/%s.pkl' % name
        print("checkpointing %s" % name)
        with open(d_path, 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    def load_data(self, name):
        d_path = './data/%s.pkl' % name
        if not os.path.isfile(d_path):
            print("data file- %s is not exist" % name)
            return
        print("Loading %s" % name)
        with open(d_path, 'rb') as f:
            d = pickle.load(f)
            for att in d:
                x = d[att]
                if hasattr(x, 'device'):
                    x = x.to(self.device)
                self.__setattr__(att, x)

    def is_step_valid_a(self, row, col):
        new_value = self.heightfield[0, 0, row, col]
        for i in range(len(self.raw_images)):
            # check left
            if i == 0 and col > self.radius:
                window = self.heightfield[0, 0, row, col - self.radius - 1: col] + self.shadow_vec_flipped
            # check right
            elif i == 1 and col < self.size - self.radius - 1:
                window = self.heightfield[0, 0, row, col + 1: col + self.radius + 2] + self.shadow_vec
            # check down
            elif i == 2 and row > self.radius:
                window = self.heightfield[0, 0, row - self.radius - 1: row, col] + self.shadow_vec_flipped
            # check up
            elif row < self.size - self.radius - 1:
                window = self.heightfield[0, 0, row + 1: row + self.radius + 2, col] + self.shadow_vec
            else:
                continue
            if window.max() < new_value:
                return False
        return True

    def is_step_valid_b(self, row, col):
        for i in range(len(self.raw_images)):
            # check left
            if i == 0 and col > self.radius:
                to_compare = self.heightfield[0, 0, row, col - self.radius - 1]
                window = self.heightfield[0, 0, row, col - self.radius: col + 1] + self.shadow_vec
            # check right
            elif i == 1 and col < self.size - self.radius - 1:
                to_compare = self.heightfield[0, 0, row, col + self.radius + 1]
                window = self.heightfield[0, 0, row, col: col + self.radius + 1] + self.shadow_vec_flipped
            # check down
            elif i == 2 and row > self.radius:
                to_compare = self.heightfield[0, 0, row - self.radius - 1, col]
                window = self.heightfield[0, 0, row - self.radius: row + 1, col] + self.shadow_vec
            # check up
            elif row < self.size - self.radius - 1:
                to_compare = self.heightfield[0, 0, row + self.radius + 1, col]
                window = self.heightfield[0, 0, row: row + self.radius + 1, col] + self.shadow_vec_flipped
            else:
                continue
            if window.max() < to_compare:
                return False
        return True

    def is_step_valid(self, row, col, delta):
        if delta:
            return self.is_step_valid_a(row, col)
        else:
            return self.is_step_valid_b(row, col)

    def export_mesh(self, name, light_angle):
        s = 1 / np.tan(light_angle * np.pi / 180)
        height_field = self.heightfield.cpu().squeeze(0).squeeze(0).numpy() * s
        height_field -= height_field.min()
        heightfield_to_mesh(height_field, './models/%s.obj' % name)

    def step(self):
        for _ in range(self.size * self.size):
            row = np.random.randint(0, self.size)
            col = np.random.randint(0, self.size)
            delta = 0
            while not delta:
                delta = np.random.randint(-5, 6)
            self.heightfield[0, 0,row, col] += delta
            if self.is_step_valid(row, col, delta > 0):
                updated_images = self.calculate_update(row, col)
                new_value = self.objective(updated_images)
                profit = (self.value - new_value) * 100
                # if profit < 0:
                #     print('prob is: ' + str(np.e**(profit / self.T).item()))
                if profit > 0 or (self.T > 0 and np.random.random() < np.e**(profit / self.T)):
                    self.T -= self.alpha
                    self.heightfield_images = updated_images
                    self.value = new_value
                    return new_value
                else:
                    self.heightfield[0, 0, row, col] -= delta
            else:
                self.heightfield[0, 0, row, col] -= delta
        return -1

    def optimize(self, steps, name, temperature=0.001):
        self.T = temperature
        min_t = 0
        failed_counter = 0
        self.alpha = (self.T - min_t) / steps
        for i in range(steps):
            value = self.step()
            if value < 0:
                failed_counter += 1
                if failed_counter == 100:
                    print("step %d failed and exit" % i)
                    break
            else:
                failed_counter = 0

            if (i + 1) % 5000 == 0:
                self.export_data(name)
            if i % 1000 == 0:
                if value > 0:
                    print("Objective value after %d steps is %.3f" % (i +1, value))
                else:
                    print("step %d failed" % i)
        print('done')
        self.export_data(name)

    @staticmethod
    def init_grad_conv(device):
        filter_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        filter_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            conv_x.weight = nn.Parameter(filter_x.unsqueeze(0).unsqueeze(0))
            conv_y.weight = nn.Parameter(filter_y.unsqueeze(0).unsqueeze(0))
        conv_x = conv_x.to(device)
        conv_y = conv_y.to(device)

        def apply_gradient_filter(images):
            gradient_x = conv_x(images)
            gradient_y = conv_y(images)
            g = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))
            g = torch.clamp(g, 0, 1)
            return g

        return apply_gradient_filter

    @staticmethod
    def init_smooth_conv(device):
        filter = torch.Tensor([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
        conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            conv.weight = nn.Parameter(filter.unsqueeze(0).unsqueeze(0))
        conv = conv.to(device)

        def apply_smooth_filter(images):
            return torch.clamp(conv(images), 0, 1)
        return apply_smooth_filter


def global_method():
    path_a = './images/celeb_a.jpg'
    path_b = './images/celeb_b.jpg'
    path_c = './images/celeb_c.jpg'
    # path_d = './images/roy_c.jpg'
    train_name = 'global_roy2_gpu'
    paths = [path_a, path_b, path_c]
    gbm = GlobalMethod(paths, 400, 0, 0, radius=20, device=torch.device('cuda:0'))
    # gbm.optimize(10000000, train_name, temprature=0)
    gbm.load_data(train_name)
    gbm.export_mesh(train_name, 60)


if __name__ == '__main__':
    global_method()
