from image_utils import *
import numpy as np


class GlobalMethodState:
    def __init__(self, images_paths, image_size, light_angle, w_g, w_s, temperature):
        if len(images_paths) > 4:
            raise ValueError("can't handle more than 4 images")
        self.raw_images = load_images(image_size, *images_paths)
        self.gp_images = apply_filter(BOTH_KERNELS, self.raw_images)
        self.size = image_size
        self.s = 1 / np.tan(light_angle * np.pi / 180)
        self.height_field = np.zeros([image_size, image_size])
        self.T = temperature
        self.w_g = w_g
        self.w_s = w_s
        self.radius = 10
        self.value = self.objective(self.height_field)
        self.buffer_vector = np.arange(0, self.radius)

    def calculate_height_field_images(self, height_field):
        pass

    def objective(self, height_field):
        height_field_images = self.calculate_height_field_images(height_field)
        height_field_image_smooth = apply_filter(GAUSSIAN_KERNEL, height_field_images)
        height_field_image_gradient = apply_filter(GRADIENT_KERNEL, height_field_images)
        height_field_image_both = apply_filter(BOTH_KERNELS, height_field_images)
        value = 0
        for idx, image in self.raw_images:
            value += squared_error(image, height_field_image_smooth[idx])
            value += self.w_g * squared_error(self.gp_images[idx], height_field_image_both[idx])
            value += self.w_s * squared_error(height_field_image_gradient[idx], 0)
        return value


    def export(self):
        pass

    def step(self):
        new_height_field = self.height_field.copy()
        for _ in range(self.size * self.size):
            row = np.random.randint(0, self.size)
            col = np.random.randint(0, self.size)
            delta = np.random.randint(-5, 6)
            new_height_field[row, col] += delta
            if self.is_step_valid(row, col, new_height_field):
                new_value = self.objective(new_height_field)
                profit = self.value - new_value
                if profit > 0 or np.random.random() < np.e^(profit / self.T):
                    self.T -= 1
                    self.height_field = new_height_field
                    return new_value
            else:
                new_height_field[row, col] -= delta
        return -1

    def optimize(self):
        steps = self.T
        for i in range(steps):
            value = self.step()
            if value > 0:
                print("Objective value on step %d is %.3f" % (i, value))
            else:
                print("step %d failed" % i)



    def is_step_valid(self, row, col, new_value):
        for i in range(len(self.raw_images)):
            check_col = None
            window = None
            # check left
            if i == 0 and col > self.radius:
                check_col = self.height_field[row, col - self.radius - 1]
                window = self.height_field[row, col - self.radius: col] - self.buffer_vector
            # check right
            elif i == 1 and col < self.size - self.radius - 1:
                check_col = self.height_field[row, col + self.radius + 1]
                window = self.height_field[row, col: col +  self.radius] - np.flip(self.buffer_vector, 0)
            # check down
            elif i == 2 and row > self.radius:
                check_col = self.height_field[row - self.radius - 1, col]
                window = self.height_field[row - self.radius: row, col] - self.buffer_vector
            # check up
            elif row < self.size - self.radius - 1:
                check_col = self.height_field[row + self.radius + 1, col]
                window = self.height_field[row: row + self.radius, col] - np.flip(self.buffer_vector, 0)
            if check_col and new_value - self.radius > check_col:
                if np.max(window) < check_col:
                    return False
        return True


def global_method():




# if __name__ == '__main__':
#     path_a = './images/hopper_gas.jpg'
#     path_b = './images/hockney_chairs.jpg'
#     path_c = './images/modigliani_women.jpg'
#     path_d = './images/miro.jpg'
#     paths = [path_a, path_b, path_c, path_d]
#     global_method(paths, 200, 60, './models/paintings.obj')