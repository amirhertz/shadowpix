from image_utils import *
import numpy as np


class GlobalMethod:
    def __init__(self, images_paths, image_size, light_angle, w_g, w_s, temperature):
        self.num_images = len(images_paths)
        if self.num_images > 4:
            raise ValueError("can't handle more than 4 images")
        self.raw_images = load_images(image_size, *images_paths)
        self.gp_images = apply_filter(BOTH_KERNELS, self.raw_images)
        self.size = image_size
        self.s = 1 / np.tan(light_angle * np.pi / 180)
        self.height_field = np.zeros([image_size, image_size])
        self.height_field_images = np.ones([self.num_images, image_size, image_size])  # all white
        self.T = temperature
        self.w_g = w_g
        self.w_s = w_s
        self.radius = 10
        self.value = self.objective(self.height_field_images)
        self.buffer_vector = np.arange(0, self.radius)
        selecting_matrix = np.expand_dims(np.arange(0, self.size), axis=1)
        self.selecting_matrix = selecting_matrix + self.buffer_vector + 1

    def calculate_update(self, row, col):
        update_images = self.height_field_images.copy()
        for i in range(self.num_images):
            if i == 0:
                to_compare = self.height_field[row, :]
            elif i == 1:
                to_compare = np.flip(self.height_field[row, :], 0)
            elif i == 2:
                to_compare = self.height_field[:, col]
            else:
                to_compare = np.flip(self.height_field[:, col], 0)
            height_select = np.pad(to_compare, (0, self.radius), 'constant', constant_values=(0, -10000))
            compare_values = height_select[self.selecting_matrix]
            compare_values = np.max(compare_values, axis=1)
            mask_image =  1 - (to_compare < compare_values).astype(np.float64)
            if i % 2 == 1:
                mask_image = np.flip(mask_image, 0)
            if i < 2:
                update_images[i, row, :] = mask_image
            else:
                update_images[i, :, col] = mask_image
        return update_images

    def objective(self, height_field_images):
        height_field_image_smooth = apply_filter(GAUSSIAN_KERNEL, height_field_images)
        height_field_gradient = apply_filter(GRADIENT_KERNEL, [self.height_field]) [0]
        height_field_image_both = apply_filter(BOTH_KERNELS, height_field_images)
        value = 0
        for idx, image in enumerate(self.raw_images):
            value += squared_error(image, height_field_image_smooth[idx])
            value += self.w_g * squared_error(self.gp_images[idx], height_field_image_both[idx])
        value += self.w_s * squared_error(height_field_gradient, 0)
        return value

    def export(self):
        pass

    def step(self):
        for _ in range(self.size * self.size):
            row = np.random.randint(0, self.size)
            col = np.random.randint(0, self.size)
            delta = np.random.randint(-5, 6)
            self.height_field[row, col] += delta
            if self.is_step_valid(row, col):
                updated_images = self.calculate_update(row, col)
                new_value = self.objective(updated_images)
                profit = self.value - new_value
                if profit > 0 or np.random.random() < np.e**(profit / self.T):
                    self.T -= 1
                    self.height_field_images = updated_images
                    return new_value
                else:
                    self.height_field[row, col] -= delta
            else:
                self.height_field[row, col] -= delta
        return -1

    def optimize(self):
        steps = self.T
        for i in range(steps):
            value = self.step()
            if value > 0:
                print("Objective value on step %d is %.3f" % (i, value))
            else:
                print("step %d failed" % i)

    def is_step_valid(self, row, col):
        new_value = self.height_field[row, col]
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
    path_a = './images/hopper_gas.jpg'
    path_b = './images/hockney_chairs.jpg'
    path_c = './images/modigliani_women.jpg'
    path_d = './images/iit.jpg'
    paths = [path_a, path_b, path_c, path_d]
    GBM = GlobalMethod(paths, 20, 60, .5, .5, 1000)
    GBM.optimize()


if __name__ == '__main__':
    global_method()
