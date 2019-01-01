from image_utils import *
import numpy as np
import pickle
import os
from mesh_utils import heightfield_to_mesh

class GlobalMethod:
    def __init__(self, images_paths, image_size, w_g, w_s):
        self.num_images = len(images_paths)
        if self.num_images > 4:
            raise ValueError("can't handle more than 4 images")
        self.raw_images = load_images(image_size, *images_paths)
        self.gp_images = apply_filter(BOTH_KERNELS, self.raw_images)
        self.size = image_size
        self.height_field = np.zeros([image_size, image_size])
        self.height_field_images = np.ones([self.num_images, image_size, image_size])  # all white
        self.T = 0
        self.alpha = 0
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

    def export_data(self, name):
        untype = ["<class 'builtin_function_or_method'>", "<class 'method'>", "<class 'method-wrapper'>",
                  "<class 'type'>", "<class 'NoneType'>", "<class 'dict'>", "<class 'str'>"]
        d = {}
        for att in self.__dir__():
            att_type = str(type(eval("self." + att)))
            if att_type not in untype:
                d[att] = eval("self." + att)
        d_path = './data/%s.pkl' % name
        with open(d_path, 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    def load_data(self, name):
        d_path = './data/%s.pkl' % name
        if not os.path.isfile(d_path):
            print("data file is not exist")
            return
        with open(d_path, 'rb') as f:
            d = pickle.load(f)
            for att in d:
                self.__setattr__(att, d[att])

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
                # if profit < 0:
                #     print('prob is: ' + str(np.e**(profit / self.T )))
                if profit > 0 or np.random.random() < np.e**(profit / self.T):
                    self.T -= self.alpha
                    self.height_field_images = updated_images
                    self.value = new_value
                    return new_value
                else:
                    self.height_field[row, col] -= delta
            else:
                self.height_field[row, col] -= delta
        return -1

    def optimize(self, steps, name):
        self.T = 2
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
                print("checkpointing")
            if (i) % 100 == 0:
                if value > 0:
                    print("Objective value after %d steps is %.3f" % (i +1, value))
                else:
                    print("step %d failed" % i)
        print('done')
        self.export_data(name)

    def is_step_valid(self, row, col):
        return True
        # new_value = self.height_field[row, col]
        # for i in range(len(self.raw_images)):
        #     check_col = None
        #     window = None
        #     # check left
        #     if i == 0 and col > self.radius:
        #         check_col = self.height_field[row, col - self.radius - 1]
        #         window = self.height_field[row, col - self.radius: col] - self.buffer_vector
        #     # check right
        #     elif i == 1 and col < self.size - self.radius - 1:
        #         check_col = self.height_field[row, col + self.radius + 1]
        #         window = self.height_field[row, col: col +  self.radius] - np.flip(self.buffer_vector, 0)
        #     # check down
        #     elif i == 2 and row > self.radius:
        #         check_col = self.height_field[row - self.radius - 1, col]
        #         window = self.height_field[row - self.radius: row, col] - self.buffer_vector
        #     # check up
        #     elif row < self.size - self.radius - 1:
        #         check_col = self.height_field[row + self.radius + 1, col]
        #         window = self.height_field[row: row + self.radius, col] - np.flip(self.buffer_vector, 0)
        #     if check_col and new_value - self.radius > check_col:
        #         if np.max(window) < check_col:
        #             return False
        # return True

    def export_mesh(self, path, light_angle):
        s = 1 / np.tan(light_angle * np.pi / 180)
        height_field = self.height_field * s
        height_field -= height_field.min()
        heightfield_to_mesh(height_field, path)


def global_method():
    path_a = './images/hopper_gas.jpg'
    path_b = './images/hockney_chairs.jpg'
    path_c = './images/modigliani_women.jpg'
    path_d = './images/miro.jpg'
    paths = [path_a, path_b, path_c, path_d]
    GBM = GlobalMethod(paths, 200, .1, .1)
    GBM.optimize(1000000, 'global_test')
    GBM.export_data('global_test')
    # GBM.load_data('global_test1')
    # GBM.export_mesh('global_test1.obj', 60)


if __name__ == '__main__':
    global_method()
