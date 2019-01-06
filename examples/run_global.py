import torch
from global_method import GlobalMethod

# path params
path_a = '../images/roy_a.jpg'
path_b = '../images/roy_b.jpg'
path_c = '../images/roy_c.jpg'
path_d = '../images/roy_d.jpg'
output_name = '../models/lichtenstein'


# object params
light_angle = 60
resolution = 400

# optimization parameters
steps = 10000000
temperature = 100
radius = 20
w_g = 0.1  # gradient image weight
w_s = 0.05   # heightfield smoothness weight

# choose gpu or cpu
device = torch.device('cuda:0')

# initialize the global method and optimize
paths = [path_a, path_b, path_c, path_d]
gbm = GlobalMethod(paths, resolution, 0, 0, radius=radius, device=device)
gbm.optimize(steps, output_name, temperature=temperature)

# load global method state and export to mesh
gbm.load_data(output_name)
gbm.export_mesh(output_name, light_angle)

