import torch
from global_method import GlobalMethod

# images paths
path_a = '../images/roy_a.jpg'
path_b = '../images/roy_b.jpg'
path_c = '../images/roy_c.jpg'
path_d = '../images/roy_d.jpg'
output_name = '../models/lichtenstein'  # The SHADOWPIX obj and the checkpoints file will be exported to here

light_angle = 60  # The expected light angle from z axis.
resolution = 400  # Define the width / height resolution of the SHADOWPIX.

# optimization parameters
steps = 10000000
temperature = 100
radius = 20
w_g = 0.1  # gradient image weight
w_s = 0.05   # heightfield smoothness weight

device = torch.device('cuda:0')  # choose gpu or cpu

# initialize the global method and start the optimization
paths = [path_a, path_b, path_c, path_d]
gbm = GlobalMethod(paths, resolution, w_g, w_s, radius=radius, device=device)
gbm.optimize(steps, output_name, temperature=temperature)

# load global method state and export to mesh
gbm.load_data(output_name)
gbm.export_mesh(output_name, light_angle)

