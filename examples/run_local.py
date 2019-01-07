from local_method import local_method

# Paths to the images from which we create the shadowpix
path_a = '../images/hopper_gas.jpg'
path_b = '../images/hockney_chairs.jpg'
path_c = '../images/modigliani_woman.jpg'

light_angle = 60  # The expected light angle from z axis.
resolution = 200  # Define the width / height resolution of the SHADOWPIX.

output_file = '../models/paintings.obj'  # The SHADOWPIX obj file will be exported to here

# Run the algorithm
paths = [path_a, path_b, path_c]
local_method(paths, resolution, light_angle, output_file)