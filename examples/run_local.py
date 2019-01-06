from local_method import local_method


path_a = '../images/hopper_gas.jpg'
path_b = '../images/hockney_chairs.jpg'
path_c = '../images/modigliani_woman.jpg'

light_angle = 60
resolution = 200
output_file = '../models/tmp.obj'

paths = [path_a, path_b, path_c]
local_method(paths, resolution, light_angle, output_file)