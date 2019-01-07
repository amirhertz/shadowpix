# SHADOWPIX

This is a python implementation for the paper <a href="https://people.csail.mit.edu/wojciech/SHADOWPIX/index.html" target="_blank">"SHADOWPIX: Multiple Images from Self Shadowing"</a> by Amit Bermano, Ilya Baran, Marc Alexa and Wojciech Matusk.

Our code includes an implementation for:
- The local method (without chamfers).
- The global method.
- An experimental global deep learning approach.

**Prerequisites**
* python  ≥ 3.5
* <a href="https://pytorch.org/" target="_blank">Pytorch</a> ≥ 0.4

## Quick Start
The simplest way to create a SHADOWPIX is to follow the example scripts below (found within the examples directory).<br/>
For a better understating of the different parameters, please refer to the original paper.

#### Local method
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/shadowpix/local_view_small.png" title="3d local SHDOWPIX object, close view"></a>
```python
from local_method import local_method

# Paths to the images from which we create the shadowpix
path_a = '../images/hopper_gas.jpg'
path_b = '../images/hockney_chairs.jpg'
path_c = '../images/modigliani_woman.jpg'

# The expected light angle from z axis. 
light_angle = 60

# Define the width / height resolution of the SHADOWPIX.
resolution = 200

# The SHADOWPIX obj file will be exported to:
output_file = '../models/paintings.obj'

# Run the algorithm
paths = [path_a, path_b, path_c]
local_method(paths, resolution, light_angle, output_file)
```


#### Global method
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/shadowpix/global_view_small.png" title="3d global SHDOWPIX object, close view"></a>
```python
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
```
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/shadowpix/anim_global_lichtenstein.gif" title="Roy Lichtenstein"></a>
