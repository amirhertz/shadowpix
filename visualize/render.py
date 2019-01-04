import os
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Euler
import sys


def rotate(mesh, val=None, axis='z'):
    if val is not None:
        setattr(mesh.rotation_euler, axis, val)


def render(filename):
    bpy.data.scenes['Scene'].render.filepath = filename
    bpy.ops.render.render(write_still=True)

obj_file = '/home/amir/projects/shadowpix/models/.obj'
outfile = '/home/amir/projects/shadowpix/renders/mies.png'
gif_frames = 0
# obj_file = sys.argv[-3]
# outfile = sys.argv[-2]
# gif_frames = int(sys.argv[-1])
bpy.ops.import_scene.obj(filepath=obj_file)
mat = bpy.data.materials.new(name="Material")
sun = bpy.data.objects['Sun']
mesh = bpy.context.selected_objects[0]
mesh.data.materials.append(mat)

gif_frames = 20
if gif_frames > 1:
    file_id, file_extension = os.path.splitext(os.path.basename(outfile))
    output_folder = os.path.dirname(outfile)
    for i in range(gif_frames):
        output_file = os.path.join(output_folder, '%s_%d%s' % (file_id, i, file_extension))
        rotate(sun,  -i * np.pi / (gif_frames-1))
        render(output_file)
else:
    render(outfile)






