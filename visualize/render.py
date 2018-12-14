import os

import numpy as np
import bpy
import bmesh
from mathutils import Vector, Euler
import sys

'''
@input: 
    <obj_file>
    <edges_file> list of edges
    <eseg_file> list of edge seg ids
    <outfile> name of rendered image file
    <freestyle.blend> blender file with basic rendering set up
    
@output:
    adds colors to the edges based on the segmentation ids
    to run it from cmd line:
    /opt/blender/blender <freestyle.blend> --background --python seg_render.py -- '<obj_file>' '<outfile>'
'''

# /opt/blender/blender /home/rana/code/meshnet/dataPrep/process_models/freestyle4.blend --background --python ~/code/meshnet/dataPrep/process_models/seg_render.py -- '/mnt/data/datasets/coseg/seg/vase/rp_1000/0074.obj' '/home/rana/Downloads/blabla.png'


# class Config():
#
#     def __init__(self, set):
#         self.type = set
#         self.create()
#     def create(self):
#         method_to_call = getattr(self, self.type)
#         result = method_to_call()
#     def vase(self):
#         self.rotation = None
#         self.scale = 2
#         self.cam_rot = Euler((1.477233648300171, -4.4167700252728537e-07, 1.1572351455688477), 'XYZ')
#         self.cam_loc = Vector((6.113453388214111, -3.049792528152466, 1.689035177230835))
#
#     def alien(self):
#         self.rotation = np.pi/4
#         self.scale = 2.7
#         self.cam_rot = Euler((1.477233648300171, -4.4167700252728537e-07, 1.1572351455688477), 'XYZ')
#         self.cam_loc = Vector((6.113453388214111, -3.049792528152466, 1.689035177230835))
#
#     def chair(self):
#         self.rotation = None
#         self.scale = 2
#         self.cam_rot = Euler((1.1762362718582153, -9.636925142331165e-07, 1.157235860824585), 'XYZ')
#         self.cam_loc = Vector((3.89676570892334, -2.0769591331481934, 2.873819351196289))


class MeshDS():
    def __init__(self, mesh=None):
        self.mesh = mesh
        self.vertices = None
        if self.mesh is not None:
            self.set_attr()
    def set_attr(self):
        self.vertices = [(self.mesh.matrix_world * v.co) for v in self.mesh.data.vertices]
        self.edge_map = {ek: self.mesh.data.edges[i] for i, ek in enumerate(self.mesh.data.edge_keys)}
        #
        bm = bmesh.new()
        bm.from_mesh(self.mesh.data)
        self.bm = bm
        self.edges = [e for e in bm.edges]
        #
        self.edges_faces = {k: [] for k in range(len(self.edge_map))}
        for f in bm.faces:
            for i in range(3):
                if f.index not in self.edges_faces[f.edges[i].index]:
                    self.edges_faces[f.edges[i].index].append(f.index)
    def get_vertices(self, mesh=None):
        if self.vertices is not None:
            return self.vertices
        elif mesh is not None:
            return [(mesh.matrix_world * v.co) for v in mesh.data.vertices]

class BlenderOps():
    def __init__(self):
        self.obs = {}
    def load_obj(self, obj_file):
        bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_edges=False,
                                 use_smooth_groups=True, use_split_objects=False, use_split_groups=False,
                                 use_groups_as_vgroups=False, use_image_search=True, split_mode='ON',
                                 global_clamp_size=0)
        ob = bpy.context.selected_objects[0]
        self.obs[ob.name] = ob
        return ob
    def copy_obj(self, obj):
        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        new_obj.animation_data_clear()
        scn = bpy.context.scene
        scn.objects.link(new_obj)
        return new_obj
    def unit_scale(self, mesh):
        mesh.dimensions = mesh.dimensions / max(mesh.dimensions)
    def scale(self, mesh, fac):
        mesh.scale = mesh.scale * fac
    def shift_z(self, mesh, target=1e-6, input_mlocz=None):
        mesh.data.update()
        bpy.context.scene.update()
        _mds = MeshDS()
        vertices = np.array(_mds.get_vertices(mesh))
        if input_mlocz:
            mlocz = input_mlocz
        else:
            mlocz = mesh.location.z
        mesh.location.z = mlocz + (target - min(vertices[:, 2]))
        return mlocz
    def translate(self, mesh, new_loc=Vector((0, 0, 0))):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        loc = mesh.location
        mesh.location = new_loc
        return loc
    def rotate(self, mesh, val=None, axis='z'):
        if val is not None:
            setattr(mesh.rotation_euler, axis, val)
    def deselect(self):
        for obj in bpy.data.objects:
            obj.select = False
    def select_only(self, obj):
        self.deselect()
        obj.select = True
    def render(self, filename):
        bpy.data.scenes['Scene'].render.filepath = filename
        bpy.ops.render.render(write_still=True)

class Material():
    def create_material(self, dict_attr):
        # for example: dict_attr = {'name': 'color', 'line_color': [0, 0.5, 0.1, 1]}
        mat = bpy.data.materials.new(name="Material")
        for key, value in dict_attr.items():
            setattr(mat, key, value)
        return mat
    def get_material_index(self, mesh, mat_name):
        mat_names = mesh.data.materials.keys()
        return mat_names.index(mat_name)
    def add_material(self, mat, mesh):
        mesh.data.materials.append(mat)
        mat_idx = self.get_material_index(mesh, mat.name)
        return mat_idx
    def assign_face_material(self, mesh, faces, material_index):
        bpy.context.scene.objects.active = mesh
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(mesh.data)
        bm.faces.ensure_lookup_table() #todo delete?
        bm.select_mode = {'FACE'}
        for f in faces:
            bm.faces[f].select_set(True)
        mesh.data.update()
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(mesh.data)
        for f in faces:
            bm.faces[f].material_index = material_index
        mesh.data.update()
        bpy.ops.object.mode_set(mode='OBJECT')
    def paint_faces(self, mesh, faces, dict_attr):
        mat = self.create_material(dict_attr)
        mat_idx = self.add_material(mat, mesh)
        self.assign_face_material(mesh, faces, mat_idx)

class PaintEdges():
    def __init__(self, obj_file):
        self.obj_file = obj_file
        # self.render_settings()
    def render_settings(self):
        bpy.data.scenes['Scene'].render.use_freestyle = True
        bpy.data.scenes['Scene'].render.engine = 'CYCLES'
        bpy.data.scenes['Scene'].use_nodes = True
        # lighting
        # bpy.data.objects['light'].data.materials['Material'].node_tree.nodes['Emission'].inputs[
        #     'Strength'].default_value = 1.8
        # self.set_cam()
    def set_cam(self):
        cam = bpy.data.objects['Camera']
        cam.location =  Vector((3.89676570892334, -2.0769591331481934, 2.873819351196289))
        cam.rotation_euler = Euler((1.1762362718582153, -9.636925142331165e-07, 1.157235860824585), 'XYZ')
    def load_file(self):
        bpy.ops.import_scene.obj(filepath=self.obj_file)
    def paint_edges(self, filename=None, gif_frames=0):
        if filename is None:
            return
        bpy.data.scenes['Scene'].render.filepath = filename
        bpy.ops.render.render(write_still=True)
        # mat = Material()
        # mds = MeshDS(mesh)
        # mat.paint_faces(mesh, faces, mat_attr)
        # for seg_id in range(self.nseg):
        #     mesh_i = ops.copy_obj(mesh)
        #     mds = MeshDS(mesh_i)
        #     faces = self.mark_edges_with_seg(seg_id, mds)
        #     mat_attr = {'name': 'seg_{}'.format(seg_id), 'line_color': self.color[seg_id]}
        #     mat.paint_faces(mesh_i, faces, mat_attr)
        #     mesh_is.append(mesh_i)
        # normalize everything...
        # t = None
        # sz = None
        # for _m in mesh_is:
        #     ops.unit_scale(_m)
        #     ops.scale(_m, self.cfg.scale)
        #     if t is None:
        #         t = ops.translate(_m)
        #     else:
        #         ops.translate(_m, t)
        #     if sz is None:
        #         sz = ops.shift_z(_m)
        #     else:
        #         ops.shift_z(_m, input_mlocz=sz)
        #     if self.cfg.rotation:
        #         ops.rotate(_m, self.cfg.rotation)
        # ops.rotate(bpy.data.objects['light'], np.pi)
        # ops.render(filename)

# if __name__ == '__main__':
# id = '0074'
# obj_file = '/mnt/data/datasets/coseg/seg/vase/rp_1000/{}.obj'.format(id)
# edges_file = '/mnt/data/datasets/coseg/seg/vase/rp_1000/{}.edges'.format(id)
# eseg_file = '/mnt/data/datasets/coseg/seg/vase/rp_1000/seg/{}.eseg'.format(id)
# paint = PaintEdges(obj_file, edges_file, eseg_file)
# paint.paint_edges()

# id = '0074'
#obj_file = '/home/rana/code/meshnet/checkpoints/unet_deeper_1500_1200_900_600/fav_seg/0135_0.obj'
#paint = PaintEdges(obj_file)
#paint.paint_edges()

obj_file = '/home/amir/projects/shadowpix/models/test.obj'
outfile = '/home/amir/projects/shadowpix/renders/test.png'
# obj_file = sys.argv[-2]
# outfile = sys.argv[-1]

setup = PaintEdges(obj_file)
setup.paint_edges(outfile)