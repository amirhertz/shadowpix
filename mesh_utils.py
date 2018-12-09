import numpy as np
import trimesh


def export_mesh(vertices_groups, faces_groups, file):
    with open(file, 'w+') as f:
        for v_group in vertices_groups:
            v_group.shape = (v_group.size // 3, 3)
            for v in v_group:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f_group in faces_groups:
            f_group.shape = (f_group.size // 3, 3)
            f_group += 1
            for face in f_group:
                f.write("f %d %d %d\n" % (face[0], face[1], face[2]))
    # show_mesh(file)


def show_mesh(file):
    mesh = trimesh.load_mesh(file)
    # for facet in mesh.facets:
    #     mesh.visual.face_colors[facet] = trimesh.visual.random_color()
    mesh.show()


def ds_to_mesh(r, u, v, wall_th, file_name):
    rows, cols = u.shape
    cols_helper = np.arange(0, cols, dtype=np.uint32)
    row_helper = np.arange(0, rows, dtype=np.uint32)
    r_vertices = np.zeros([rows, cols - 1, 4, 3])
    u_vertices = np.zeros([rows, cols, 4, 3])
    v_vertices = np.zeros([rows, cols - 1, 4, 3])

    r_faces = np.zeros([rows, cols - 1, 2, 3], dtype=np.uint32)
    u_faces = np.zeros([rows, cols, 2, 3], dtype=np.uint32)

    # set z
    r_vertices[:, :, :, 1] = r[:, :, np.newaxis]
    u_vertices[:, :, :, 1] = u[:, :, np.newaxis]
    v_vertices[:, :, :, 1] = v[:, :, np.newaxis]

    # set x
    r_vertices[:, :, 0, 0] = wall_th + cols_helper[np.newaxis, :-1] * (1 + wall_th)
    r_vertices[:, :, 1, 0] = r_vertices[:, :, 0, 0]
    r_vertices[:, :, 2, 0] = r_vertices[:, :, 0, 0] + 1
    r_vertices[:, :, 3, 0] = r_vertices[:, :, 2, 0]

    u_vertices[:, :, 0, 0] = cols_helper[np.newaxis, :] * (1 + wall_th)
    u_vertices[:, :, 1, 0] = u_vertices[:, :, 0, 0]
    u_vertices[:, :, 2, 0] = u_vertices[:, :, 0, 0] + wall_th
    u_vertices[:, :, 3, 0] = u_vertices[:, :, 2, 0]

    v_vertices[:, :, 0, 0] = r_vertices[:, :, 0, 0]
    v_vertices[:, :, 1, 0] = r_vertices[:, :, 0, 0]
    v_vertices[:, :, 2, 0] = r_vertices[:, :, 2, 0]
    v_vertices[:, :, 3, 0] = r_vertices[:, :, 2, 0]

    # set y
    r_vertices[:, :, 0, 2] = wall_th + row_helper[:, np.newaxis] * (1 + wall_th)
    r_vertices[:, :, 3, 2] = r_vertices[:, :, 0, 2]
    r_vertices[:, :, 1, 2] = r_vertices[:, :, 0, 2] + 1
    r_vertices[:, :, 2, 2] = r_vertices[:, :, 1, 2]

    u_vertices[:, :, 0, 2] = wall_th + row_helper[:, np.newaxis] * (1 + wall_th)
    u_vertices[:, :, 3, 2] = u_vertices[:, :, 0, 2]
    u_vertices[:, :, 1, 2] = u_vertices[:, :, 0, 2] + 1
    u_vertices[:, :, 2, 2] = u_vertices[:, :, 1, 2]

    v_vertices[:, :, 0, 2] = row_helper[:, np.newaxis] * (1 + wall_th)
    v_vertices[:, :, 3, 2] = v_vertices[:, :, 0, 2]
    v_vertices[:, :, 1, 2] = v_vertices[:, :, 0, 2] + wall_th
    v_vertices[:, :, 2, 2] = v_vertices[:, :, 1, 2]

    # faces
    fill = np.array([cols_helper[:-1] * 4, cols_helper[:-1] * 4 + 1, cols_helper[:-1] * 4 + 2])
    r_faces[:, :, 0, :] = np.transpose(fill)[np.newaxis, :, :]
    fill = np.array([cols_helper[:-1] * 4, cols_helper[:-1] * 4 + 2, cols_helper[:-1] * 4 + 3])
    r_faces[:, :, 1, :] = np.transpose(fill)[np.newaxis, :, :]
    r_faces += (row_helper * (cols - 1) * 4)[:, np.newaxis, np.newaxis, np.newaxis]

    fill = np.array([cols_helper * 4, cols_helper * 4 + 1, cols_helper * 4 + 2])
    u_faces[:, :, 0, :] = np.transpose(fill)[np.newaxis, :, :]
    fill = np.array([cols_helper * 4, cols_helper * 4 + 2, cols_helper * 4 + 3])
    u_faces[:, :, 1, :] = np.transpose(fill)[np.newaxis, :, :]
    u_faces += (row_helper * cols * 4)[:, np.newaxis, np.newaxis, np.newaxis] + (v_vertices.size + r_vertices.size) // 3
    v_faces = r_faces + r_vertices.size // 3
    export_mesh([r_vertices, u_vertices, v_vertices], [r_faces, u_faces, v_faces], file_name)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # TESTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
    col_s = 100
    row_s = 100
    r_a = np.random.rand(row_s, col_s)
    u_a = np.random.rand(row_s, col_s + 1)
    v_a = np.random.rand(row_s, col_s)
    ds_to_mesh(r_a, u_a, v_a, 0.3, './test.obj')
