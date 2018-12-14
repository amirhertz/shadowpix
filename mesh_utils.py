import numpy as np
# from mayavi import mlab


def reshape_array(arrays_group):
    for np_array in arrays_group:
        np_array.shape = (np_array.size // 3, 3)
    return np.concatenate(arrays_group, axis=0)


def export_mesh(vertices, faces, file, center=True, scale=1):
    ranges = np.zeros(3)
    if center:
        for i in range(3):
            min_value = np.min(vertices[:, i])
            ranges[i] = np.max(vertices[:, i]) - min_value
            if center:
                vertices[:, i] -= min_value
                if i != 1:
                    vertices[:, i] -= ranges[i] / 2
    if scale:
        vertices[:, :] /= (np.max(ranges) / scale)
    faces += 1
    with open(file, 'w+') as f:
        for v in vertices:
            f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0], face[1], face[2]))
    # show_mesh(file)


def side_surface(top_ind, bottom_ind, middle_ind):
    if middle_ind[0] is None:
        if middle_ind[1] is None:
            return [[top_ind[0], top_ind[1], bottom_ind[1]], [top_ind[0], bottom_ind[1], bottom_ind[0]]]
        else:
            return [[top_ind[0], top_ind[1], middle_ind[1]], [top_ind[0], middle_ind[1], bottom_ind[0]],
                    [middle_ind[1], bottom_ind[1], bottom_ind[0]]]
    else:
        if middle_ind[1] is None:
            return [[top_ind[0], top_ind[1], middle_ind[0]], [top_ind[1], bottom_ind[1], middle_ind[0]],
                    [middle_ind[0], bottom_ind[1], bottom_ind[0]]]
        else:
            return [[top_ind[0], top_ind[1], middle_ind[0]], [top_ind[1], middle_ind[1], middle_ind[0]],
                    [middle_ind[0], middle_ind[1], bottom_ind[0]], [middle_ind[1], bottom_ind[1], bottom_ind[0]]]


def get_side_faces(vertices, r_faces, u_faces, v_faces):
    side_faces = []
    for row_id, row in enumerate(u_faces):
        for element_id, u_element in enumerate(row):
            if element_id < len(row) - 1:
                top_ind = [u_element[1, 2], u_element[1, 1]]
                bottom_ind = [r_faces[row_id, element_id, 0, 0], r_faces[row_id, element_id, 0, 1]]
                middle_ind = [v_faces[row_id, element_id, 0, 1], None]
                if row_id < len(u_faces) - 1:
                    middle_ind[1] = v_faces[row_id + 1, element_id, 0, 0]
                for i in range(2):
                    if middle_ind[i] is not None and vertices[middle_ind[i], 1] > vertices[top_ind[i], 1]:
                        middle_ind[i] = None
                side_faces += side_surface(top_ind, bottom_ind, middle_ind)
            if element_id != 0:
                top_ind = [u_element[0, 1], u_element[0, 0]]
                bottom_ind = [r_faces[row_id, element_id - 1, 1, 1], r_faces[row_id, element_id - 1, 1, 2]]
                middle_ind = [None, v_faces[row_id, element_id - 1, 1, 1]]
                if row_id < len(u_faces) - 1:
                    middle_ind[0] = v_faces[row_id + 1, element_id - 1, 1, 2]
                for i in range(2):
                    if middle_ind[i] is not None and vertices[middle_ind[i], 1] > vertices[top_ind[i], 1]:
                        middle_ind[i] = None
                side_faces += side_surface(top_ind, bottom_ind, middle_ind)
    for row_id, row in enumerate(v_faces):
        for element_id, v_element in enumerate(row):
            top_ind = [v_element[0, 2], v_element[0, 1]]
            bottom_ind = [r_faces[row_id, element_id, 1, 2], r_faces[row_id, element_id, 1, 0]]
            middle_ind = [u_faces[row_id, element_id + 1, 0, 0], u_faces[row_id, element_id, 1, 2]]
            for i in range(2):
                if middle_ind[i] is not None and vertices[middle_ind[i], 1] > vertices[top_ind[i], 1]:
                    middle_ind[i] = None
            side_faces += side_surface(top_ind, bottom_ind, middle_ind)
            if row_id != 0:
                top_ind = [v_element[1, 0], v_element[1, 2]]
                bottom_ind = [r_faces[row_id - 1, element_id, 0, 1], r_faces[row_id - 1, element_id, 0, 2]]
                middle_ind = [u_faces[row_id - 1, element_id, 0, 2], u_faces[row_id - 1, element_id + 1, 0, 1]]
                for i in range(2):
                    if middle_ind[i] is not None and vertices[middle_ind[i], 1] > vertices[top_ind[i], 1]:
                        middle_ind[i] = None
                side_faces += side_surface(top_ind, bottom_ind, middle_ind)
    return side_faces


def get_top_side_faces(vertices, b_faces, u_faces, v_faces):
    top_side_faces = []
    for row_id, row in enumerate(u_faces):
        for element_id, u_element in enumerate(row):
            top_ind = [b_faces[row_id, element_id, 0, 2], b_faces[row_id, element_id, 0, 1]]
            bottom_ind = [u_element[1, 2], u_element[1, 0]]
            middle_ind = [None, None]
            if element_id < len(row) - 1:
                middle_ind[0] = v_faces[row_id, element_id, 0, 1]
            if element_id != 0:
                middle_ind[1] = v_faces[row_id, element_id - 1, 0, 2]
            for i in range(2):
                if middle_ind[i] is not None and vertices[middle_ind[i], 1] > vertices[bottom_ind[i], 1]:
                    middle_ind[i] = None
            top_side_faces += side_surface(top_ind, bottom_ind, middle_ind)
            if row_id < len(u_faces) - 1:
                top_ind = [b_faces[row_id + 1, element_id, 0, 0], b_faces[row_id + 1, element_id, 1, 2]]
                bottom_ind = [u_element[0, 1], u_element[0, 2]]
                middle_ind = [None, None]
                if element_id != 0:
                    middle_ind[0] = v_faces[row_id + 1, element_id - 1, 1, 2]
                if element_id < len(row) - 1:
                    middle_ind[1] = v_faces[row_id + 1, element_id, 0, 0]
                for i in range(2):
                    if middle_ind[i] is not None and vertices[middle_ind[i], 1] > vertices[bottom_ind[i], 1]:
                        middle_ind[i] = None
                top_side_faces += side_surface(top_ind, bottom_ind, middle_ind)
    for row_id, row in enumerate(v_faces):
        for element_id, v_element in enumerate(row):
            top_ind = [b_faces[row_id, element_id, 1, 2], b_faces[row_id, element_id, 1, 1]]
            bottom_ind = [v_element[0, 0], v_element[0, 1]]
            middle_ind = [None, u_faces[row_id, element_id, 1, 2]]
            if row_id != 0:
                middle_ind[0] = u_faces[row_id - 1, element_id, 1, 1]
            for i in range(2):
                if middle_ind[i] is not None and vertices[middle_ind[i], 1] > vertices[bottom_ind[i], 1]:
                    middle_ind[i] = None
            top_side_faces += side_surface(top_ind, bottom_ind, middle_ind)
            top_ind = [b_faces[row_id, element_id + 1, 0, 1], b_faces[row_id, element_id + 1, 0, 0]]
            bottom_ind = [v_element[1, 1], v_element[1, 2]]
            middle_ind = [u_faces[row_id, element_id + 1, 0, 0], None]
            if row_id != 0:
                middle_ind[1] = u_faces[row_id - 1, element_id + 1, 0, 1]
            for i in range(2):
                if middle_ind[i] is not None and vertices[middle_ind[i], 1] > vertices[bottom_ind[i], 1]:
                    middle_ind[i] = None
            top_side_faces += side_surface(top_ind, bottom_ind, middle_ind)
    return top_side_faces

# def show_mesh(vs, fs):
#     s = mlab.triangular_mesh(vs[:, 0], vs[:, 2], vs[:, 1], fs, color=(1, 1, 1))
#     mlab.show()


def ds_to_mesh(r, u, v, wall_th, file_name):
    rows, cols = u.shape
    cols_helper = np.arange(0, cols, dtype=np.uint32)
    row_helper = np.arange(0, rows, dtype=np.uint32)
    r_vertices = np.zeros([rows, cols - 1, 4, 3])
    u_vertices = np.zeros([rows, cols, 4, 3])
    v_vertices = np.zeros([rows, cols - 1, 4, 3])
    b_vertices = np.zeros([rows, cols, 4, 3])
    middle_heights = np.zeros([rows, cols, 4, 4])

    r_faces = np.zeros([rows, cols - 1, 2, 3], dtype=np.uint32)
    u_faces = np.zeros([rows, cols, 2, 3], dtype=np.uint32)

    # set z
    r_vertices[:, :, :, 1] = r[:, :, np.newaxis]
    u_vertices[:, :, :, 1] = u[:, :, np.newaxis]
    v_vertices[:, :, :, 1] = v[:, :, np.newaxis]

    middle_heights[:, :, :, 0] = u_vertices[:, :, :, 1]
    middle_heights[:, :cols - 1, :, 1] = v_vertices[:, :, :, 1]
    middle_heights[1:, :, :, 2] = u_vertices[:rows - 1, :, :, 1]
    middle_heights[:, 1:, :, 3] = v_vertices[:, :, :, 1]
    b_vertices[:, :, :, 1] = np.max(middle_heights, axis=3)

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

    b_vertices[:, :, :, 0] = u_vertices[:, :, :, 0]

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

    b_vertices[:, :, 0, 2] = u_vertices[:, :, 0, 2] - wall_th
    b_vertices[:, :, 3, 2] = b_vertices[:, :, 0, 2]
    b_vertices[:, :, 1, 2] = u_vertices[:, :, 0, 2]
    b_vertices[:, :, 2, 2] = b_vertices[:, :, 1, 2]

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
    b_faces = u_faces + u_vertices.size // 3
    vertices = reshape_array([r_vertices, v_vertices, u_vertices, b_vertices])
    side_faces = np.array(get_side_faces(vertices, r_faces, u_faces, v_faces), dtype=np.uint32)
    top_side_faces = np.array(get_top_side_faces(vertices, b_faces, u_faces, v_faces), dtype=np.uint32)
    faces = reshape_array([r_faces, u_faces, v_faces, b_faces, side_faces, top_side_faces])
    # show_mesh(vertices, faces)
    export_mesh(vertices, faces, file_name)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # TESTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
    col_s = 8
    row_s = 6
    r_a = np.random.rand(row_s, col_s)
    u_a = np.random.rand(row_s, col_s + 1) + 1
    v_a = np.random.rand(row_s, col_s) + 1
    ds_to_mesh(r_a, u_a, v_a, 0.1, './test.obj')
