import numpy as np
import torch
import math

def fill_quad_recursively(vert_mat, r_min, r_max, c_min, c_max):
    if r_min >= r_max-1 or c_min >= c_max-1:
        return vert_mat

    r_center = np.ceil((r_min + r_max) / 2).astype(int)
    c_center = np.ceil((c_min + c_max) / 2).astype(int)

    # diagonal edge
    vert_mat[r_center, c_center, :] = (vert_mat[r_min, c_max, :] + vert_mat[r_max, c_min, :]) / 2
    vert_mat[r_center, c_center, :] = vert_mat[r_center, c_center, :] / np.sqrt(np.sum(vert_mat[r_center, c_center, :]**2))

    # left edge
    vert_mat[r_center, c_min, :] = (vert_mat[r_min, c_min, :] + vert_mat[r_max, c_min, :]) / 2
    vert_mat[r_center, c_min, :] = vert_mat[r_center, c_min, :] / np.sqrt(np.sum(vert_mat[r_center, c_min, :]**2))

    # right edge
    vert_mat[r_center, c_max, :] = (vert_mat[r_min, c_max, :] + vert_mat[r_max, c_max,:]) / 2
    vert_mat[r_center, c_max, :] = vert_mat[r_center, c_max, :] / np.sqrt(np.sum(vert_mat[r_center, c_max, :]**2))

    # top edge
    vert_mat[r_min, c_center, :] = (vert_mat[r_min, c_min, :] + vert_mat[r_min, c_max, :]) / 2
    vert_mat[r_min, c_center, :] = vert_mat[r_min, c_center, :] / np.sqrt(np.sum(vert_mat[r_min, c_center, :]**2))

    # bottom edge
    vert_mat[r_max, c_center, :] = (vert_mat[r_max, c_min, :] + vert_mat[r_max, c_max, :]) / 2
    vert_mat[r_max, c_center, :] = vert_mat[r_max, c_center, :] / np.sqrt(np.sum(vert_mat[r_max, c_center, :]**2))

    # top left quad
    vert_mat = fill_quad_recursively(vert_mat, r_min, r_center, c_min, c_center)

    # top right quad
    vert_mat = fill_quad_recursively(vert_mat, r_min, r_center, c_center, c_max)

    # bottom left quad
    vert_mat = fill_quad_recursively(vert_mat, r_center, r_max, c_min, c_center)

    # bottom right quad
    vert_mat = fill_quad_recursively(vert_mat, r_center, r_max, c_center, c_max)

    return vert_mat


def get_icosahedral_grid(subdivisions):
    # create vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1,  t,  0], [1,  t,  0], [-1, -t,  0], [1, -t,  0],
        [0, -1,  t], [0,  1,  t],  [0, -1, -t],  [0,  1, -t],
        [t,  0, -1], [t,  0,  1], [-t,  0, -1], [-t,  0,  1]])
    v = v / np.sqrt(np.sum(v**2, 1))[np.newaxis].T

    # rotate v4 around x axis so it is at [0, 0, 1]
    angle = np.arctan2(v[4, 1], v[4, 2])
    rx = np.array([[1, 0, 0], [0, np.cos(angle), - np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    v = rx.dot(v.T).T

    # zero out values very close to zero
    tol = 1.e-10
    v[np.abs(v) < tol] = 0

    # reorder vertices for easier map and face creation
    map_ = [4, 9, 5, 11, 2, 3, 1, 0, 10, 6, 8, 7]
    v = v[map_, :]

    # create base faces
    f = np.array([
        [0, 1, 2], [1,  6, 2], [2,  6,  7], [ 6, 11,  7],   # map 0
        [0, 2, 3], [2,  7, 3], [3,  7,  8], [ 7, 11,  8],   # map 1
        [0, 3, 4], [3,  8, 4], [4,  8,  9], [ 8, 11,  9],   # map 2
        [0, 4, 5], [4,  9, 5], [5,  9, 10], [ 9, 11, 10],   # map 3
        [0, 5, 1], [5, 10, 1], [1, 10,  6], [10, 11,  6],   # map 4
    ])

    # create subdivisons and planar mapping
    num_vertical_vertices = 2**subdivisions + 1
    num_horizontal_vertices = 2**(subdivisions + 1) + 1
    vert_mat = np.zeros([5, num_vertical_vertices, num_horizontal_vertices, 3])
    planar_mapping_out = np.zeros([5, num_vertical_vertices, num_horizontal_vertices]).astype(int)

    for i in np.arange(5):
        f_idx = i * 4

        # left quad
        # fill corners of quad
        vert_mat[i, 0, 0, :] = v[f[f_idx, 0], :]
        vert_mat[i, num_vertical_vertices-1, 0, :] = v[f[f_idx, 1], :]
        vert_mat[i, 0, num_vertical_vertices-1, :] = v[f[f_idx, 2], :]

        f_idx = f_idx + 1
        vert_mat[i, num_vertical_vertices-1, num_vertical_vertices-1, :] = v[f[f_idx, 1], :]

        # interpolate other values of quad
        vert_mat[i, :, :, :] = fill_quad_recursively(vert_mat[i, :, :, :], 0, num_vertical_vertices - 1, 0, num_vertical_vertices - 1)

        # right quad
        f_idx = f_idx + 1

        # fill corners of quad
        vert_mat[i, 0, num_vertical_vertices-1, :] = v[f[f_idx, 0], :]
        vert_mat[i, num_vertical_vertices-1, num_vertical_vertices-1, :] = v[f[f_idx, 1], :]
        vert_mat[i, 0, -1, :] = v[f[f_idx, 2], :]

        f_idx = f_idx + 1
        vert_mat[i, -1, -1, :] = v[f[f_idx, 1], :]

        # interpolate other values of quad
        vert_mat[i, :, :, :] = fill_quad_recursively(vert_mat[i, :, :, :], 0, num_vertical_vertices - 1, num_vertical_vertices - 1, num_horizontal_vertices - 1)

    # flip map dimension, so map 4 is top map when ordering on plane
    vert_mat = np.flip(vert_mat, 0)

    # extract vertices and planar mapping
    v_out = np.reshape(vert_mat[:, 1:, 0:-1, :], [-1, 3])
    ind = np.arange((num_vertical_vertices - 1) * 5 * (num_horizontal_vertices - 1))
    planar_mapping_out[:, 1:, 0:-1] = ind.reshape(5, num_vertical_vertices-1, num_horizontal_vertices-1)

    # add vertices at north/south pole
    v_out = np.append(v_out, vert_mat[0, 0, 0, :][np.newaxis], 0)
    planar_mapping_out[:, 0, 0] = v_out.shape[0]-1
    v_out = np.append(v_out, vert_mat[0, -1, -1, :][np.newaxis], 0)
    planar_mapping_out[:, -1, -1] = v_out.shape[0]-1

    # fill in boundaries of the planar mapping
    # next_idx = [1, 2, 3, 4, 0]
    prev_idx = [4, 0, 1, 2, 3]
    # top left
    planar_mapping_out[:, 0, 1:num_vertical_vertices] = planar_mapping_out[prev_idx, 1:num_vertical_vertices, 0]
    # top right
    planar_mapping_out[:, 0, num_vertical_vertices:] = planar_mapping_out[prev_idx, -1, 1:num_vertical_vertices]
    # right
    planar_mapping_out[:, 1:num_vertical_vertices, -1] = planar_mapping_out[prev_idx, -1, num_vertical_vertices:]

    # create faces
    v_idx_1 = planar_mapping_out[:, 0:-1, 0:-1].flatten()
    v_idx_2 = planar_mapping_out[:, 1:, 0:-1].flatten()
    v_idx_3 = planar_mapping_out[:, 0:-1, 1:].flatten()
    v_idx_4 = planar_mapping_out[:, 1:, 1:].flatten()

    f_out = np.concatenate((
        np.stack((v_idx_1, v_idx_2, v_idx_3), 1),
        np.stack((v_idx_2, v_idx_4, v_idx_3), 1)), 0)

    return v_out, f_out


def get_ico_faces(subdivisions):
    assert subdivisions >= 0
    # get faces and mapping
    _, f = get_icosahedral_grid(subdivisions)
    return f


def imgbatch_to_icomesh(data, highlight_boundaries=False):
    # assert if image dimensions wrong
    assert data.dim() == 4
    assert np.log2(data.shape[2]/5) + 1 == np.log2(data.shape[3])
    assert data.shape[1] == 1 or data.shape[1] == 3

    # create corresponding mesh
    subdivisions = np.log2(data.shape[2] / 5).astype(int)
    v, f = get_icosahedral_grid(subdivisions)

    # copy and assign input data to vertices
    v_colors = torch.zeros([data.shape[0], v.shape[0], data.shape[1]])
    v_colors[:, :-2, :] = data.view(data.shape[0], data.shape[1], -1).permute(0, 2, 1)

    # resize data to fit tensorboard's add_mesh function
    if data.shape[1] == 1:
        if highlight_boundaries:
            boundary_map = torch.zeros([data.shape[0], v.shape[0], data.shape[1]])

            # set red channel to 255 for boundary
            boundary_indices = torch.zeros(2**subdivisions * 5, 2**(subdivisions+1), dtype=torch.long)
            boundary_indices[:, [0, 2 ** subdivisions]] = 1
            row_indices = torch.arange(1, 6) * (2**subdivisions) - 1
            boundary_indices[row_indices.long(), :] = 1
            boundary_indices_flat = torch.nonzero(boundary_indices.flatten())

            boundary_map[:, boundary_indices_flat, :] = 255.

            v_colors = torch.cat((boundary_map, v_colors, torch.zeros([data.shape[0], v.shape[0], data.shape[1]])), 2)
        else:
            v_colors = v_colors.expand((-1, -1, 3))
    v = torch.from_numpy(v).unsqueeze(0).expand((data.shape[0], -1, -1))
    f = torch.from_numpy(f).unsqueeze(0).expand((data.shape[0], -1, -1))

    return v, f, v_colors


def get_mesh_from_icomapping(data, targets, max_distance=None, interpolate_poles=True):
    with torch.no_grad():
        # assert if image dimensions wrong
        assert data.dim() == 4
        assert np.log2(data.shape[2] / 5) + 1 == np.log2(data.shape[3])
        assert max_distance is None or data.shape[0] == max_distance.shape[0]

        # create corresponding mesh
        subdivisions = int(math.log2(data.shape[2] / 5))
        v_map, f = get_icosahedral_grid(subdivisions)

        # create distance map for colors
        distance_map = torch.sqrt(torch.sum((data - targets) ** 2, dim=1))
        if max_distance is not None and not torch.any(max_distance < 0):
            for i in range(distance_map.shape[0]):
                distance_map[i, distance_map[i] > max_distance[i]] = max_distance[i]
                distance_map[i] /= max_distance[i]

        # copy and assign input data to vertices
        v = torch.zeros([data.shape[0], v_map.shape[0], 3])
        v_colors = torch.zeros([data.shape[0], v_map.shape[0], 3])
        f = torch.from_numpy(f).unsqueeze(0).expand((data.shape[0], -1, -1))

        v[:, :-2, :] = data.view(data.shape[0], data.shape[1], -1).permute([0, 2, 1])
        v_colors[:, :-2, :] = distance_map.view(distance_map.shape[0], -1, 1)

        if interpolate_poles:
            # interpolate north and southpole vertices
            base_height = 2 ** subdivisions
            top_corner_src_y = torch.arange(5) * base_height
            top_corner_src_x = 0
            bottom_corner_src_y = torch.arange(1, 6) * base_height - 1
            bottom_corner_src_x = -1

            v[:, -2, :] = torch.mean(data[:, :, top_corner_src_y, top_corner_src_x], -1)
            v[:, -1, :] = torch.mean(data[:, :, bottom_corner_src_y, bottom_corner_src_x], -1)
            v_colors[:, -2, 0] = torch.mean(distance_map[:, top_corner_src_y, top_corner_src_x], -1)
            v_colors[:, -1, 0] = torch.mean(distance_map[:, bottom_corner_src_y, bottom_corner_src_x], -1)

        v_colors *= 255.

        return v, f, v_colors
