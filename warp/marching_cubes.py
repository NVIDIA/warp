# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np

import warp as wp


def marching_cubes_extract_vertices(
    field: wp.array3d(dtype=wp.float32),
    threshold: wp.float32,
    domain_bounds_lower_corner: wp.vec3,
    grid_pos_delta: wp.vec3,
):
    """
    Invokes kernels to extract vertices and indices to uniquely identify them.
    """

    device = field.device
    nnode_x, nnode_y, nnode_z = field.shape[0], field.shape[1], field.shape[2]

    ### First pass: count the vertices each thread will generate
    thread_output_count = wp.zeros(shape=(nnode_x * nnode_y * nnode_z * 3), dtype=wp.int32, device=device)
    wp.launch(
        extract_vertices_kernel,
        dim=(nnode_x, nnode_y, nnode_z, 3),
        inputs=[
            field,
            threshold,
            domain_bounds_lower_corner,
            grid_pos_delta,
            None,
            True,  # count only == True : just count the vertices
        ],
        outputs=[thread_output_count, None, None, None],
        device=device,
    )

    ### Evaluate a cumulative sum, to compute the output index for each generated vertex
    vertex_result_ind = wp.zeros(shape=(nnode_x * nnode_y * nnode_z * 3), dtype=wp.int32, device=device)
    wp.utils.array_scan(thread_output_count, vertex_result_ind, inclusive=True)

    # (synchronization point!)
    N_vert = int(vertex_result_ind[-1:].numpy()[0])

    # Allocate output arrays
    # edge_generated_vert_ind: corresponds to the the 3 positive-facing edges emanating from each node.
    #   The last boundary entries of this array will be unused, but we can't make it any smaller.
    edge_generated_vert_ind = wp.zeros(shape=(nnode_x, nnode_y, nnode_z, 3), dtype=wp.int32, device=device)
    verts_pos_out = wp.empty(
        shape=N_vert, dtype=wp.vec3, device=device, requires_grad=field.requires_grad
    )  # TODO is this the right way to decide setting requires_grad?
    verts_is_boundary_out = wp.empty(shape=N_vert, dtype=wp.bool, device=device)

    ### Second pass: actually generate the vertices and write to the output arrays
    wp.launch(
        extract_vertices_kernel,
        dim=(nnode_x, nnode_y, nnode_z, 3),
        inputs=[
            field,
            threshold,
            domain_bounds_lower_corner,
            grid_pos_delta,
            vertex_result_ind,
            False,  # count only == False : actually write out the vertices
        ],
        outputs=[
            None,
            verts_pos_out,
            verts_is_boundary_out,
            edge_generated_vert_ind,
        ],
        device=device,
    )

    return (
        verts_pos_out,
        verts_is_boundary_out,
        edge_generated_vert_ind,
    )


@wp.kernel
def extract_vertices_kernel(
    values: wp.array3d(dtype=wp.float32),
    threshold: wp.float32,
    domain_bounds_lower_corner: wp.vec3,
    grid_pos_delta: wp.vec3,
    vertex_result_ind: wp.array(dtype=wp.int32),
    count_only: bool,
    thread_output_count: wp.array(dtype=wp.int32),
    verts_pos_out: wp.array(dtype=wp.vec3),
    verts_is_boundary_out: wp.array(dtype=wp.bool),
    edge_generated_vert_ind: wp.array(dtype=wp.int32, ndim=4),
):
    """
    Kernel for vertex extraction

    This kernel runs in two different "modes", which share much of their logic.
    In a first pass when count_only==True, we just count the number of vertices that will be
    generated. We then cumulative-sum those counts to generate output indices. Then, in a
    second pass when count_only==False, we actually generate the vertices and write them to
    the appropriate output location.
    """

    ti, tj, tk, t_side = wp.tid()
    nnode_x, nnode_y, nnode_z = values.shape[0], values.shape[1], values.shape[2]

    # Assemble indices
    i_opp = ti + wp.where(t_side == 0, 1, 0)
    j_opp = tj + wp.where(t_side == 1, 1, 0)
    k_opp = tk + wp.where(t_side == 2, 1, 0)
    out_ind = -1

    # Out of bounds edges off the sides of the grid
    if i_opp >= nnode_x or j_opp >= nnode_y or k_opp >= nnode_z:
        if not count_only:
            edge_generated_vert_ind[ti, tj, tk, t_side] = out_ind
        return

    # Fetch values from the field
    this_val = values[ti, tj, tk]
    opp_val = values[i_opp, j_opp, k_opp]

    ind = ti * nnode_y * nnode_z * 3 + tj * nnode_z * 3 + tk * 3 + t_side

    # Check if we generate a vertex
    if (this_val >= threshold and opp_val < threshold) or (this_val < threshold and opp_val >= threshold):
        if count_only:
            thread_output_count[ind] = 1
        else:
            out_ind = vertex_result_ind[ind] - 1

            # generated vertex along the edge
            t_interp = (threshold - this_val) / (opp_val - this_val)
            t_interp = wp.clamp(t_interp, 0.0, 1.0)
            this_pos = domain_bounds_lower_corner + wp.vec3(
                wp.float32(ti) * grid_pos_delta.x,
                wp.float32(tj) * grid_pos_delta.y,
                wp.float32(tk) * grid_pos_delta.z,
            )
            opp_pos = domain_bounds_lower_corner + wp.vec3(
                wp.float32(i_opp) * grid_pos_delta.x,
                wp.float32(j_opp) * grid_pos_delta.y,
                wp.float32(k_opp) * grid_pos_delta.z,
            )
            interp_pos = wp.lerp(this_pos, opp_pos, t_interp)
            this_boundary = (
                ti == 0 or (ti + 1) == nnode_x or tj == 0 or (tj + 1) == nnode_y or tk == 0 or (tk + 1) == nnode_z
            )
            opp_boundary = (
                i_opp == 0
                or (i_opp + 1) == nnode_x
                or j_opp == 0
                or (j_opp + 1) == nnode_y
                or k_opp == 0
                or (k_opp + 1) == nnode_z
            )
            vert_is_boundary = this_boundary and opp_boundary

            # store output data
            verts_pos_out[out_ind] = interp_pos
            verts_is_boundary_out[out_ind] = vert_is_boundary

    if not count_only:
        edge_generated_vert_ind[ti, tj, tk, t_side] = out_ind


def marching_cubes_extract_faces(
    values: wp.array3d(dtype=wp.float32),
    threshold: wp.float32,
    edge_generated_vert_ind: wp.array(dtype=wp.int32, ndim=4),
):
    """
    Invokes kernels to extract faces and index the appropriate vertices.
    """

    device = values.device
    nnode_x, nnode_y, nnode_z = values.shape[0], values.shape[1], values.shape[2]
    ncell_x, ncell_y, ncell_z = nnode_x - 1, nnode_y - 1, nnode_z - 1

    # First pass: count the number of faces each thread will generate
    thread_output_count = wp.zeros(shape=(ncell_x * ncell_y * ncell_z), dtype=wp.int32, device=device)
    wp.launch(
        extract_faces_kernel,
        dim=(ncell_x, ncell_y, ncell_z),
        inputs=[
            values,
            threshold,
            edge_generated_vert_ind,
            None,
            get_mc_case_to_tri_range_table(device),
            get_mc_tri_local_inds_table(device),
            get_mc_edge_offset_table(device),
            True,
        ],
        outputs=[
            thread_output_count,
            None,
        ],
        device=device,
    )

    ### Evaluate a cumulative sum, to compute the output index for each generated face
    face_result_ind = wp.zeros(shape=(ncell_x * ncell_y * ncell_z), dtype=wp.int32, device=device)
    wp.utils.array_scan(thread_output_count, face_result_ind, inclusive=True)

    # (synchronization point!)
    N_faces = int(face_result_ind[-1:].numpy()[0])

    # Allocate output array
    faces_out = wp.empty(shape=3 * N_faces, dtype=wp.int32, device=device)

    ### Second pass: actually generate the faces and write to the output array
    wp.launch(
        extract_faces_kernel,
        dim=(ncell_x, ncell_y, ncell_z),
        inputs=[
            values,
            threshold,
            edge_generated_vert_ind,
            face_result_ind,
            get_mc_case_to_tri_range_table(device),
            get_mc_tri_local_inds_table(device),
            get_mc_edge_offset_table(device),
            False,
        ],
        outputs=[
            None,
            faces_out,
        ],
        device=device,
    )

    return faces_out


# NOTE: differentiating this kernel does nothing, since all of its outputs are discrete, but
# Warp issues warnings if we set enable_backward=False
@wp.kernel
def extract_faces_kernel(
    values: wp.array3d(dtype=wp.float32),
    threshold: wp.float32,
    edge_genererated_vert_ind: wp.array(dtype=wp.int32, ndim=4),
    face_result_ind: wp.array(dtype=wp.int32),
    mc_case_to_tri_range_table: wp.array(dtype=wp.int32),
    mc_tri_local_inds_table: wp.array(dtype=wp.int32),
    mc_edge_offset_table: wp.array(dtype=wp.int32, ndim=2),
    count_only: bool,
    thread_output_count: wp.array(dtype=wp.int32),
    faces_out: wp.array(dtype=wp.int32),
):
    """
    Kernel for face extraction

    This kernel runs in two different "modes", which share much of their logic.
    In a first pass when count_only==True, we just count the number of faces that will be
    generated. We then cumulative-sum those counts to generate output indices. Then, in a
    second pass when count_only==False, we actually generate the faces and write them to
    the appropriate output location.
    """

    ti, tj, tk = wp.tid()
    nnode_x, nnode_y, nnode_z = values.shape[0], values.shape[1], values.shape[2]
    _ncell_x, ncell_y, ncell_z = nnode_x - 1, nnode_y - 1, nnode_z - 1
    ind = ti * ncell_y * ncell_z + tj * ncell_z + tk

    # Check which case we're in
    # NOTE: this loop should get unrolled (confirmed it does in Warp 1.4.1)
    case_code = 0
    for i_c in range(8):
        indX = ti + wp.static(mc_cube_corner_offsets[i_c][0])
        indY = tj + wp.static(mc_cube_corner_offsets[i_c][1])
        indZ = tk + wp.static(mc_cube_corner_offsets[i_c][2])
        val = values[indX, indY, indZ]
        if val >= threshold:
            case_code += wp.static(2**i_c)

    # Gather the range of triangles we will emit
    tri_range_start = mc_case_to_tri_range_table[case_code]
    tri_range_end = mc_case_to_tri_range_table[case_code + 1]
    N_tri = wp.int32(tri_range_end - tri_range_start) // 3

    # If we are just counting, record the number of triangles and move on
    if count_only:
        thread_output_count[ind] = N_tri
        return

    if N_tri == 0:
        return

    # Find the output index for this thread's faces
    # The indexing logic is slightly awkward here because we use an inclusive sum,
    # so we need to check the previous thread's output index, with a special case for
    # the first thread. Doing it this way makes it simpler to fetch N_faces in the host
    # function.
    prev_thread_id = ti * ncell_y * ncell_z + tj * ncell_z + tk - 1
    if prev_thread_id < 0:
        out_ind = 0
    else:
        out_ind = face_result_ind[prev_thread_id]

    # Emit triangles
    for i_tri in range(N_tri):
        for s in range(3):
            local_ind = mc_tri_local_inds_table[tri_range_start + 3 * i_tri + s]

            global_ind = edge_genererated_vert_ind[
                ti + mc_edge_offset_table[local_ind][0],
                tj + mc_edge_offset_table[local_ind][1],
                tk + mc_edge_offset_table[local_ind][2],
                mc_edge_offset_table[local_ind][3],
            ]

            faces_out[3 * (out_ind + i_tri) + s] = global_ind

    return


### Marching Cubes tables

# cached warp device arrays for the tables below
mc_case_to_tri_range_wpcache = {}
mc_tri_local_inds_wpcache = {}
mc_edge_offset_wpcache = {}


def get_mc_case_to_tri_range_table(device):
    """
    Helper to get marching cubes tri range table
    """
    device = str(device)
    if device not in mc_case_to_tri_range_wpcache:
        mc_case_to_tri_range_wpcache[device] = wp.from_numpy(mc_case_to_tri_range_np, dtype=wp.int32, device=device)

    return mc_case_to_tri_range_wpcache[device]


def get_mc_tri_local_inds_table(device):
    """
    Helper to get marching cubes tri local inds table
    """
    device = str(device)
    if device not in mc_tri_local_inds_wpcache:
        mc_tri_local_inds_wpcache[device] = wp.from_numpy(mc_tri_local_inds, dtype=wp.int32, device=device)

    return mc_tri_local_inds_wpcache[device]


def get_mc_edge_offset_table(device):
    """
    Helper to get marching cubes edge offset table
    """
    device = str(device)
    if device not in mc_edge_offset_wpcache:
        mc_edge_offset_wpcache[device] = wp.from_numpy(mc_edge_offset_np, dtype=wp.int32, device=device)

    return mc_edge_offset_wpcache[device]


# fmt: off
mc_case_to_tri_range_np = np.array( [
      0, 0, 3, 6, 12, 15, 21, 27, 36, 39, 45, 51, 60, 66, 75, 84, 90, 93, 99, 105, 114,
      120, 129, 138, 150, 156, 165, 174, 186, 195, 207, 219, 228, 231, 237, 243, 252,
      258, 267, 276, 288, 294, 303, 312, 324, 333, 345, 357, 366, 372, 381, 390, 396,
      405, 417, 429, 438, 447, 459, 471, 480, 492, 507, 522, 528, 531, 537, 543, 552,
      558, 567, 576, 588, 594, 603, 612, 624, 633, 645, 657, 666, 672, 681, 690, 702,
      711, 723, 735, 750, 759, 771, 783, 798, 810, 825, 840, 852, 858, 867, 876, 888,
      897, 909, 915, 924, 933, 945, 957, 972, 984, 999, 1008, 1014, 1023, 1035, 1047,
      1056, 1068, 1083, 1092, 1098, 1110, 1125, 1140, 1152, 1167, 1173, 1185, 1188, 1191,
      1197, 1203, 1212, 1218, 1227, 1236, 1248, 1254, 1263, 1272, 1284, 1293, 1305, 1317,
      1326, 1332, 1341, 1350, 1362, 1371, 1383, 1395, 1410, 1419, 1425, 1437, 1446, 1458,
      1467, 1482, 1488, 1494, 1503, 1512, 1524, 1533, 1545, 1557, 1572, 1581, 1593, 1605,
      1620, 1632, 1647, 1662, 1674, 1683, 1695, 1707, 1716, 1728, 1743, 1758, 1770, 1782,
      1791, 1806, 1812, 1827, 1839, 1845, 1848, 1854, 1863, 1872, 1884, 1893, 1905, 1917,
      1932, 1941, 1953, 1965, 1980, 1986, 1995, 2004, 2010, 2019, 2031, 2043, 2058, 2070,
      2085, 2100, 2106, 2118, 2127, 2142, 2154, 2163, 2169, 2181, 2184, 2193, 2205, 2217,
      2232, 2244, 2259, 2268, 2280, 2292, 2307, 2322, 2328, 2337, 2349, 2355, 2358, 2364,
      2373, 2382, 2388, 2397, 2409, 2415, 2418, 2427, 2433, 2445, 2448, 2454, 2457, 2460,
      2460
    ])

mc_tri_local_inds = np.array([
    0, 8, 3, 0, 1, 9, 1, 8, 3, 9, 8, 1, 1, 2, 10, 0, 8, 3, 1, 2, 10, 9, 2, 10, 0, 2, 9, 2, 8, 3, 2,
    10, 8, 10, 9, 8, 3, 11, 2, 0, 11, 2, 8, 11, 0, 1, 9, 0, 2, 3, 11, 1, 11, 2, 1, 9, 11, 9, 8, 11, 3,
    10, 1, 11, 10, 3, 0, 10, 1, 0, 8, 10, 8, 11, 10, 3, 9, 0, 3, 11, 9, 11, 10, 9, 9, 8, 10, 10, 8, 11, 4,
    7, 8, 4, 3, 0, 7, 3, 4, 0, 1, 9, 8, 4, 7, 4, 1, 9, 4, 7, 1, 7, 3, 1, 1, 2, 10, 8, 4, 7, 3,
    4, 7, 3, 0, 4, 1, 2, 10, 9, 2, 10, 9, 0, 2, 8, 4, 7, 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, 8,
    4, 7, 3, 11, 2, 11, 4, 7, 11, 2, 4, 2, 0, 4, 9, 0, 1, 8, 4, 7, 2, 3, 11, 4, 7, 11, 9, 4, 11, 9,
    11, 2, 9, 2, 1, 3, 10, 1, 3, 11, 10, 7, 8, 4, 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, 4, 7, 8, 9,
    0, 11, 9, 11, 10, 11, 0, 3, 4, 7, 11, 4, 11, 9, 9, 11, 10, 9, 5, 4, 9, 5, 4, 0, 8, 3, 0, 5, 4, 1,
    5, 0, 8, 5, 4, 8, 3, 5, 3, 1, 5, 1, 2, 10, 9, 5, 4, 3, 0, 8, 1, 2, 10, 4, 9, 5, 5, 2, 10, 5,
    4, 2, 4, 0, 2, 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, 9, 5, 4, 2, 3, 11, 0, 11, 2, 0, 8, 11, 4,
    9, 5, 0, 5, 4, 0, 1, 5, 2, 3, 11, 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, 10, 3, 11, 10, 1, 3, 9,
    5, 4, 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, 5, 4, 8, 5,
    8, 10, 10, 8, 11, 9, 7, 8, 5, 7, 9, 9, 3, 0, 9, 5, 3, 5, 7, 3, 0, 7, 8, 0, 1, 7, 1, 5, 7, 1,
    5, 3, 3, 5, 7, 9, 7, 8, 9, 5, 7, 10, 1, 2, 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, 8, 0, 2, 8,
    2, 5, 8, 5, 7, 10, 5, 2, 2, 10, 5, 2, 5, 3, 3, 5, 7, 7, 9, 5, 7, 8, 9, 3, 11, 2, 9, 5, 7, 9,
    7, 2, 9, 2, 0, 2, 7, 11, 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, 11, 2, 1, 11, 1, 7, 7, 1, 5, 9,
    5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, 11, 10, 0, 11,
    0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, 11, 10, 5, 7, 11, 5, 10, 6, 5, 0, 8, 3, 5, 10, 6, 9, 0, 1, 5,
    10, 6, 1, 8, 3, 1, 9, 8, 5, 10, 6, 1, 6, 5, 2, 6, 1, 1, 6, 5, 1, 2, 6, 3, 0, 8, 9, 6, 5, 9,
    0, 6, 0, 2, 6, 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, 2, 3, 11, 10, 6, 5, 11, 0, 8, 11, 2, 0, 10,
    6, 5, 0, 1, 9, 2, 3, 11, 5, 10, 6, 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, 6, 3, 11, 6, 5, 3, 5,
    1, 3, 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, 6, 5, 9, 6,
    9, 11, 11, 9, 8, 5, 10, 6, 4, 7, 8, 4, 3, 0, 4, 7, 3, 6, 5, 10, 1, 9, 0, 5, 10, 6, 8, 4, 7, 10,
    6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, 6, 1, 2, 6, 5, 1, 4, 7, 8, 1, 2, 5, 5, 2, 6, 3, 0, 4, 3,
    4, 7, 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, 3,
    11, 2, 7, 8, 4, 10, 6, 5, 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, 0, 1, 9, 4, 7, 8, 2, 3, 11, 5,
    10, 6, 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, 5,
    1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, 6,
    5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, 10, 4, 9, 6, 4, 10, 4, 10, 6, 4, 9, 10, 0, 8, 3, 10, 0, 1, 10,
    6, 0, 6, 4, 0, 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, 1, 4, 9, 1, 2, 4, 2, 6, 4, 3, 0, 8, 1,
    2, 9, 2, 4, 9, 2, 6, 4, 0, 2, 4, 4, 2, 6, 8, 3, 2, 8, 2, 4, 4, 2, 6, 10, 4, 9, 10, 6, 4, 11,
    2, 3, 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, 6, 4, 1, 6,
    1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, 8, 11, 1, 8, 1, 0, 11,
    6, 1, 9, 1, 4, 6, 4, 1, 3, 11, 6, 3, 6, 0, 0, 6, 4, 6, 4, 8, 11, 6, 8, 7, 10, 6, 7, 8, 10, 8,
    9, 10, 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, 10, 6, 7, 10,
    7, 1, 1, 7, 3, 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7,
    3, 9, 7, 8, 0, 7, 0, 6, 6, 0, 2, 7, 3, 2, 6, 7, 2, 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, 2,
    0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, 11,
    2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, 0, 9, 1, 11,
    6, 7, 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, 7, 11, 6, 7, 6, 11, 3, 0, 8, 11, 7, 6, 0, 1, 9, 11,
    7, 6, 8, 1, 9, 8, 3, 1, 11, 7, 6, 10, 1, 2, 6, 11, 7, 1, 2, 10, 3, 0, 8, 6, 11, 7, 2, 9, 0, 2,
    10, 9, 6, 11, 7, 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, 7, 2, 3, 6, 2, 7, 7, 0, 8, 7, 6, 0, 6,
    2, 0, 2, 7, 6, 2, 3, 7, 0, 1, 9, 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, 10, 7, 6, 10, 1, 7, 1,
    3, 7, 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, 7, 6, 10, 7,
    10, 8, 8, 10, 9, 6, 8, 4, 11, 8, 6, 3, 6, 11, 3, 0, 6, 0, 4, 6, 8, 6, 11, 8, 4, 6, 9, 0, 1, 9,
    4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, 6, 8, 4, 6, 11, 8, 2, 10, 1, 1, 2, 10, 3, 0, 11, 0, 6, 11, 0,
    4, 6, 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, 8,
    2, 3, 8, 4, 2, 4, 6, 2, 0, 4, 2, 4, 6, 2, 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, 1, 9, 4, 1,
    4, 2, 2, 4, 6, 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, 10, 1, 0, 10, 0, 6, 6, 0, 4, 4, 6, 3, 4,
    3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, 10, 9, 4, 6, 10, 4, 4, 9, 5, 7, 6, 11, 0, 8, 3, 4, 9, 5, 11,
    7, 6, 5, 0, 1, 5, 4, 0, 7, 6, 11, 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, 9, 5, 4, 10, 1, 2, 7,
    6, 11, 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, 3, 4, 8, 3,
    5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, 7, 2, 3, 7, 6, 2, 5, 4, 9, 9, 5, 4, 0, 8, 6, 0, 6, 2, 6,
    8, 7, 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, 9,
    5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, 4, 0, 10, 4,
    10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, 6, 9, 5, 6, 11, 9, 11,
    8, 9, 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, 6, 11, 3, 6,
    3, 5, 5, 3, 1, 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1,
    2, 10, 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, 5,
    8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, 9, 5, 6, 9, 6, 0, 0, 6, 2, 1, 5, 8, 1, 8, 0, 5, 6, 8, 3,
    8, 2, 6, 2, 8, 1, 5, 6, 2, 1, 6, 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, 10, 1, 0, 10,
    0, 6, 9, 5, 0, 5, 6, 0, 0, 3, 8, 5, 6, 10, 10, 5, 6, 11, 5, 10, 7, 5, 11, 11, 5, 10, 11, 7, 5, 8,
    3, 0, 5, 11, 7, 5, 10, 11, 1, 9, 0, 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, 11, 1, 2, 11, 7, 1, 7,
    5, 1, 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, 7, 5, 2, 7,
    2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, 2, 5, 10, 2, 3, 5, 3, 7, 5, 8, 2, 0, 8, 5, 2, 8, 7, 5, 10,
    2, 5, 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, 1,
    3, 5, 3, 7, 5, 0, 8, 7, 0, 7, 1, 1, 7, 5, 9, 0, 3, 9, 3, 5, 5, 3, 7, 9, 8, 7, 5, 9, 7, 5,
    8, 4, 5, 10, 8, 10, 11, 8, 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, 0, 1, 9, 8, 4, 10, 8, 10, 11, 10,
    4, 5, 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, 0,
    4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, 9,
    4, 5, 2, 11, 3, 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, 5, 10, 2, 5, 2, 4, 4, 2, 0, 3, 10, 2, 3,
    5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, 8, 4, 5, 8, 5, 3, 3,
    5, 1, 0, 4, 5, 1, 0, 5, 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, 9, 4, 5, 4, 11, 7, 4, 9, 11, 9,
    10, 11, 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, 3, 1, 4, 3,
    4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, 9, 7, 4, 9, 11, 7, 9,
    1, 11, 2, 11, 1, 0, 8, 3, 11, 7, 4, 11, 4, 2, 2, 4, 0, 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, 2,
    9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, 3, 7, 10, 3,
    10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, 1, 10, 2, 8, 7, 4, 4, 9, 1, 4, 1, 7, 7, 1, 3, 4, 9, 1, 4,
    1, 7, 0, 8, 1, 8, 7, 1, 4, 0, 3, 7, 4, 3, 4, 8, 7, 9, 10, 8, 10, 11, 8, 3, 0, 9, 3, 9, 11, 11,
    9, 10, 0, 1, 10, 0, 10, 8, 8, 10, 11, 3, 1, 10, 11, 3, 10, 1, 2, 11, 1, 11, 9, 9, 11, 8, 3, 0, 9, 3,
    9, 11, 1, 2, 9, 2, 11, 9, 0, 2, 11, 8, 0, 11, 3, 2, 11, 2, 3, 8, 2, 8, 10, 10, 8, 9, 9, 10, 2, 0,
    9, 2, 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, 1, 10, 2, 1, 3, 8, 9, 1, 8, 0, 9, 1, 0, 3, 8
    ])

mc_edge_offset_np = np.array([
        [0, 0, 0,  0],
        [1, 0, 0,  1],
        [0, 1, 0,  0],
        [0, 0, 0,  1],

        [0, 0, 1,  0],
        [1, 0, 1,  1],
        [0, 1, 1,  0],
        [0, 0, 1,  1],

        [0, 0, 0,  2],
        [1, 0, 0,  2],
        [1, 1, 0,  2],
        [0, 1, 0,  2]
])

mc_cube_corner_offsets = [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]]
# fmt: on


class MarchingCubes:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        max_verts: int = 0,
        max_tris: int = 0,
        device=None,
        domain_bounds_lower_corner=None,
        domain_bounds_upper_corner=None,
    ):
        """Marching Cubes algorithm to extract a 2D surface mesh from a 3D volume.

        Attributes:
            id: Unique identifier for this object.
            verts (:class:`warp.array`): Array of vertex positions of type :class:`warp.vec3f`
              for the output surface mesh.
              This is populated after running :func:`surface`.
            indices (:class:`warp.array`): Array containing indices of type :class:`warp.int32`
              defining triangles for the output surface mesh.
              This is populated after running :func:`surface`.

              Each set of three consecutive integers in the array represents a single triangle,
              in which each integer is an index referring to a vertex in the :attr:`verts` array.

        Args:
            nx: Number of nodes in the x-direction.
            ny: Number of nodes in the y-direction.
            nz: Number of nodes in the z-direction.
            max_verts: Maximum expected number of vertices (used for array preallocation). (deprecated)
            max_tris: Maximum expected number of triangles (used for array preallocation). (deprecated)
            device (Devicelike): CUDA device on which to run marching cubes and allocate memory.
            domain_bounds_low: Tuple. Lower bound coordinate of the input geometry
            domain_bounds_high: Tuple. Upper bound coordinate of the input geometry

        Raises:
            RuntimeError: ``device`` not a CUDA device.

        .. note::
            The shape of the marching cubes should match the shape of the scalar field being surfaced.

        """

        # Input domain sizes, as number of nodes in the grid (note this is 1 more than the number of cubes)
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # Geometry of the extraction domain
        # (or None, to implicitly use a domain with integer-coordinate nodes)
        self.domain_bounds_lower_corner = domain_bounds_lower_corner
        self.domain_bounds_upper_corner = domain_bounds_upper_corner

        # These are unused, but retained for backwards-compatibility for code which might use them
        self.max_verts = max_verts
        self.max_tris = max_tris

        # Output arrays
        self.verts: wp.array(dtype=wp.vec3f) | None = None
        self.indices: wp.array(dtype=wp.int32) | None = None

        # These are unused, but retained for backwards-compatibility for code which might use them
        self.id = 0
        self.runtime = wp.context.runtime
        self.device = self.runtime.get_device(device)

    def resize(self, nx: int, ny: int, nz: int, max_verts: int = 0, max_tris: int = 0) -> None:
        """Update the expected input and maximum output sizes for the marching cubes calculation.

        This function has no immediate effect on the underlying buffers.
        The new values take effect on the next :func:`surface` call.

        Args:
          nx: Number of nodes in the x-direction.
          ny: Number of nodes in the y-direction.
          nz: Number of nodes in the z-direction.
          max_verts: Deprecated unused argument
          max_tris: Deprecated unused argument
        """
        # actual allocations will be resized on next call to surface()
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def surface(self, field: wp.array(dtype=float, ndim=3), threshold: float) -> None:
        """Compute a 2D surface mesh of a given isosurface from a 3D scalar field.

        The triangles and vertices defining the output mesh are written to the
        :attr:`indices` and :attr:`verts` arrays.

        Args:
          field: Scalar field from which to generate a mesh.
          threshold: Target isosurface value.

        Raises:
          ValueError: ``field`` is not a 3D array.
          ValueError: Marching cubes shape does not match the shape of ``field``.
          RuntimeError: :attr:`max_verts` and/or :attr:`max_tris` might be too small to hold the surface mesh.
        """

        # nx, ny, nz is the number of nodes, which should agree with the size of the field
        assert field.shape[0] == (self.nx)
        assert field.shape[1] == (self.ny)
        assert field.shape[2] == (self.nz)

        verts, faces = self.extract_surface_marching_cubes(
            field=field,
            threshold=wp.float32(threshold),
            domain_bounds_lower_corner=self.domain_bounds_lower_corner,
            domain_bounds_upper_corner=self.domain_bounds_upper_corner,
        )

        self.verts = verts
        self.indices = faces

    @staticmethod
    def extract_surface_marching_cubes(
        field: wp.array3d(dtype=wp.float32),
        threshold: wp.float32 = 0.0,
        domain_bounds_lower_corner: wp.vec3 | None = None,
        domain_bounds_upper_corner: wp.vec3 | None = None,
    ) -> tuple[wp.array(dtype=wp.vec3), wp.array(dtype=wp.int32)]:
        """
        Extract a triangular mesh from a 3d scalar field sampled to a regular grid.

        The shape of the `field` array defines the grid resolution at which extraction
        is performed, and the resolution may differ along each dimension.

        By default, the mesh output coordinates correspond to a grid with integer coordinates 0,1,2,3...
        Alternate, the domain bounds can be specified explicitly to define the output location. For
        example, setting `domain_bounds_lower_corner=wp.vec3(0.0, 0.0, 0.0)` and
        `domain_bounds_upper_corner=wp.vec3(1.0, 1.0, 1.0)` extracts a mesh with vertex coordinates
        in the unit cube, corresponding to a grid with lower-most and upper-most nodes at (0,0,0)
        and (1,1,1) respectively.

        Args:
            field: A 3d array of scalar field field, per-node of a regular grid
            threshold: The value of the isosurface to extract(default: 0.0)
            domain_bounds_lower_corner: The lower corner of the domain to extract (default: None, for a grid with integer coordinates)
            domain_bounds_upper_corner: The upper corner of the domain to extract (default: None, for a grid with integer coordinates)

        Returns:
            tuple[wp.array(dtype=wp.vec3), wp.array(dtype=wp.int32)]:
                A (vertices,triangles) tuple giving the output mesh. The triangles are a flat array of 3*F indices.

        Raises:
            ValueError: ``field`` is not a 3D array.
        """

        # Do some validation
        assert len(field.shape) == 3, "field must be a 3D array"
        assert field.size > 0, "field must be a non-empty array"
        assert field.dtype == wp.float32, "field must be a float32 array"

        # Parse out dimensions, being careful to distinguish between nodes and cells
        nnode_x, nnode_y, nnode_z = field.shape[0], field.shape[1], field.shape[2]
        ncell_x, ncell_y, ncell_z = nnode_x - 1, nnode_y - 1, nnode_z - 1

        # Apply default policies for bounds
        if domain_bounds_lower_corner is None:
            domain_bounds_lower_corner = wp.vec3((0.0, 0.0, 0.0))
        if domain_bounds_upper_corner is None:
            # The default convention is to treat the nodes of the grid as having integer coordinates at 0,1,2,...
            # This means the upper-rightmost node of the grid has coordinates (nnode_x-1, nnode_y-1, nnode_z-1)
            # (which happens to be the same as the number cells, although it may be more confusing to think of it that way)
            domain_bounds_upper_corner = wp.vec3((float(nnode_x - 1), float(nnode_y - 1), float(nnode_z - 1)))

        # quietly allow tuples as input too, although this technically violates
        # the type hinting
        domain_bounds_lower_corner = wp.vec3(domain_bounds_lower_corner)
        domain_bounds_upper_corner = wp.vec3(domain_bounds_upper_corner)

        # Compute the grid spacing
        domain_width = domain_bounds_upper_corner - domain_bounds_lower_corner
        grid_delta = wp.cw_div(domain_width, wp.vec3(ncell_x, ncell_y, ncell_z))

        # Extract the vertices
        # The second output of this kernel is an is-boundary flag for each vertex, which
        # we currently do not expose. (maybe this should be exposed in the future)
        verts, _, edge_generated_vert_ind = marching_cubes_extract_vertices(
            field, threshold, domain_bounds_lower_corner, grid_delta
        )

        # Extract faces between those vertices
        tris = marching_cubes_extract_faces(field, threshold, edge_generated_vert_ind)

        return verts, tris

    def __del__(self):
        return


####################
