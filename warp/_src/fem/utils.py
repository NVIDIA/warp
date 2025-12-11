# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional, Union

import numpy as np

import warp as wp
import warp._src.fem.cache as cache
from warp._src.fem.linalg import array_axpy, inverse_qr, symmetric_eigenvalues_qr  # noqa: F401
from warp._src.fem.types import NULL_NODE_INDEX
from warp._src.types import scalar_types, type_is_matrix
from warp._src.utils import array_scan, radix_sort_pairs, runlength_encode

_wp_module_name_ = "warp.fem.utils"


def type_zero_element(dtype):
    suffix = cache.pod_type_key(dtype)

    if dtype in scalar_types:

        @cache.dynamic_func(suffix=suffix)
        def zero_element():
            return dtype(0.0)

        return zero_element

    @cache.dynamic_func(suffix=suffix)
    def zero_element():
        return dtype()

    return zero_element


def type_basis_element(dtype):
    suffix = cache.pod_type_key(dtype)

    if dtype in scalar_types:

        @cache.dynamic_func(suffix=suffix)
        def basis_element(coord: int):
            return dtype(1.0)

        return basis_element

    if type_is_matrix(dtype):
        cols = dtype._shape_[1]

        @cache.dynamic_func(suffix=suffix)
        def basis_element(coord: int):
            v = dtype()
            i = coord // cols
            j = coord - i * cols
            v[i, j] = v.dtype(1.0)
            return v

        return basis_element

    @cache.dynamic_func(suffix=suffix)
    def basis_element(coord: int):
        v = dtype()
        v[coord] = v.dtype(1.0)
        return v

    return basis_element


def compress_node_indices(
    node_count: int,
    node_indices: wp.array(dtype=int),
    return_unique_nodes=False,
    node_offsets: wp.array(dtype=int) = None,
    sorted_array_indices: wp.array(dtype=int) = None,
    unique_node_count: wp.array(dtype=int) = None,
    unique_node_indices: wp.array(dtype=int) = None,
    temporary_store: cache.TemporaryStore = None,
) -> Union[tuple[cache.Temporary, cache.Temporary], tuple[cache.Temporary, cache.Temporary, int, cache.Temporary]]:
    """
    Compress an unsorted list of node indices into:
     - the `node_offsets` array, giving for each node the start offset of corresponding indices in sorted_array_indices
     - the `sorted_array_indices` array, listing the indices in the input array corresponding to each node

    Plus if `return_unique_nodes` is ``True``,
     - the `unique_node_count` array containing the number of unique node indices
     - the `unique_node_indices` array containing the sorted list of unique node indices (i.e. the list of indices i for which node_offsets[i] < node_offsets[i+1])

    Node indices equal to NULL_NODE_INDEX will be ignored

    If the ``node_offsets``, ``sorted_array_indices``, ``unique_node_count`` and ``unique_node_indices`` arrays are provided and adequately shaped, they will be used to store the results instead of creating new arrays.

    """

    index_count = node_indices.size
    device = node_indices.device

    with wp.ScopedDevice(device):
        sorted_node_indices = cache.borrow_temporary(temporary_store, shape=2 * index_count, dtype=int)

        if sorted_array_indices is None or sorted_array_indices.shape != sorted_node_indices.shape:
            sorted_array_indices = cache.borrow_temporary_like(sorted_node_indices, temporary_store)

        indices_per_element = 1 if node_indices.ndim == 1 else node_indices.shape[-1]
        wp.launch(
            kernel=_prepare_node_sort_kernel,
            dim=index_count,
            inputs=[node_indices.flatten(), sorted_node_indices, sorted_array_indices, indices_per_element],
        )

        # Sort indices
        radix_sort_pairs(sorted_node_indices, sorted_array_indices, count=index_count)

        # Build prefix sum of number of elements per node
        node_element_counts = cache.borrow_temporary(temporary_store, shape=index_count, dtype=int)
        if unique_node_indices is None or unique_node_indices.shape != node_element_counts.shape:
            unique_node_indices = cache.borrow_temporary_like(node_element_counts, temporary_store)

        if unique_node_count is None or unique_node_count.shape != (1,):
            unique_node_count = cache.borrow_temporary(temporary_store, shape=(1,), dtype=int)

        runlength_encode(
            sorted_node_indices,
            unique_node_indices,
            node_element_counts,
            value_count=index_count,
            run_count=unique_node_count,
        )

        # Scatter seen run counts to global array of element count per node
        if node_offsets is None or node_offsets.shape != (node_count + 1,):
            node_offsets = cache.borrow_temporary(temporary_store, shape=(node_count + 1), dtype=int)

        node_offsets.zero_()
        wp.launch(
            kernel=_scatter_node_counts,
            dim=node_count + 1,  # +1 to accommodate possible NULL node,
            inputs=[node_element_counts, unique_node_indices, node_offsets, unique_node_count],
        )

        # Prefix sum of number of elements per node
        array_scan(node_offsets, node_offsets, inclusive=True)

        sorted_node_indices.release()
        node_element_counts.release()

        if not return_unique_nodes:
            return node_offsets, sorted_array_indices

        return node_offsets, sorted_array_indices, unique_node_count, unique_node_indices


def host_read_at_index(array: wp.array, index: int = -1, temporary_store: cache.TemporaryStore = None) -> int:
    """Returns the value of the array element at the given index on host"""

    if index < 0:
        index += array.shape[0]
    return array[index : index + 1].numpy()[0]


def masked_indices(
    mask: wp.array,
    missing_index: int = -1,
    max_index_count: int = -1,
    local_to_global: Optional[wp.array] = None,
    global_to_local: Optional[wp.array] = None,
    temporary_store: cache.TemporaryStore = None,
) -> tuple[wp.array, wp.array]:
    """
    From an array of boolean masks (must be either 0 or 1), returns:
      - Local to global map: The list of indices for which the mask is 1
      - Global to local map: A map associating to each element of the input mask array its local index if non-zero, or missing_index if zero.

    If ``max_index_count`` is provided, it will be used to limit the number of indices returned instead of synchronizing back to the host

    If ``local_to_global`` and ``global_to_local`` are provided and adequately sized, they will be used to store the indices instead of creating new arrays.
    """

    if global_to_local is None or global_to_local.shape != mask.shape:
        offsets = cache.borrow_temporary_like(mask, temporary_store)
        global_to_local = offsets
    else:
        offsets = global_to_local

    array_scan(mask, offsets, inclusive=True)

    # Get back total counts (on host if no estimate is provided)
    local_count = (
        min(max_index_count, mask.shape[0])
        if max_index_count >= 0
        else int(host_read_at_index(offsets, temporary_store=temporary_store))
    )

    # Convert counts to indices
    if local_to_global is None or local_to_global.shape[0] != local_count:
        local_to_global = cache.borrow_temporary(temporary_store, shape=local_count, device=mask.device, dtype=int)

    if max_index_count >= 0:
        # We might (and hopefully have) reserved more space than necessary
        # Fill with missing index to avoid uninitialized values
        local_to_global.fill_(missing_index)

    wp.launch(
        kernel=_masked_indices_kernel,
        dim=offsets.shape,
        inputs=[missing_index, mask, offsets, local_to_global, offsets],
        device=mask.device,
    )

    return local_to_global, global_to_local


@wp.kernel
def _prepare_node_sort_kernel(
    node_indices: wp.array(dtype=int),
    sort_keys: wp.array(dtype=int),
    sort_values: wp.array(dtype=int),
    divisor: int,
):
    i = wp.tid()
    node = node_indices[i]
    sort_keys[i] = wp.where(node >= 0, node, NULL_NODE_INDEX)
    sort_values[i] = i // divisor


@wp.kernel
def _scatter_node_counts(
    unique_counts: wp.array(dtype=int),
    unique_node_indices: wp.array(dtype=int),
    node_counts: wp.array(dtype=int),
    unique_node_count: wp.array(dtype=int),
):
    i = wp.tid()

    if i >= unique_node_count[0]:
        if i < unique_node_indices.shape[0]:
            unique_node_indices[i] = NULL_NODE_INDEX
        return

    node_index = unique_node_indices[i]
    if node_index == NULL_NODE_INDEX:
        wp.atomic_sub(unique_node_count, 0, 1)
        return

    node_counts[1 + node_index] = unique_counts[i]


@wp.kernel
def _masked_indices_kernel(
    missing_index: int,
    mask: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    masked_to_global: wp.array(dtype=int),
    global_to_masked: wp.array(dtype=int),
):
    i = wp.tid()

    max_count = masked_to_global.shape[0]
    masked_idx = offsets[i] - 1

    if i + 1 == offsets.shape[0] and masked_idx >= max_count:
        if max_count < offsets[i]:
            wp.printf(
                "Number of elements exceeded the %d limit; increase to %d.\n",
                max_count,
                masked_idx + 1,
            )

    if mask[i] == 0 or masked_idx >= max_count:
        # index not in mask, or greater than reserved index count
        global_to_masked[i] = missing_index
    else:
        global_to_masked[i] = masked_idx
        masked_to_global[masked_idx] = i


def grid_to_tris(Nx: int, Ny: int):
    """Constructs a triangular mesh topology by dividing each cell of a dense 2D grid into two triangles.

    The resulting triangles will be oriented counter-clockwise assuming that `y` is the fastest moving index direction

    Args:
        Nx: Resolution of the grid along `x` dimension
        Ny: Resolution of the grid along `y` dimension

    Returns:
        Array of shape (2 * Nx * Ny, 3) containing vertex indices for each triangle
    """

    cx, cy = np.meshgrid(np.arange(Nx, dtype=int), np.arange(Ny, dtype=int), indexing="ij")

    vidx = np.transpose(
        np.array(
            [
                (Ny + 1) * cx + cy,
                (Ny + 1) * (cx + 1) + cy,
                (Ny + 1) * (cx + 1) + (cy + 1),
                (Ny + 1) * cx + cy,
                (Ny + 1) * (cx + 1) + (cy + 1),
                (Ny + 1) * (cx) + (cy + 1),
            ]
        )
    ).reshape((-1, 3))

    return vidx


def grid_to_tets(Nx: int, Ny: int, Nz: int):
    """Constructs a tetrahedral mesh topology by diving each cell of a dense 3D grid into five tetrahedrons

    The resulting tets have positive volume assuming that `z` is the fastest moving index direction

    Args:
        Nx: Resolution of the grid along `x` dimension
        Ny: Resolution of the grid along `y` dimension
        Nz: Resolution of the grid along `z` dimension

    Returns:
        Array of shape (5 * Nx * Ny * Nz, 4) containing vertex indices for each tet
    """

    # Global node indices for each cell
    cx, cy, cz = np.meshgrid(
        np.arange(Nx, dtype=int), np.arange(Ny, dtype=int), np.arange(Nz, dtype=int), indexing="ij"
    )

    grid_vidx = np.array(
        [
            (Ny + 1) * (Nz + 1) * cx + (Nz + 1) * cy + cz,
            (Ny + 1) * (Nz + 1) * cx + (Nz + 1) * cy + cz + 1,
            (Ny + 1) * (Nz + 1) * cx + (Nz + 1) * (cy + 1) + cz,
            (Ny + 1) * (Nz + 1) * cx + (Nz + 1) * (cy + 1) + cz + 1,
            (Ny + 1) * (Nz + 1) * (cx + 1) + (Nz + 1) * cy + cz,
            (Ny + 1) * (Nz + 1) * (cx + 1) + (Nz + 1) * cy + cz + 1,
            (Ny + 1) * (Nz + 1) * (cx + 1) + (Nz + 1) * (cy + 1) + cz,
            (Ny + 1) * (Nz + 1) * (cx + 1) + (Nz + 1) * (cy + 1) + cz + 1,
        ]
    )

    # decompose grid cells into 5 tets
    tet_vidx = np.array(
        [
            [0, 1, 2, 4],
            [3, 2, 1, 7],
            [5, 1, 7, 4],
            [6, 7, 4, 2],
            [4, 1, 2, 7],
        ]
    )

    # Convert to 3d index coordinates
    vidx_coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    tet_coords = vidx_coords[tet_vidx]

    # Symmetry bits for each cell
    ox, oy, oz = np.meshgrid(
        np.arange(Nx, dtype=int) % 2, np.arange(Ny, dtype=int) % 2, np.arange(Nz, dtype=int) % 2, indexing="ij"
    )
    tet_coords = np.broadcast_to(tet_coords, shape=(*ox.shape, *tet_coords.shape))

    # Flip coordinates according to symmetry
    ox_bk = np.broadcast_to(ox.reshape(*ox.shape, 1, 1), tet_coords.shape[:-1])
    oy_bk = np.broadcast_to(oy.reshape(*oy.shape, 1, 1), tet_coords.shape[:-1])
    oz_bk = np.broadcast_to(oz.reshape(*oz.shape, 1, 1), tet_coords.shape[:-1])

    tet_coords_x = tet_coords[..., 0] ^ ox_bk
    tet_coords_y = tet_coords[..., 1] ^ oy_bk
    tet_coords_z = tet_coords[..., 2] ^ oz_bk

    # Back to local vertex indices
    corner_indices = 4 * tet_coords_x + 2 * tet_coords_y + tet_coords_z

    # Now go from cell-local to global node indices
    # There must be a nicer way than this, but for small grids this works

    corner_indices = corner_indices.reshape(-1, 4)

    grid_vidx = grid_vidx.reshape((8, -1, 1))
    grid_vidx = np.broadcast_to(grid_vidx, shape=(8, grid_vidx.shape[1], 5))
    grid_vidx = grid_vidx.reshape((8, -1))

    node_indices = np.arange(corner_indices.shape[0])
    tet_grid_vidx = np.transpose(
        [
            grid_vidx[corner_indices[:, 0], node_indices],
            grid_vidx[corner_indices[:, 1], node_indices],
            grid_vidx[corner_indices[:, 2], node_indices],
            grid_vidx[corner_indices[:, 3], node_indices],
        ]
    )

    return tet_grid_vidx


def grid_to_quads(Nx: int, Ny: int):
    """Constructs a quadrilateral mesh topology from a dense 2D grid

    The resulting quads will be indexed counter-clockwise

    Args:
        Nx: Resolution of the grid along `x` dimension
        Ny: Resolution of the grid along `y` dimension

    Returns:
        Array of shape (Nx * Ny, 4) containing vertex indices for each quadrilateral
    """

    quad_vtx = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ]
    ).T

    quads = np.stack(np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), indexing="ij"))

    quads_vtx_shape = (*quads.shape, quad_vtx.shape[1])
    quads_vtx = np.broadcast_to(quads.reshape(*quads.shape, 1), quads_vtx_shape) + np.broadcast_to(
        quad_vtx.reshape(2, 1, 1, quad_vtx.shape[1]), quads_vtx_shape
    )

    quad_vtx_indices = quads_vtx[0] * (Ny + 1) + quads_vtx[1]

    return quad_vtx_indices.reshape(-1, 4)


def grid_to_hexes(Nx: int, Ny: int, Nz: int):
    """Constructs a hexahedral mesh topology from a dense 3D grid

    The resulting hexes will be indexed following usual convention assuming that `z` is the fastest moving index direction
    (counter-clockwise bottom vertices, then counter-clockwise top vertices)

    Args:
        Nx: Resolution of the grid along `x` dimension
        Ny: Resolution of the grid along `y` dimension
        Nz: Resolution of the grid along `z` dimension

    Returns:
        Array of shape (Nx * Ny * Nz, 8) containing vertex indices for each hexahedron
    """

    hex_vtx = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    ).T

    hexes = np.stack(np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), np.arange(0, Nz), indexing="ij"))

    hexes_vtx_shape = (*hexes.shape, hex_vtx.shape[1])
    hexes_vtx = np.broadcast_to(hexes.reshape(*hexes.shape, 1), hexes_vtx_shape) + np.broadcast_to(
        hex_vtx.reshape(3, 1, 1, 1, hex_vtx.shape[1]), hexes_vtx_shape
    )

    hexes_vtx_indices = hexes_vtx[0] * (Nz + 1) * (Ny + 1) + hexes_vtx[1] * (Nz + 1) + hexes_vtx[2]

    return hexes_vtx_indices.reshape(-1, 8)
