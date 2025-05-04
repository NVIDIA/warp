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

from typing import Tuple, Union

import numpy as np

import warp as wp
import warp.fem.cache as cache
import warp.types
from warp.fem.linalg import (  # noqa: F401 (for backward compatibility, not part of public API but used in examples)
    array_axpy,
    inverse_qr,
    symmetric_eigenvalues_qr,
)
from warp.fem.types import NULL_NODE_INDEX
from warp.utils import array_scan, radix_sort_pairs, runlength_encode


def type_zero_element(dtype):
    suffix = warp.types.get_type_code(dtype)

    if dtype in warp.types.scalar_types:

        @cache.dynamic_func(suffix=suffix)
        def zero_element():
            return dtype(0.0)

        return zero_element

    @cache.dynamic_func(suffix=suffix)
    def zero_element():
        return dtype()

    return zero_element


def type_basis_element(dtype):
    suffix = warp.types.get_type_code(dtype)

    if dtype in warp.types.scalar_types:

        @cache.dynamic_func(suffix=suffix)
        def basis_element(coord: int):
            return dtype(1.0)

        return basis_element

    if warp.types.type_is_matrix(dtype):
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
    temporary_store: cache.TemporaryStore = None,
) -> Union[Tuple[cache.Temporary, cache.Temporary], Tuple[cache.Temporary, cache.Temporary, int, cache.Temporary]]:
    """
    Compress an unsorted list of node indices into:
     - a node_offsets array, giving for each node the start offset of corresponding indices in sorted_array_indices
     - a sorted_array_indices array, listing the indices in the input array corresponding to each node

    Plus if `return_unique_nodes` is ``True``,
     - the number of unique node indices
     - a unique_node_indices array containing the sorted list of unique node indices (i.e. the list of indices i for which node_offsets[i] < node_offsets[i+1])

    Node indices equal to NULL_NODE_INDEX will be ignored
    """

    index_count = node_indices.size
    device = node_indices.device

    with wp.ScopedDevice(device):
        sorted_node_indices_temp = cache.borrow_temporary(temporary_store, shape=2 * index_count, dtype=int)
        sorted_array_indices_temp = cache.borrow_temporary_like(sorted_node_indices_temp, temporary_store)

        sorted_node_indices = sorted_node_indices_temp.array
        sorted_array_indices = sorted_array_indices_temp.array

        indices_per_element = 1 if node_indices.ndim == 1 else node_indices.shape[-1]
        wp.launch(
            kernel=_prepare_node_sort_kernel,
            dim=index_count,
            inputs=[node_indices.flatten(), sorted_node_indices, sorted_array_indices, indices_per_element],
        )

        # Sort indices
        radix_sort_pairs(sorted_node_indices, sorted_array_indices, count=index_count)

        # Build prefix sum of number of elements per node
        unique_node_indices_temp = cache.borrow_temporary(temporary_store, shape=index_count, dtype=int)
        node_element_counts_temp = cache.borrow_temporary(temporary_store, shape=index_count, dtype=int)

        unique_node_indices = unique_node_indices_temp.array
        node_element_counts = node_element_counts_temp.array

        unique_node_count_dev = cache.borrow_temporary(temporary_store, shape=(1,), dtype=int)

        runlength_encode(
            sorted_node_indices,
            unique_node_indices,
            node_element_counts,
            value_count=index_count,
            run_count=unique_node_count_dev.array,
        )

        # Scatter seen run counts to global array of element count per node
        node_offsets_temp = cache.borrow_temporary(temporary_store, shape=(node_count + 1), dtype=int)
        node_offsets = node_offsets_temp.array

        node_offsets.zero_()
        wp.launch(
            kernel=_scatter_node_counts,
            dim=node_count + 1,  # +1 to accommodate possible NULL node,
            inputs=[node_element_counts, unique_node_indices, node_offsets, unique_node_count_dev.array],
        )

        if device.is_cuda and return_unique_nodes:
            unique_node_count_host = cache.borrow_temporary(
                temporary_store, shape=(1,), dtype=int, pinned=True, device="cpu"
            )
            wp.copy(src=unique_node_count_dev.array, dest=unique_node_count_host.array, count=1)
            copy_event = cache.capture_event(device)

        # Prefix sum of number of elements per node
        array_scan(node_offsets, node_offsets, inclusive=True)

        sorted_node_indices_temp.release()
        node_element_counts_temp.release()

        if not return_unique_nodes:
            unique_node_count_dev.release()
            return node_offsets_temp, sorted_array_indices_temp

        if device.is_cuda:
            cache.synchronize_event(copy_event)
            unique_node_count_dev.release()
        else:
            unique_node_count_host = unique_node_count_dev
        unique_node_count = int(unique_node_count_host.array.numpy()[0])
        unique_node_count_host.release()
        return node_offsets_temp, sorted_array_indices_temp, unique_node_count, unique_node_indices_temp


def host_read_at_index(array: wp.array, index: int = -1, temporary_store: cache.TemporaryStore = None) -> int:
    """Returns the value of the array element at the given index on host"""

    if index < 0:
        index += array.shape[0]

    if array.device.is_cuda:
        temp = cache.borrow_temporary(temporary_store, shape=1, dtype=int, pinned=True, device="cpu")
        wp.copy(dest=temp.array, src=array, src_offset=index, count=1)
        wp.synchronize_stream(wp.get_stream(array.device))
        return temp.array.numpy()[0]

    return array.numpy()[index]


def masked_indices(
    mask: wp.array, missing_index=-1, temporary_store: cache.TemporaryStore = None
) -> Tuple[cache.Temporary, cache.Temporary]:
    """
    From an array of boolean masks (must be either 0 or 1), returns:
      - The list of indices for which the mask is 1
      - A map associating to each element of the input mask array its local index if non-zero, or missing_index if zero.
    """

    offsets_temp = cache.borrow_temporary_like(mask, temporary_store)
    offsets = offsets_temp.array

    wp.utils.array_scan(mask, offsets, inclusive=True)

    # Get back total counts on host
    masked_count = int(host_read_at_index(offsets, temporary_store=temporary_store))

    # Convert counts to indices
    indices_temp = cache.borrow_temporary(temporary_store, shape=masked_count, device=mask.device, dtype=int)

    wp.launch(
        kernel=_masked_indices_kernel,
        dim=offsets.shape,
        inputs=[missing_index, mask, offsets, indices_temp.array, offsets],
        device=mask.device,
    )

    return indices_temp, offsets_temp


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

    if mask[i] == 0:
        global_to_masked[i] = missing_index
    else:
        masked_idx = offsets[i] - 1
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
