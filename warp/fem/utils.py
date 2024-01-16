from typing import Any, Tuple

import numpy as np

import warp as wp
from warp.fem.cache import (
    Temporary,
    TemporaryStore,
    borrow_temporary,
    borrow_temporary_like,
)
from warp.utils import array_scan, radix_sort_pairs, runlength_encode


@wp.func
def generalized_outer(x: Any, y: Any):
    """Generalized outer product allowing for the first argument to be a scalar"""
    return wp.outer(x, y)


@wp.func
def generalized_outer(x: wp.float32, y: wp.vec2):
    return x * y


@wp.func
def generalized_outer(x: wp.float32, y: wp.vec3):
    return x * y


@wp.func
def generalized_inner(x: Any, y: Any):
    """Generalized inner product allowing for the first argument to be a tensor"""
    return wp.dot(x, y)


@wp.func
def generalized_inner(x: wp.mat22, y: wp.vec2):
    return x[0] * y[0] + x[1] * y[1]


@wp.func
def generalized_inner(x: wp.mat33, y: wp.vec3):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


@wp.func
def apply_right(x: Any, y: Any):
    """Performs x y multiplication with y a square matrix and x either a row-vector or a matrix.
    Will be removed once native @ operator is implemented.
    """
    return x * y


@wp.func
def apply_right(x: wp.vec2, y: wp.mat22):
    return x[0] * y[0] + x[1] * y[1]


@wp.func
def apply_right(x: wp.vec3, y: wp.mat33):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


@wp.func
def unit_element(template_type: Any, coord: int):
    """Returns a instance of `template_type` with a single coordinate set to 1 in the canonical basis"""

    t = type(template_type)(0.0)
    t[coord] = 1.0
    return t


@wp.func
def unit_element(template_type: wp.float32, coord: int):
    return 1.0


@wp.func
def unit_element(template_type: wp.mat22, coord: int):
    t = wp.mat22(0.0)
    row = coord // 2
    col = coord - 2 * row
    t[row, col] = 1.0
    return t


@wp.func
def unit_element(template_type: wp.mat33, coord: int):
    t = wp.mat33(0.0)
    row = coord // 3
    col = coord - 3 * row
    t[row, col] = 1.0
    return t


@wp.func
def symmetric_part(x: Any):
    """Symmetric part of a square tensor"""
    return 0.5 * (x + wp.transpose(x))


@wp.func
def skew_part(x: wp.mat22):
    """Skew part of a 2x2 tensor as corresponding rotation angle"""
    return 0.5 * (x[1, 0] - x[0, 1])


@wp.func
def skew_part(x: wp.mat33):
    """Skew part of a 3x3 tensor as the corresponding rotation vector"""
    a = 0.5 * (x[2, 1] - x[1, 2])
    b = 0.5 * (x[0, 2] - x[2, 0])
    c = 0.5 * (x[1, 0] - x[0, 1])
    return wp.vec3(a, b, c)


def compress_node_indices(
    node_count: int, node_indices: wp.array(dtype=int), temporary_store: TemporaryStore = None
) -> Tuple[Temporary, Temporary, int, Temporary]:
    """
    Compress an unsorted list of node indices into:
     - a node_offsets array, giving for each node the start offset of corresponding indices in sorted_array_indices
     - a sorted_array_indices array, listing the indices in the input array corresponding to each node
     - the number of unique node indices
     - a unique_node_indices array containing the sorted list of unique node indices (i.e. the list of indices i for which node_offsets[i] < node_offsets[i+1])
    """

    index_count = node_indices.size

    sorted_node_indices_temp = borrow_temporary(
        temporary_store, shape=2 * index_count, dtype=int, device=node_indices.device
    )
    sorted_array_indices_temp = borrow_temporary_like(sorted_node_indices_temp, temporary_store)

    sorted_node_indices = sorted_node_indices_temp.array
    sorted_array_indices = sorted_array_indices_temp.array

    wp.copy(dest=sorted_node_indices, src=node_indices, count=index_count)

    indices_per_element = 1 if node_indices.ndim == 1 else node_indices.shape[-1]
    wp.launch(
        kernel=_iota_kernel,
        dim=index_count,
        inputs=[sorted_array_indices, indices_per_element],
        device=sorted_array_indices.device,
    )

    # Sort indices
    radix_sort_pairs(sorted_node_indices, sorted_array_indices, count=index_count)

    # Build prefix sum of number of elements per node
    unique_node_indices_temp = borrow_temporary(
        temporary_store, shape=index_count, dtype=int, device=node_indices.device
    )
    node_element_counts_temp = borrow_temporary(
        temporary_store, shape=index_count, dtype=int, device=node_indices.device
    )

    unique_node_indices = unique_node_indices_temp.array
    node_element_counts = node_element_counts_temp.array

    unique_node_count_dev = borrow_temporary(temporary_store, shape=(1,), dtype=int, device=sorted_node_indices.device)
    runlength_encode(
        sorted_node_indices,
        unique_node_indices,
        node_element_counts,
        value_count=index_count,
        run_count=unique_node_count_dev.array,
    )

    # Transfer unique node count to host
    if node_indices.device.is_cuda:
        unique_node_count_host = borrow_temporary(temporary_store, shape=(1,), dtype=int, pinned=True, device="cpu")
        wp.copy(src=unique_node_count_dev.array, dest=unique_node_count_host.array, count=1)
        wp.synchronize_stream(wp.get_stream(node_indices.device))
        unique_node_count_dev.release()
        unique_node_count = int(unique_node_count_host.array.numpy()[0])
        unique_node_count_host.release()
    else:
        unique_node_count = int(unique_node_count_dev.array.numpy()[0])
        unique_node_count_dev.release()

    # Scatter seen run counts to global array of element count per node
    node_offsets_temp = borrow_temporary(
        temporary_store, shape=(node_count + 1), device=node_element_counts.device, dtype=int
    )
    node_offsets = node_offsets_temp.array

    node_offsets.zero_()
    wp.launch(
        kernel=_scatter_node_counts,
        dim=unique_node_count,
        inputs=[node_element_counts, unique_node_indices, node_offsets],
        device=node_offsets.device,
    )

    # Prefix sum of number of elements per node
    array_scan(node_offsets, node_offsets, inclusive=True)

    sorted_node_indices_temp.release()
    node_element_counts_temp.release()

    return node_offsets_temp, sorted_array_indices_temp, unique_node_count, unique_node_indices_temp


def masked_indices(
    mask: wp.array, missing_index=-1, temporary_store: TemporaryStore = None
) -> Tuple[Temporary, Temporary]:
    """
    From an array of boolean masks (must be either 0 or 1), returns:
      - The list of indices for which the mask is 1
      - A map associating to each element of the input mask array its local index if non-zero, or missing_index if zero.
    """

    offsets_temp = borrow_temporary_like(mask, temporary_store)
    offsets = offsets_temp.array

    wp.utils.array_scan(mask, offsets, inclusive=True)

    # Get back total counts on host
    if offsets.device.is_cuda:
        masked_count_temp = borrow_temporary(temporary_store, shape=1, dtype=int, pinned=True, device="cpu")
        wp.copy(dest=masked_count_temp.array, src=offsets, src_offset=offsets.shape[0] - 1, count=1)
        wp.synchronize_stream(wp.get_stream(offsets.device))
        masked_count = int(masked_count_temp.array.numpy()[0])
        masked_count_temp.release()
    else:
        masked_count = int(offsets.numpy()[-1])

    # Convert counts to indices
    indices_temp = borrow_temporary(temporary_store, shape=masked_count, device=mask.device, dtype=int)

    wp.launch(
        kernel=_masked_indices_kernel,
        dim=offsets.shape,
        inputs=[missing_index, mask, offsets, indices_temp.array, offsets],
        device=mask.device,
    )

    return indices_temp, offsets_temp


def array_axpy(x: wp.array, y: wp.array, alpha: float = 1.0, beta: float = 1.0):
    """Performs y = alpha*x + beta*y"""

    dtype = wp.types.type_scalar_type(x.dtype)

    alpha = dtype(alpha)
    beta = dtype(beta)

    if not wp.types.types_equal(x.dtype, y.dtype) or x.shape != y.shape or x.device != y.device:
        raise ValueError("x and y arrays must have same dat atype, shape and device")

    wp.launch(kernel=_array_axpy_kernel, dim=x.shape, device=x.device, inputs=[x, y, alpha, beta])


@wp.kernel
def _iota_kernel(indices: wp.array(dtype=int), divisor: int):
    indices[wp.tid()] = wp.tid() // divisor


@wp.kernel
def _scatter_node_counts(
    unique_counts: wp.array(dtype=int), unique_node_indices: wp.array(dtype=int), node_counts: wp.array(dtype=int)
):
    i = wp.tid()
    node_counts[1 + unique_node_indices[i]] = unique_counts[i]


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


@wp.kernel
def _array_axpy_kernel(x: wp.array(dtype=Any), y: wp.array(dtype=Any), alpha: Any, beta: Any):
    i = wp.tid()
    y[i] = beta * y[i] + alpha * x[i]


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
        Array of shape (Nx * Ny * Nz, 8) containing vertex indices for each hexaedron
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
