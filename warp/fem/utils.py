from typing import Any, Tuple, Union

import numpy as np

import warp as wp
import warp.fem.cache as cache
from warp.fem.types import NULL_NODE_INDEX
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


@wp.func
def householder_qr_decomposition(A: Any):
    """
    QR decomposition of a square matrix using Householder reflections

    Returns Q and R such that Q R = A, Q orthonormal (such that QQ^T = Id), R upper triangular
    """

    x = type(A[0])()
    Q = wp.identity(n=type(x).length, dtype=A.dtype)

    zero = x.dtype(0.0)
    two = x.dtype(2.0)

    for i in range(type(x).length):
        for k in range(type(x).length):
            x[k] = wp.select(k < i, A[k, i], zero)

        alpha = wp.length(x) * wp.sign(x[i])
        x[i] += alpha
        two_over_x_sq = wp.select(alpha == zero, two / wp.length_sq(x), zero)

        A -= wp.outer(two_over_x_sq * x, x * A)
        Q -= wp.outer(Q * x, two_over_x_sq * x)

    return Q, A


@wp.func
def householder_make_hessenberg(A: Any):
    """Transforms a square matrix to Hessenberg form (single lower diagonal) using Householder reflections

    Returns:
        Q and H such that Q H Q^T = A, Q orthonormal, H under Hessenberg form
        If A is symmetric, H will be tridiagonal
    """

    x = type(A[0])()
    Q = wp.identity(n=type(x).length, dtype=A.dtype)

    zero = x.dtype(0.0)
    two = x.dtype(2.0)

    for i in range(1, type(x).length):
        for k in range(type(x).length):
            x[k] = wp.select(k < i, A[k, i - 1], zero)

        alpha = wp.length(x) * wp.sign(x[i])
        x[i] += alpha
        two_over_x_sq = wp.select(alpha == zero, two / wp.length_sq(x), zero)

        # apply on both sides
        A -= wp.outer(two_over_x_sq * x, x * A)
        A -= wp.outer(A * x, two_over_x_sq * x)
        Q -= wp.outer(Q * x, two_over_x_sq * x)

    return Q, A


@wp.func
def solve_triangular(R: Any, b: Any):
    """Solves for R x = b where R is an upper triangular matrix

    Returns x
    """
    zero = b.dtype(0)
    x = type(b)(b.dtype(0))
    for i in range(b.length, 0, -1):
        j = i - 1
        r = b[j] - wp.dot(R[j], x)
        x[j] = wp.select(R[j, j] == zero, r / R[j, j], zero)

    return x


@wp.func
def inverse_qr(A: Any):
    # Computes a square matrix inverse using QR factorization

    Q, R = householder_qr_decomposition(A)

    A_inv = type(A)()
    for i in range(type(A[0]).length):
        A_inv[i] = solve_triangular(R, Q[i])  # ith column of Q^T

    return wp.transpose(A_inv)


@wp.func
def symmetric_eigenvalues_qr(A: Any, tol: Any):
    """
    Computes the eigenvalues and eigen vectors of a square symmetric matrix A using the QR algorithm

    Args:
        A: square symmetric matrix
        tol: Tolerance for the diagonalization residual (squared L2 norm of off-diagonal terms)

    Returns a tuple (D: vector of eigenvalues, P: matrix with one eigenvector per row) such that A = P^T D P
    """

    two = A.dtype(2.0)
    zero = A.dtype(0.0)

    # temp storage for matrix rows
    ri = type(A[0])()
    rn = type(ri)()

    # tridiagonal storage for R
    R_L = type(ri)()
    R_L = type(ri)(zero)
    R_U = type(ri)(zero)

    # so that we can use the type length in expression
    # this will prevent unrolling by warp, but should be ok for native code
    m = int(0)
    for _ in range(type(ri).length):
        m += 1

    # Put A under Hessenberg form (tridiagonal)
    Q, H = householder_make_hessenberg(A)
    Q = wp.transpose(Q)  # algorithm below works and transposed Q as rows are easier to index

    for _ in range(16 * m):  # failsafe, usually converges faster than that
        # Initialize R with current H
        R_D = wp.get_diag(H)
        for i in range(1, type(ri).length):
            R_L[i - 1] = H[i, i - 1]
            R_U[i - 1] = H[i - 1, i]

        # compute QR decomposition, directly transform H and eigenvectors
        for n in range(1, m):
            i = n - 1

            # compute reflection
            xi = R_D[i]
            xn = R_L[i]

            xii = xi * xi
            xnn = xn * xn
            alpha = wp.sqrt(xii + xnn) * wp.sign(xi)

            xi += alpha
            xii = xi * xi
            xin = xi * xn

            two_over_x_sq = wp.select(alpha == zero, two / (xii + xnn), zero)
            xii *= two_over_x_sq
            xin *= two_over_x_sq
            xnn *= two_over_x_sq

            # Left-multiply R and Q, multiply H on both sides
            # Note that R should get non-zero coefficients on the second upper diagonal,
            # but those won't get read afterwards, so we can ignore them

            R_D[n] -= R_U[i] * xin + R_D[n] * xnn
            R_U[n] -= R_U[n] * xnn

            ri = Q[i]
            rn = Q[n]
            Q[i] -= ri * xii + rn * xin
            Q[n] -= ri * xin + rn * xnn

            # H is multiplied on both sides, but stays tridiagonal except for moving buldge
            # Note: we could reduce the stencil to for 4 columns qui we do below,
            # but unlikely to be worth it for our small matrix sizes
            ri = H[i]
            rn = H[n]
            H[i] -= ri * xii + rn * xin
            H[n] -= ri * xin + rn * xnn

            # multiply on right, manually. We just need to consider 4 rows
            if i > 0:
                ci = H[i - 1, i]
                cn = H[i - 1, n]
                H[i - 1, i] -= ci * xii + cn * xin
                H[i - 1, n] -= ci * xin + cn * xnn

            for k in range(2):
                ci = H[i + k, i]
                cn = H[i + k, n]
                H[i + k, i] -= ci * xii + cn * xin
                H[i + k, n] -= ci * xin + cn * xnn

            if n + 1 < m:
                ci = H[n + 1, i]
                cn = H[n + 1, n]
                H[n + 1, i] -= ci * xii + cn * xin
                H[n + 1, n] -= ci * xin + cn * xnn

        # Terminate if the upper diagonal of R is near zero
        if wp.length_sq(R_U) < tol:
            break

    return wp.get_diag(H), Q


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

        wp.copy(dest=sorted_node_indices, src=node_indices, count=index_count)

        indices_per_element = 1 if node_indices.ndim == 1 else node_indices.shape[-1]
        wp.launch(
            kernel=_iota_kernel,
            dim=index_count,
            inputs=[sorted_array_indices, indices_per_element],
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
