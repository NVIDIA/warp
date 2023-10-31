from typing import Union, Any, Tuple

import warp as wp
import warp.types

from warp.sparse import BsrMatrix, bsr_zeros, bsr_get_diag, bsr_mv
from warp.utils import array_inner


def bsr_to_scipy(matrix: BsrMatrix) -> "scipy.sparse.bsr_array":
    try:
        from scipy.sparse import csr_array, bsr_array
    except ImportError:
        # WAR for older scipy
        from scipy.sparse import csr_matrix as csr_array, bsr_matrix as bsr_array

    if matrix.block_shape == (1, 1):
        return csr_array(
            (
                matrix.values.numpy().flatten()[: matrix.nnz],
                matrix.columns.numpy()[: matrix.nnz],
                matrix.offsets.numpy(),
            ),
            shape=matrix.shape,
        )

    return bsr_array(
        (
            matrix.values.numpy().reshape((matrix.values.shape[0], *matrix.block_shape))[: matrix.nnz],
            matrix.columns.numpy()[: matrix.nnz],
            matrix.offsets.numpy(),
        ),
        shape=matrix.shape,
    )


def scipy_to_bsr(sp: Union["scipy.sparse.bsr_array", "scipy.sparse.csr_array"], device=None, dtype=None) -> BsrMatrix:
    try:
        from scipy.sparse import csr_array
    except ImportError:
        # WAR for older scipy
        from scipy.sparse import csr_matrix as csr_array

    if dtype is None:
        dtype = warp.types.np_dtype_to_warp_type[sp.dtype]

    sp.sort_indices()

    if isinstance(sp, csr_array):
        matrix = bsr_zeros(sp.shape[0], sp.shape[1], dtype, device=device)
    else:
        block_shape = sp.blocksize
        block_type = wp.types.matrix(shape=block_shape, dtype=dtype)
        matrix = bsr_zeros(sp.shape[0] // block_shape[0], sp.shape[1] // block_shape[1], block_type, device=device)

    matrix.nnz = sp.nnz
    matrix.values = wp.array(sp.data.flatten(), dtype=matrix.values.dtype, device=device)
    matrix.columns = wp.array(sp.indices, dtype=matrix.columns.dtype, device=device)
    matrix.offsets = wp.array(sp.indptr, dtype=matrix.offsets.dtype, device=device)

    return matrix


@wp.kernel
def _bsr_cg_kernel_1(
    rs_old: wp.array(dtype=Any),
    p_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
):
    i = wp.tid()

    if p_Ap[0] != 0.0:
        alpha = rs_old[0] / p_Ap[0]
        x[i] = x[i] + alpha * p[i]
        r[i] = r[i] - alpha * Ap[i]


@wp.kernel
def _bsr_cg_kernel_2(
    tol: Any,
    rs_old: wp.array(dtype=Any),
    rs_new: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
):
    #    p = r + (rsnew / rsold) * p;
    i = wp.tid()

    if rs_new[0] > tol:
        beta = rs_new[0] / rs_old[0]
    else:
        beta = rs_new[0] - rs_new[0]

    p[i] = z[i] + beta * p[i]


@wp.kernel
def _bsr_cg_solve_block_diag_precond_kernel(
    diag: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
):
    i = wp.tid()
    d = wp.get_diag(diag[i])

    if wp.dot(d, d) == 0.0:
        z[i] = r[i]
    else:
        d_abs = wp.max(d, -d)
        z[i] = wp.cw_div(r[i], d_abs)


@wp.kernel
def _bsr_cg_solve_scalar_diag_precond_kernel(
    diag: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
):
    i = wp.tid()
    d = diag[i]

    if d == 0.0:
        z[i] = r[i]
    else:
        z[i] = r[i] / wp.abs(d)


def bsr_cg(
    A: BsrMatrix,
    x: wp.array,
    b: wp.array,
    max_iters: int = 0,
    tol: float = 0.0001,
    check_every=10,
    use_diag_precond=True,
    mv_routine=bsr_mv,
    device=None,
    quiet=False,
) -> Tuple[float, int]:
    """Solves the linear system A x = b using the Conjugate Gradient method, optionally with diagonal preconditioning

    Args:
        A: system left-hand side
        x: result vector and initial guess
        b: system right-hand-side
        max_iters: maximum number of iterations to performing before aborting. If set to zero, equal to the system size.
        tol: relative tolerance under which to stop the solve
        check_every: number of iterations every which to evaluate the current residual norm to compare against tolerance
        use_diag_precond: Whether to use diagonal preconditioning
        mv_routine: Matrix-vector multiplication routine to for multiplications with ``A``
        device: Warp device to use for the computation

    Returns:
        Tuple (residual norm, iteration count)

    """

    if max_iters == 0:
        max_iters = A.shape[0]
    if device is None:
        device = A.values.device

    scalar_dtype = A.scalar_type

    r = wp.zeros_like(b)
    p = wp.zeros_like(b)
    Ap = wp.zeros_like(b)

    if use_diag_precond:
        A_diag = bsr_get_diag(A)
        z = wp.zeros_like(b)

        if A.block_shape == (1, 1):
            precond_kernel = _bsr_cg_solve_scalar_diag_precond_kernel
        else:
            precond_kernel = _bsr_cg_solve_block_diag_precond_kernel
    else:
        z = r

    rz_old = wp.empty(n=1, dtype=scalar_dtype, device=device)
    rz_new = wp.empty(n=1, dtype=scalar_dtype, device=device)
    p_Ap = wp.empty(n=1, dtype=scalar_dtype, device=device)

    # r = b - A * x;
    r.assign(b)
    mv_routine(A, x, r, alpha=-1.0, beta=1.0)

    # z = M^-1 r
    if use_diag_precond:
        wp.launch(kernel=precond_kernel, dim=A.nrow, device=device, inputs=[A_diag, r, z])

    # p = z;
    p.assign(z)

    # rsold = r' * z;
    array_inner(r, z, out=rz_old)

    tol_sq = tol * tol * A.shape[0]

    err = rz_old.numpy()[0]
    end_iter = 0

    if err > tol_sq:
        end_iter = max_iters

        for i in range(max_iters):
            # Ap = A * p;
            mv_routine(A, p, Ap)

            array_inner(p, Ap, out=p_Ap)

            wp.launch(kernel=_bsr_cg_kernel_1, dim=A.nrow, device=device, inputs=[rz_old, p_Ap, x, r, p, Ap])

            # z = M^-1 r
            if use_diag_precond:
                wp.launch(kernel=precond_kernel, dim=A.nrow, device=device, inputs=[A_diag, r, z])

            # rznew = r' * z;
            array_inner(r, z, out=rz_new)

            if ((i + 1) % check_every) == 0:
                err = rz_new.numpy()[0]
                if not quiet:
                    print(f"At iteration {i} error = \t {err}  \t tol: {tol_sq}")
                if err <= tol_sq:
                    end_iter = i
                    break

            wp.launch(kernel=_bsr_cg_kernel_2, dim=A.nrow, device=device, inputs=[tol_sq, rz_old, rz_new, z, p])

            # swap buffers
            rs_tmp = rz_old
            rz_old = rz_new
            rz_new = rs_tmp

        err = rz_old.numpy()[0]

    if not quiet:
        print(f"Terminated after {end_iter} iterations with error = \t {err}")
    return err, end_iter


def invert_diagonal_bsr_mass_matrix(A: BsrMatrix):
    """Inverts each block of a block-diagonal mass matrix"""

    scale = A.scalar_type(A.block_shape[0])
    values = A.values
    if not wp.types.type_is_matrix(values.dtype):
        values = values.view(dtype=wp.mat(shape=(1, 1), dtype=A.scalar_type))

    wp.launch(kernel=_block_diagonal_mass_invert, dim=A.nrow, inputs=[values, scale], device=values.device)


@wp.kernel
def _block_diagonal_mass_invert(values: wp.array(dtype=Any), scale: Any):
    i = wp.tid()
    values[i] = scale * values[i] / wp.ddot(values[i], values[i])
