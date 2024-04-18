from typing import Any, Optional, Tuple, Union

import warp as wp
import warp.types
from warp.optim.linear import LinearOperator, aslinearoperator, preconditioner
from warp.sparse import BsrMatrix, bsr_get_diag, bsr_mv, bsr_transposed, bsr_zeros


def bsr_to_scipy(matrix: BsrMatrix) -> "scipy.sparse.bsr_array":  # noqa: F821
    try:
        from scipy.sparse import bsr_array, csr_array
    except ImportError:
        # WAR for older scipy
        from scipy.sparse import bsr_matrix as bsr_array
        from scipy.sparse import csr_matrix as csr_array

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


def scipy_to_bsr(
    sp: Union["scipy.sparse.bsr_array", "scipy.sparse.csr_array"],  # noqa: F821
    device=None,
    dtype=None,
) -> BsrMatrix:
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
        matrix = bsr_zeros(
            sp.shape[0] // block_shape[0],
            sp.shape[1] // block_shape[1],
            block_type,
            device=device,
        )

    matrix.nnz = sp.nnz
    matrix.values = wp.array(sp.data.flatten(), dtype=matrix.values.dtype, device=device)
    matrix.columns = wp.array(sp.indices, dtype=matrix.columns.dtype, device=device)
    matrix.offsets = wp.array(sp.indptr, dtype=matrix.offsets.dtype, device=device)

    return matrix


def get_linear_solver_func(method_name: str):
    from warp.optim.linear import bicgstab, cg, cr, gmres

    if method_name == "bicgstab":
        return bicgstab
    if method_name == "gmres":
        return gmres
    if method_name == "cr":
        return cr
    return cg


def bsr_cg(
    A: BsrMatrix,
    x: wp.array,
    b: wp.array,
    max_iters: int = 0,
    tol: float = 0.0001,
    check_every=10,
    use_diag_precond=True,
    mv_routine=None,
    quiet=False,
    method: str = "cg",
) -> Tuple[float, int]:
    """Solves the linear system A x = b using an iterative solver, optionally with diagonal preconditioning

    Args:
        A: system left-hand side
        x: result vector and initial guess
        b: system right-hand-side
        max_iters: maximum number of iterations to perform before aborting. If set to zero, equal to the system size.
        tol: relative tolerance under which to stop the solve
        check_every: number of iterations every which to evaluate the current residual norm to compare against tolerance
        use_diag_precond: Whether to use diagonal preconditioning
        mv_routine: Matrix-vector multiplication routine to use for multiplications with ``A``
        quiet: if True, do not print iteration residuals
        method: Iterative solver method to use, defaults to Conjugate Gradient

    Returns:
        Tuple (residual norm, iteration count)

    """

    if mv_routine is None:
        M = preconditioner(A, "diag") if use_diag_precond else None
    else:
        A = LinearOperator(A.shape, A.dtype, A.device, matvec=mv_routine)
        M = None

    func = get_linear_solver_func(method_name=method)

    def print_callback(i, err, tol):
        print(f"{func.__name__}: at iteration {i} error = \t {err}  \t tol: {tol}")

    callback = None if quiet else print_callback

    end_iter, err, atol = func(
        A=A,
        b=b,
        x=x,
        maxiter=max_iters,
        tol=tol,
        check_every=check_every,
        M=M,
        callback=callback,
    )

    if not quiet:
        res_str = "OK" if err <= atol else "TRUNCATED"
        print(f"{func.__name__}: terminated after {end_iter} iterations with error = \t {err} ({res_str})")

    return err, end_iter


class SaddleSystem(LinearOperator):
    """Builds a linear operator corresponding to the saddle-point linear system [A B^T; B 0]

    If use_diag_precond` is ``True``,  builds the corresponding diagonal preconditioner `[diag(A); diag(B diag(A)^-1 B^T)]`
    """

    def __init__(
        self,
        A: BsrMatrix,
        B: BsrMatrix,
        Bt: Optional[BsrMatrix] = None,
        use_diag_precond: bool = True,
    ):
        if Bt is None:
            Bt = bsr_transposed(B)

        self._A = A
        self._B = B
        self._Bt = Bt

        self._u_dtype = wp.vec(length=A.block_shape[0], dtype=A.scalar_type)
        self._p_dtype = wp.vec(length=B.block_shape[0], dtype=B.scalar_type)
        self._p_byte_offset = A.nrow * wp.types.type_size_in_bytes(self._u_dtype)

        saddle_shape = (A.shape[0] + B.shape[0], A.shape[0] + B.shape[0])

        super().__init__(saddle_shape, dtype=A.scalar_type, device=A.device, matvec=self._saddle_mv)

        if use_diag_precond:
            self._preconditioner = self._diag_preconditioner()
        else:
            self._preconditioner = None

    def _diag_preconditioner(self):
        A = self._A
        B = self._B

        M_u = preconditioner(A, "diag")

        A_diag = bsr_get_diag(A)

        schur_block_shape = (B.block_shape[0], B.block_shape[0])
        schur_dtype = wp.mat(shape=schur_block_shape, dtype=B.scalar_type)
        schur_inv_diag = wp.empty(dtype=schur_dtype, shape=B.nrow, device=self.device)
        wp.launch(
            _compute_schur_inverse_diagonal,
            dim=B.nrow,
            device=A.device,
            inputs=[B.offsets, B.columns, B.values, A_diag, schur_inv_diag],
        )

        if schur_block_shape == (1, 1):
            # Downcast 1x1 mats to scalars
            schur_inv_diag = schur_inv_diag.view(dtype=B.scalar_type)

        M_p = aslinearoperator(schur_inv_diag)

        def precond_mv(x, y, z, alpha, beta):
            x_u = self.u_slice(x)
            x_p = self.p_slice(x)
            y_u = self.u_slice(y)
            y_p = self.p_slice(y)
            z_u = self.u_slice(z)
            z_p = self.p_slice(z)

            M_u.matvec(x_u, y_u, z_u, alpha=alpha, beta=beta)
            M_p.matvec(x_p, y_p, z_p, alpha=alpha, beta=beta)

        return LinearOperator(
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            matvec=precond_mv,
        )

    @property
    def preconditioner(self):
        return self._preconditioner

    def u_slice(self, a: wp.array):
        return wp.array(
            ptr=a.ptr,
            dtype=self._u_dtype,
            shape=self._A.nrow,
            strides=None,
            device=a.device,
            pinned=a.pinned,
            copy=False,
        )

    def p_slice(self, a: wp.array):
        return wp.array(
            ptr=a.ptr + self._p_byte_offset,
            dtype=self._p_dtype,
            shape=self._B.nrow,
            strides=None,
            device=a.device,
            pinned=a.pinned,
            copy=False,
        )

    def _saddle_mv(self, x, y, z, alpha, beta):
        x_u = self.u_slice(x)
        x_p = self.p_slice(x)
        z_u = self.u_slice(z)
        z_p = self.p_slice(z)

        if y.ptr != z.ptr and beta != 0.0:
            wp.copy(src=y, dest=z)

        bsr_mv(self._A, x_u, z_u, alpha=alpha, beta=beta)
        bsr_mv(self._Bt, x_p, z_u, alpha=alpha, beta=1.0)
        bsr_mv(self._B, x_u, z_p, alpha=alpha, beta=beta)


def bsr_solve_saddle(
    saddle_system: SaddleSystem,
    x_u: wp.array,
    x_p: wp.array,
    b_u: wp.array,
    b_p: wp.array,
    max_iters: int = 0,
    tol: float = 0.0001,
    check_every=10,
    quiet=False,
    method: str = "cg",
) -> Tuple[float, int]:
    """Solves the saddle-point linear system [A B^T; B 0] (x_u; x_p) = (b_u; b_p) using an iterative solver, optionally with diagonal preconditioning

    Args:
        saddle_system: Saddle point system
        x_u: primal part of the result vector and initial guess
        x_p: Lagrange multiplier part of the result vector and initial guess
        b_u: primal left-hand-side
        b_p: constraint left-hand-side
        max_iters: maximum number of iterations to perform before aborting. If set to zero, equal to the system size.
        tol: relative tolerance under which to stop the solve
        check_every: number of iterations every which to evaluate the current residual norm to compare against tolerance
        quiet: if True, do not print iteration residuals
        method: Iterative solver method to use, defaults to BiCGSTAB

    Returns:
        Tuple (residual norm, iteration count)

    """
    x = wp.empty(dtype=saddle_system.scalar_type, shape=saddle_system.shape[0], device=saddle_system.device)
    b = wp.empty_like(x)

    wp.copy(src=x_u, dest=saddle_system.u_slice(x))
    wp.copy(src=x_p, dest=saddle_system.p_slice(x))
    wp.copy(src=b_u, dest=saddle_system.u_slice(b))
    wp.copy(src=b_p, dest=saddle_system.p_slice(b))

    func = get_linear_solver_func(method_name=method)

    def print_callback(i, err, tol):
        print(f"{func.__name__}: at iteration {i} error = \t {err}  \t tol: {tol}")

    callback = None if quiet else print_callback

    end_iter, err, atol = func(
        A=saddle_system,
        b=b,
        x=x,
        maxiter=max_iters,
        tol=tol,
        check_every=check_every,
        M=saddle_system.preconditioner,
        callback=callback,
    )

    if not quiet:
        res_str = "OK" if err <= atol else "TRUNCATED"
        print(f"{func.__name__}: terminated after {end_iter} iterations with absolute error = \t {err} ({res_str})")

    wp.copy(dest=x_u, src=saddle_system.u_slice(x))
    wp.copy(dest=x_p, src=saddle_system.p_slice(x))

    return err, end_iter


@wp.kernel
def _compute_schur_inverse_diagonal(
    B_offsets: wp.array(dtype=int),
    B_indices: wp.array(dtype=int),
    B_values: wp.array(dtype=Any),
    A_diag: wp.array(dtype=Any),
    P_diag: wp.array(dtype=Any),
):
    row = wp.tid()

    zero = P_diag.dtype(P_diag.dtype.dtype(0.0))

    schur = zero

    beg = B_offsets[row]
    end = B_offsets[row + 1]

    for b in range(beg, end):
        B = B_values[b]
        col = B_indices[b]
        Ai = wp.inverse(A_diag[col])
        S = B * Ai * wp.transpose(B)
        schur += S

    schur_diag = wp.get_diag(schur)
    id_diag = type(schur_diag)(schur_diag.dtype(1.0))

    inv_diag = wp.select(schur == zero, wp.cw_div(id_diag, schur_diag), id_diag)
    P_diag[row] = wp.diag(inv_diag)


def invert_diagonal_bsr_mass_matrix(A: BsrMatrix):
    """Inverts each block of a block-diagonal mass matrix"""

    scale = A.scalar_type(A.block_shape[0])
    values = A.values
    if not wp.types.type_is_matrix(values.dtype):
        values = values.view(dtype=wp.mat(shape=(1, 1), dtype=A.scalar_type))

    wp.launch(
        kernel=_block_diagonal_mass_invert,
        dim=A.nrow,
        inputs=[values, scale],
        device=values.device,
    )


@wp.kernel
def _block_diagonal_mass_invert(values: wp.array(dtype=Any), scale: Any):
    i = wp.tid()
    values[i] = scale * values[i] / wp.ddot(values[i], values[i])
