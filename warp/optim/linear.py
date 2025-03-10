# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from math import sqrt
from typing import Any, Callable, Optional, Tuple, Union

import warp as wp
import warp.sparse as sparse
from warp.utils import array_inner

# No need to auto-generate adjoint code for linear solvers
wp.set_module_options({"enable_backward": False})


class LinearOperator:
    """
    Linear operator to be used as left-hand-side of linear iterative solvers.

    Args:
        shape: Tuple containing the number of rows and columns of the operator
        dtype: Type of the operator elements
        device: Device on which computations involving the operator should be performed
        matvec: Matrix-vector multiplication routine

    The matrix-vector multiplication routine should have the following signature:

    .. code-block:: python

        def matvec(x: wp.array, y: wp.array, z: wp.array, alpha: Scalar, beta: Scalar):
            '''Performs the operation z = alpha * x + beta * y'''
            ...

    For performance reasons, by default the iterative linear solvers in this module will try to capture the calls
    for one or more iterations in CUDA graphs. If the `matvec` routine of a custom :class:`LinearOperator`
    cannot be graph-captured, the ``use_cuda_graph=False`` parameter should be passed to the solver function.

    """

    def __init__(self, shape: Tuple[int, int], dtype: type, device: wp.context.Device, matvec: Callable):
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._matvec = matvec

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def device(self) -> wp.context.Device:
        return self._device

    @property
    def matvec(self) -> Callable:
        return self._matvec

    @property
    def scalar_type(self):
        return wp.types.type_scalar_type(self.dtype)


_Matrix = Union[wp.array, sparse.BsrMatrix, LinearOperator]


def aslinearoperator(A: _Matrix) -> LinearOperator:
    """
    Casts the dense or sparse matrix `A` as a :class:`LinearOperator`

    `A` must be of one of the following types:

        - :class:`warp.sparse.BsrMatrix`
        - two-dimensional `warp.array`; then `A` is assumed to be a dense matrix
        - one-dimensional `warp.array`; then `A` is assumed to be a diagonal matrix
        - :class:`warp.sparse.LinearOperator`; no casting necessary
    """

    if A is None or isinstance(A, LinearOperator):
        return A

    def bsr_mv(x, y, z, alpha, beta):
        if z.ptr != y.ptr and beta != 0.0:
            wp.copy(src=y, dest=z)
        sparse.bsr_mv(A, x, z, alpha, beta)

    def dense_mv(x, y, z, alpha, beta):
        wp.launch(_dense_mv_kernel, dim=A.shape[1], device=A.device, inputs=[A, x, y, z, alpha, beta])

    def diag_mv(x, y, z, alpha, beta):
        scalar_type = wp.types.type_scalar_type(A.dtype)
        alpha = scalar_type(alpha)
        beta = scalar_type(beta)
        wp.launch(_diag_mv_kernel, dim=A.shape, device=A.device, inputs=[A, x, y, z, alpha, beta])

    def diag_mv_vec(x, y, z, alpha, beta):
        scalar_type = wp.types.type_scalar_type(A.dtype)
        alpha = scalar_type(alpha)
        beta = scalar_type(beta)
        wp.launch(_diag_mv_vec_kernel, dim=A.shape, device=A.device, inputs=[A, x, y, z, alpha, beta])

    if isinstance(A, wp.array):
        if A.ndim == 2:
            return LinearOperator(A.shape, A.dtype, A.device, matvec=dense_mv)
        if A.ndim == 1:
            if wp.types.type_is_vector(A.dtype):
                return LinearOperator(A.shape, A.dtype, A.device, matvec=diag_mv_vec)
            return LinearOperator(A.shape, A.dtype, A.device, matvec=diag_mv)
    if isinstance(A, sparse.BsrMatrix):
        return LinearOperator(A.shape, A.dtype, A.device, matvec=bsr_mv)

    raise ValueError(f"Unable to create LinearOperator from {A}")


def preconditioner(A: _Matrix, ptype: str = "diag") -> LinearOperator:
    """Constructs and returns a preconditioner for an input matrix.

    Args:
        A: The matrix for which to build the preconditioner
        ptype: The type of preconditioner. Currently the following values are supported:

         - ``"diag"``: Diagonal (a.k.a. Jacobi) preconditioner
         - ``"diag_abs"``: Similar to Jacobi, but using the absolute value of diagonal coefficients
         - ``"id"``: Identity (null) preconditioner
    """

    if ptype == "id":
        return None

    if ptype in ("diag", "diag_abs"):
        use_abs = 1 if ptype == "diag_abs" else 0
        if isinstance(A, sparse.BsrMatrix):
            A_diag = sparse.bsr_get_diag(A)
            if wp.types.type_is_matrix(A.dtype):
                inv_diag = wp.empty(
                    shape=A.nrow, dtype=wp.vec(length=A.block_shape[0], dtype=A.scalar_type), device=A.device
                )
                wp.launch(
                    _extract_inverse_diagonal_blocked,
                    dim=inv_diag.shape,
                    device=inv_diag.device,
                    inputs=[A_diag, inv_diag, use_abs],
                )
            else:
                inv_diag = wp.empty(shape=A.shape[0], dtype=A.scalar_type, device=A.device)
                wp.launch(
                    _extract_inverse_diagonal_scalar,
                    dim=inv_diag.shape,
                    device=inv_diag.device,
                    inputs=[A_diag, inv_diag, use_abs],
                )
        elif isinstance(A, wp.array) and A.ndim == 2:
            inv_diag = wp.empty(shape=A.shape[0], dtype=A.dtype, device=A.device)
            wp.launch(
                _extract_inverse_diagonal_dense,
                dim=inv_diag.shape,
                device=inv_diag.device,
                inputs=[A, inv_diag, use_abs],
            )
        else:
            raise ValueError("Unsupported source matrix type for building diagonal preconditioner")

        return aslinearoperator(inv_diag)

    raise ValueError(f"Unsupported preconditioner type '{ptype}'")


def cg(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: Optional[float] = None,
    atol: Optional[float] = None,
    maxiter: Optional[float] = 0,
    M: Optional[_Matrix] = None,
    callback: Optional[Callable] = None,
    check_every=10,
    use_cuda_graph=True,
) -> Tuple[int, float, float]:
    """Computes an approximate solution to a symmetric, positive-definite linear system
    using the Conjugate Gradient algorithm.

    Args:
        A: the linear system's left-hand-side
        b: the linear system's right-hand-side
        x: initial guess and solution vector
        tol: relative tolerance for the residual, as a ratio of the right-hand-side norm
        atol: absolute tolerance for the residual
        maxiter: maximum number of iterations to perform before aborting. Defaults to the system size.
            Note that the current implementation always performs iterations in pairs, and as a result may exceed the specified maximum number of iterations by one.
        M: optional left-preconditioner, ideally chosen such that ``M A`` is close to identity.
        callback: function to be called every `check_every` iteration with the current iteration number, residual and tolerance
        check_every: number of iterations every which to call `callback`, check the residual against the tolerance and possibility terminate the algorithm.
        use_cuda_graph: If true and when run on a CUDA device, capture the solver iteration as a CUDA graph for reduced launch overhead.
          The linear operator and preconditioner must only perform graph-friendly operations.

    Returns:
        Tuple (final iteration number, residual norm, absolute tolerance)

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """

    A = aslinearoperator(A)
    M = aslinearoperator(M)

    if maxiter == 0:
        maxiter = A.shape[0]

    r, r_norm_sq, atol = _initialize_residual_and_tolerance(A, b, x, tol=tol, atol=atol)

    device = A.device
    scalar_dtype = wp.types.type_scalar_type(A.dtype)

    # Notations below follow pseudo-code from https://en.wikipedia.org/wiki/Conjugate_gradient_method

    # z = M r
    if M is not None:
        z = wp.zeros_like(b)
        M.matvec(r, z, z, alpha=1.0, beta=0.0)

        # rz = r' z;
        rz_new = wp.empty(n=1, dtype=scalar_dtype, device=device)
        array_inner(r, z, out=rz_new)
    else:
        z = r

    rz_old = wp.empty(n=1, dtype=scalar_dtype, device=device)
    p_Ap = wp.empty(n=1, dtype=scalar_dtype, device=device)
    Ap = wp.zeros_like(b)

    p = wp.clone(z)

    def do_iteration(atol_sq, rr_old, rr_new, rz_old, rz_new):
        # Ap = A * p;
        A.matvec(p, Ap, Ap, alpha=1, beta=0)

        array_inner(p, Ap, out=p_Ap)

        wp.launch(
            kernel=_cg_kernel_1,
            dim=x.shape[0],
            device=device,
            inputs=[atol_sq, rr_old, rz_old, p_Ap, x, r, p, Ap],
        )
        array_inner(r, r, out=rr_new)

        # z = M r
        if M is not None:
            M.matvec(r, z, z, alpha=1.0, beta=0.0)
            # rz = r' z;
            array_inner(r, z, out=rz_new)

        wp.launch(kernel=_cg_kernel_2, dim=z.shape[0], device=device, inputs=[atol_sq, rr_new, rz_old, rz_new, z, p])

    # We do iterations by pairs, switching old and new residual norm buffers for each odd-even couple
    # In the non-preconditioned case we reuse the error norm buffer for the new <r,z> computation

    def do_odd_even_cycle(atol_sq: float):
        # A pair of iterations, so that we're swapping the residual buffers twice
        if M is None:
            do_iteration(atol_sq, r_norm_sq, rz_old, r_norm_sq, rz_old)
            do_iteration(atol_sq, rz_old, r_norm_sq, rz_old, r_norm_sq)
        else:
            do_iteration(atol_sq, r_norm_sq, r_norm_sq, rz_new, rz_old)
            do_iteration(atol_sq, r_norm_sq, r_norm_sq, rz_old, rz_new)

    return _run_solver_loop(
        do_odd_even_cycle,
        cycle_size=2,
        r_norm_sq=r_norm_sq,
        maxiter=maxiter,
        atol=atol,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
        device=device,
    )


def cr(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: Optional[float] = None,
    atol: Optional[float] = None,
    maxiter: Optional[float] = 0,
    M: Optional[_Matrix] = None,
    callback: Optional[Callable] = None,
    check_every=10,
    use_cuda_graph=True,
) -> Tuple[int, float, float]:
    """Computes an approximate solution to a symmetric, positive-definite linear system
    using the Conjugate Residual algorithm.

    Args:
        A: the linear system's left-hand-side
        b: the linear system's right-hand-side
        x: initial guess and solution vector
        tol: relative tolerance for the residual, as a ratio of the right-hand-side norm
        atol: absolute tolerance for the residual
        maxiter: maximum number of iterations to perform before aborting. Defaults to the system size.
            Note that the current implementation always performs iterations in pairs, and as a result may exceed the specified maximum number of iterations by one.
        M: optional left-preconditioner, ideally chosen such that ``M A`` is close to identity.
        callback: function to be called every `check_every` iteration with the current iteration number, residual and tolerance
        check_every: number of iterations every which to call `callback`, check the residual against the tolerance and possibility terminate the algorithm.
        use_cuda_graph: If true and when run on a CUDA device, capture the solver iteration as a CUDA graph for reduced launch overhead.
          The linear operator and preconditioner must only perform graph-friendly operations.

    Returns:
        Tuple (final iteration number, residual norm, absolute tolerance)

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """

    A = aslinearoperator(A)
    M = aslinearoperator(M)

    if maxiter == 0:
        maxiter = A.shape[0]

    r, r_norm_sq, atol = _initialize_residual_and_tolerance(A, b, x, tol=tol, atol=atol)

    device = A.device
    scalar_dtype = wp.types.type_scalar_type(A.dtype)

    # Notations below follow roughly pseudo-code from https://en.wikipedia.org/wiki/Conjugate_residual_method
    # with z := M^-1 r and y := M^-1 Ap

    # z = M r
    if M is None:
        z = r
    else:
        z = wp.zeros_like(r)
        M.matvec(r, z, z, alpha=1.0, beta=0.0)

    Az = wp.zeros_like(b)
    A.matvec(z, Az, Az, alpha=1, beta=0)

    p = wp.clone(z)
    Ap = wp.clone(Az)

    if M is None:
        y = Ap
    else:
        y = wp.zeros_like(Ap)

    zAz_old = wp.empty(n=1, dtype=scalar_dtype, device=device)
    zAz_new = wp.empty(n=1, dtype=scalar_dtype, device=device)
    y_Ap = wp.empty(n=1, dtype=scalar_dtype, device=device)

    array_inner(z, Az, out=zAz_new)

    def do_iteration(atol_sq, rr, zAz_old, zAz_new):
        if M is not None:
            M.matvec(Ap, y, y, alpha=1.0, beta=0.0)
        array_inner(Ap, y, out=y_Ap)

        if M is None:
            # In non-preconditioned case, first kernel is same as CG
            wp.launch(
                kernel=_cg_kernel_1,
                dim=x.shape[0],
                device=device,
                inputs=[atol_sq, rr, zAz_old, y_Ap, x, r, p, Ap],
            )
        else:
            # In preconditioned case, we have one more vector to update
            wp.launch(
                kernel=_cr_kernel_1,
                dim=x.shape[0],
                device=device,
                inputs=[atol_sq, rr, zAz_old, y_Ap, x, r, z, p, Ap, y],
            )

        array_inner(r, r, out=rr)

        A.matvec(z, Az, Az, alpha=1, beta=0)
        array_inner(z, Az, out=zAz_new)

        # beta = rz_new / rz_old
        wp.launch(
            kernel=_cr_kernel_2, dim=z.shape[0], device=device, inputs=[atol_sq, rr, zAz_old, zAz_new, z, p, Az, Ap]
        )

    # We do iterations by pairs, switching old and new residual norm buffers for each odd-even couple
    def do_odd_even_cycle(atol_sq: float):
        do_iteration(atol_sq, r_norm_sq, zAz_new, zAz_old)
        do_iteration(atol_sq, r_norm_sq, zAz_old, zAz_new)

    return _run_solver_loop(
        do_odd_even_cycle,
        cycle_size=2,
        r_norm_sq=r_norm_sq,
        maxiter=maxiter,
        atol=atol,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
        device=device,
    )


def bicgstab(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: Optional[float] = None,
    atol: Optional[float] = None,
    maxiter: Optional[float] = 0,
    M: Optional[_Matrix] = None,
    callback: Optional[Callable] = None,
    check_every=10,
    use_cuda_graph=True,
    is_left_preconditioner=False,
):
    """Computes an approximate solution to a linear system using the Biconjugate Gradient Stabilized method (BiCGSTAB).

    Args:
        A: the linear system's left-hand-side
        b: the linear system's right-hand-side
        x: initial guess and solution vector
        tol: relative tolerance for the residual, as a ratio of the right-hand-side norm
        atol: absolute tolerance for the residual
        maxiter: maximum number of iterations to perform before aborting. Defaults to the system size.
        M: optional left- or right-preconditioner, ideally chosen such that ``M A`` (resp ``A M``) is close to identity.
        callback: function to be called every `check_every` iteration with the current iteration number, residual and tolerance
        check_every: number of iterations every which to call `callback`, check the residual against the tolerance and possibility terminate the algorithm.
        use_cuda_graph: If true and when run on a CUDA device, capture the solver iteration as a CUDA graph for reduced launch overhead.
          The linear operator and preconditioner must only perform graph-friendly operations.
        is_left_preconditioner: whether `M` should be used as a left- or right- preconditioner.

    Returns:
        Tuple (final iteration number, residual norm, absolute tolerance)

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """
    A = aslinearoperator(A)
    M = aslinearoperator(M)

    if maxiter == 0:
        maxiter = A.shape[0]

    r, r_norm_sq, atol = _initialize_residual_and_tolerance(A, b, x, tol=tol, atol=atol)

    device = A.device
    scalar_dtype = wp.types.type_scalar_type(A.dtype)

    # Notations below follow pseudo-code from biconjugate https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    rho = wp.clone(r_norm_sq, pinned=False)
    r0v = wp.empty(n=1, dtype=scalar_dtype, device=device)
    st = wp.empty(n=1, dtype=scalar_dtype, device=device)
    tt = wp.empty(n=1, dtype=scalar_dtype, device=device)

    # work arrays
    r0 = wp.clone(r)
    v = wp.zeros_like(r)
    t = wp.zeros_like(r)
    p = wp.clone(r)

    if M is not None:
        y = wp.zeros_like(p)
        z = wp.zeros_like(r)
        if is_left_preconditioner:
            Mt = wp.zeros_like(t)
    else:
        y = p
        z = r
        Mt = t

    def do_iteration(atol_sq: float):
        # y = M p
        if M is not None:
            M.matvec(p, y, y, alpha=1.0, beta=0.0)

        # v = A * y;
        A.matvec(y, v, v, alpha=1, beta=0)

        # alpha = rho / <r0 . v>
        array_inner(r0, v, out=r0v)

        #  x += alpha y
        #  r -= alpha v
        wp.launch(
            kernel=_bicgstab_kernel_1,
            dim=x.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, rho, r0v, x, r, y, v],
        )
        array_inner(r, r, out=r_norm_sq)

        # z = M r
        if M is not None:
            M.matvec(r, z, z, alpha=1.0, beta=0.0)

        # t = A z
        A.matvec(z, t, t, alpha=1, beta=0)

        if is_left_preconditioner:
            # Mt = M t
            if M is not None:
                M.matvec(t, Mt, Mt, alpha=1.0, beta=0.0)

            # omega = <Mt, Ms> / <Mt, Mt>
            array_inner(z, Mt, out=st)
            array_inner(Mt, Mt, out=tt)
        else:
            array_inner(r, t, out=st)
            array_inner(t, t, out=tt)

        # x += omega z
        # r -= omega t
        wp.launch(
            kernel=_bicgstab_kernel_2,
            dim=z.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, st, tt, z, t, x, r],
        )
        array_inner(r, r, out=r_norm_sq)

        # rho = <r0, r>
        array_inner(r0, r, out=rho)

        # beta = (rho / rho_old) * alpha / omega = (rho / r0v) / omega
        # p = r + beta (p - omega v)
        wp.launch(
            kernel=_bicgstab_kernel_3,
            dim=z.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, rho, r0v, st, tt, p, r, v],
        )

    return _run_solver_loop(
        do_iteration,
        cycle_size=1,
        r_norm_sq=r_norm_sq,
        maxiter=maxiter,
        atol=atol,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
        device=device,
    )


def gmres(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: Optional[float] = None,
    atol: Optional[float] = None,
    restart=31,
    maxiter: Optional[float] = 0,
    M: Optional[_Matrix] = None,
    callback: Optional[Callable] = None,
    check_every=31,
    use_cuda_graph=True,
    is_left_preconditioner=False,
):
    """Computes an approximate solution to a linear system using the restarted Generalized Minimum Residual method (GMRES[k]).

    Args:
        A: the linear system's left-hand-side
        b: the linear system's right-hand-side
        x: initial guess and solution vector
        tol: relative tolerance for the residual, as a ratio of the right-hand-side norm
        atol: absolute tolerance for the residual
        restart: The restart parameter, i.e, the `k` in `GMRES[k]`. In general, increasing this parameter reduces the number of iterations but increases memory consumption.
        maxiter: maximum number of iterations to perform before aborting. Defaults to the system size.
            Note that the current implementation always perform `restart` iterations at a time, and as a result may exceed the specified maximum number of iterations by ``restart-1``.
        M: optional left- or right-preconditioner, ideally chosen such that ``M A`` (resp ``A M``) is close to identity.
        callback: function to be called every `check_every` iteration with the current iteration number, residual and tolerance
        check_every: number of iterations every which to call `callback`, check the residual against the tolerance and possibility terminate the algorithm.
        use_cuda_graph: If true and when run on a CUDA device, capture the solver iteration as a CUDA graph for reduced launch overhead.
          The linear operator and preconditioner must only perform graph-friendly operations.
        is_left_preconditioner: whether `M` should be used as a left- or right- preconditioner.

    Returns:
        Tuple (final iteration number, residual norm, absolute tolerance)

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """

    A = aslinearoperator(A)
    M = aslinearoperator(M)

    if maxiter == 0:
        maxiter = A.shape[0]

    restart = min(restart, maxiter)
    check_every = max(restart, check_every)

    r, r_norm_sq, atol = _initialize_residual_and_tolerance(A, b, x, tol=tol, atol=atol)

    device = A.device
    scalar_dtype = wp.types.type_scalar_type(A.dtype)

    pivot_tolerance = _get_dtype_epsilon(scalar_dtype) ** 2

    beta_sq = wp.empty_like(r_norm_sq, pinned=False)
    H = wp.empty(shape=(restart + 1, restart), dtype=scalar_dtype, device=device)

    y = wp.empty(shape=restart + 1, dtype=scalar_dtype, device=device)

    w = wp.zeros_like(r)
    V = wp.zeros(shape=(restart + 1, r.shape[0]), dtype=r.dtype, device=device)

    def array_coeff(H, i, j):
        return wp.array(
            ptr=H.ptr + i * H.strides[0] + j * H.strides[1],
            dtype=H.dtype,
            shape=(1,),
            device=H.device,
            copy=False,
        )

    def array_row(V, i):
        return wp.array(
            ptr=V.ptr + i * V.strides[0],
            dtype=V.dtype,
            shape=V.shape[1],
            device=V.device,
            copy=False,
        )

    def do_arnoldi_iteration(j: int):
        # w = A * v;

        vj = array_row(V, j)

        if M is not None:
            tmp = array_row(V, j + 1)

            if is_left_preconditioner:
                A.matvec(vj, tmp, tmp, alpha=1, beta=0)
                M.matvec(tmp, w, w, alpha=1, beta=0)
            else:
                M.matvec(vj, tmp, tmp, alpha=1, beta=0)
                A.matvec(tmp, w, w, alpha=1, beta=0)
        else:
            A.matvec(vj, w, w, alpha=1, beta=0)

        for i in range(j + 1):
            vi = array_row(V, i)
            hij = array_coeff(H, i, j)
            array_inner(w, vi, out=hij)

            wp.launch(_gmres_arnoldi_axpy_kernel, dim=w.shape, device=w.device, inputs=[vi, w, hij])

        hjnj = array_coeff(H, j + 1, j)
        array_inner(w, w, out=hjnj)

        vjn = array_row(V, j + 1)
        wp.launch(_gmres_arnoldi_normalize_kernel, dim=w.shape, device=w.device, inputs=[w, vjn, hjnj])

    def do_restart_cycle(atol_sq: float):
        if M is not None and is_left_preconditioner:
            M.matvec(r, w, w, alpha=1, beta=0)
            rh = w
        else:
            rh = r

        array_inner(rh, rh, out=beta_sq)

        v0 = array_row(V, 0)
        # v0 = r / beta
        wp.launch(_gmres_arnoldi_normalize_kernel, dim=r.shape, device=r.device, inputs=[rh, v0, beta_sq])

        for j in range(restart):
            do_arnoldi_iteration(j)

        wp.launch(_gmres_normalize_lower_diagonal, dim=restart, device=device, inputs=[H])
        wp.launch(_gmres_solve_least_squares, dim=1, device=device, inputs=[restart, pivot_tolerance, beta_sq, H, y])

        # update x
        if M is None or is_left_preconditioner:
            wp.launch(_gmres_update_x_kernel, dim=x.shape, device=device, inputs=[restart, scalar_dtype(1.0), y, V, x])
        else:
            wp.launch(_gmres_update_x_kernel, dim=x.shape, device=device, inputs=[restart, scalar_dtype(0.0), y, V, w])
            M.matvec(w, x, x, alpha=1, beta=1)

        # update r and residual
        wp.copy(src=b, dest=r)
        A.matvec(x, b, r, alpha=-1.0, beta=1.0)
        array_inner(r, r, out=r_norm_sq)

    return _run_solver_loop(
        do_restart_cycle,
        cycle_size=restart,
        r_norm_sq=r_norm_sq,
        maxiter=maxiter,
        atol=atol,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
        device=device,
    )


def _get_dtype_epsilon(dtype):
    if dtype == wp.float64:
        return 1.0e-16
    elif dtype == wp.float16:
        return 1.0e-4

    return 1.0e-8


def _get_absolute_tolerance(dtype, tol, atol, lhs_norm):
    eps_tol = _get_dtype_epsilon(dtype)
    default_tol = eps_tol ** (3 / 4)
    min_tol = eps_tol ** (9 / 4)

    if tol is None and atol is None:
        tol = atol = default_tol
    elif tol is None:
        tol = atol
    elif atol is None:
        atol = tol

    return max(tol * lhs_norm, atol, min_tol)


def _initialize_residual_and_tolerance(A: LinearOperator, b: wp.array, x: wp.array, tol: float, atol: float):
    scalar_dtype = wp.types.type_scalar_type(A.dtype)
    device = A.device

    # Buffer for storing square norm or residual
    r_norm_sq = wp.empty(n=1, dtype=scalar_dtype, device=device, pinned=device.is_cuda)

    # Compute b norm to define absolute tolerance
    array_inner(b, b, out=r_norm_sq)
    atol = _get_absolute_tolerance(scalar_dtype, tol, atol, sqrt(r_norm_sq.numpy()[0]))

    # Residual r = b - Ax
    r = wp.empty_like(b)
    A.matvec(x, b, r, alpha=-1.0, beta=1.0)

    array_inner(r, r, out=r_norm_sq)

    return r, r_norm_sq, atol


def _run_solver_loop(
    do_cycle: Callable[[float], None],
    cycle_size: int,
    r_norm_sq: wp.array,
    maxiter: int,
    atol: float,
    callback: Callable,
    check_every: int,
    use_cuda_graph: bool,
    device,
):
    atol_sq = atol * atol

    cur_iter = 0

    err_sq = r_norm_sq.numpy()[0]
    err = sqrt(err_sq)
    if callback is not None:
        callback(cur_iter, err, atol)

    if err_sq <= atol_sq:
        return cur_iter, err, atol

    graph = None

    while True:
        # Do not do graph capture at first iteration -- modules may not be loaded yet
        if device.is_cuda and use_cuda_graph and cur_iter > 0:
            if graph is None:
                wp.capture_begin(device, force_module_load=False)
                try:
                    do_cycle(atol_sq)
                finally:
                    graph = wp.capture_end(device)
            wp.capture_launch(graph)
        else:
            do_cycle(atol_sq)

        cur_iter += cycle_size

        if cur_iter >= maxiter:
            break

        if (cur_iter % check_every) < cycle_size:
            err_sq = r_norm_sq.numpy()[0]

            if err_sq <= atol_sq:
                break

            if callback is not None:
                callback(cur_iter, sqrt(err_sq), atol)

    err_sq = r_norm_sq.numpy()[0]
    err = sqrt(err_sq)
    if callback is not None:
        callback(cur_iter, err, atol)

    return cur_iter, err, atol


@wp.func
def _calc_mv_product(i: wp.int32, A: wp.array2d(dtype=Any), x: wp.array1d(dtype=Any)):
    sum = A.dtype(0)
    for j in range(A.shape[1]):
        sum += A[i, j] * x[j]
    return sum


@wp.kernel
def _dense_mv_kernel(
    A: wp.array2d(dtype=Any),
    x: wp.array1d(dtype=Any),
    y: wp.array1d(dtype=Any),
    z: wp.array1d(dtype=Any),
    alpha: Any,
    beta: Any,
):
    i = wp.tid()
    z[i] = z.dtype(beta) * y[i] + z.dtype(alpha) * _calc_mv_product(i, A, x)


@wp.kernel
def _diag_mv_kernel(
    A: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    alpha: Any,
    beta: Any,
):
    i = wp.tid()
    z[i] = beta * y[i] + alpha * (A[i] * x[i])


@wp.kernel
def _diag_mv_vec_kernel(
    A: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    alpha: Any,
    beta: Any,
):
    i = wp.tid()
    z[i] = beta * y[i] + alpha * wp.cw_mul(A[i], x[i])


@wp.func
def _inverse_diag_coefficient(coeff: Any, use_abs: wp.bool):
    zero = type(coeff)(0.0)
    one = type(coeff)(1.0)
    return wp.where(coeff == zero, one, one / wp.where(use_abs, wp.abs(coeff), coeff))


@wp.kernel
def _extract_inverse_diagonal_blocked(
    diag_block: wp.array(dtype=Any),
    inv_diag: wp.array(dtype=Any),
    use_abs: int,
):
    i = wp.tid()

    d = wp.get_diag(diag_block[i])
    for k in range(d.length):
        d[k] = _inverse_diag_coefficient(d[k], use_abs != 0)

    inv_diag[i] = d


@wp.kernel
def _extract_inverse_diagonal_scalar(
    diag_array: wp.array(dtype=Any),
    inv_diag: wp.array(dtype=Any),
    use_abs: int,
):
    i = wp.tid()
    inv_diag[i] = _inverse_diag_coefficient(diag_array[i], use_abs != 0)


@wp.kernel
def _extract_inverse_diagonal_dense(
    dense_matrix: wp.array2d(dtype=Any),
    inv_diag: wp.array(dtype=Any),
    use_abs: int,
):
    i = wp.tid()
    inv_diag[i] = _inverse_diag_coefficient(dense_matrix[i, i], use_abs != 0)


@wp.kernel
def _cg_kernel_1(
    tol: Any,
    resid: wp.array(dtype=Any),
    rz_old: wp.array(dtype=Any),
    p_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
):
    i = wp.tid()

    alpha = wp.where(resid[0] > tol, rz_old[0] / p_Ap[0], rz_old.dtype(0.0))

    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]


@wp.kernel
def _cg_kernel_2(
    tol: Any,
    resid: wp.array(dtype=Any),
    rz_old: wp.array(dtype=Any),
    rz_new: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
):
    #    p = r + (rz_new / rz_old) * p;
    i = wp.tid()

    beta = wp.where(resid[0] > tol, rz_new[0] / rz_old[0], rz_old.dtype(0.0))

    p[i] = z[i] + beta * p[i]


@wp.kernel
def _cr_kernel_1(
    tol: Any,
    resid: wp.array(dtype=Any),
    zAz_old: wp.array(dtype=Any),
    y_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
):
    i = wp.tid()

    alpha = wp.where(resid[0] > tol and y_Ap[0] > 0.0, zAz_old[0] / y_Ap[0], zAz_old.dtype(0.0))

    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]
    z[i] = z[i] - alpha * y[i]


@wp.kernel
def _cr_kernel_2(
    tol: Any,
    resid: wp.array(dtype=Any),
    zAz_old: wp.array(dtype=Any),
    zAz_new: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Az: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
):
    #    p = r + (rz_new / rz_old) * p;
    i = wp.tid()

    beta = wp.where(resid[0] > tol and zAz_old[0] > 0.0, zAz_new[0] / zAz_old[0], zAz_old.dtype(0.0))

    p[i] = z[i] + beta * p[i]
    Ap[i] = Az[i] + beta * Ap[i]


@wp.kernel
def _bicgstab_kernel_1(
    tol: Any,
    resid: wp.array(dtype=Any),
    rho_old: wp.array(dtype=Any),
    r0v: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    v: wp.array(dtype=Any),
):
    i = wp.tid()

    alpha = wp.where(resid[0] > tol, rho_old[0] / r0v[0], rho_old.dtype(0.0))

    x[i] += alpha * y[i]
    r[i] -= alpha * v[i]


@wp.kernel
def _bicgstab_kernel_2(
    tol: Any,
    resid: wp.array(dtype=Any),
    st: wp.array(dtype=Any),
    tt: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    t: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
):
    i = wp.tid()

    omega = wp.where(resid[0] > tol, st[0] / tt[0], st.dtype(0.0))

    x[i] += omega * z[i]
    r[i] -= omega * t[i]


@wp.kernel
def _bicgstab_kernel_3(
    tol: Any,
    resid: wp.array(dtype=Any),
    rho_new: wp.array(dtype=Any),
    r0v: wp.array(dtype=Any),
    st: wp.array(dtype=Any),
    tt: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    v: wp.array(dtype=Any),
):
    i = wp.tid()

    beta = wp.where(resid[0] > tol, rho_new[0] * tt[0] / (r0v[0] * st[0]), st.dtype(0.0))
    beta_omega = wp.where(resid[0] > tol, rho_new[0] / r0v[0], st.dtype(0.0))

    p[i] = r[i] + beta * p[i] - beta_omega * v[i]


@wp.kernel
def _gmres_normalize_lower_diagonal(H: wp.array2d(dtype=Any)):
    # normalize lower-diagonal values of Hessenberg matrix
    i = wp.tid()
    H[i + 1, i] = wp.sqrt(H[i + 1, i])


@wp.kernel
def _gmres_solve_least_squares(
    k: int, pivot_tolerance: float, beta_sq: wp.array(dtype=Any), H: wp.array2d(dtype=Any), y: wp.array(dtype=Any)
):
    # Solve H y = (beta, 0, ..., 0)
    # H Hessenberg matrix of shape (k+1, k)

    # Keeping H in global mem; warp kernels are launched with fixed block size,
    # so would not fit in registers

    # TODO: switch to native code with thread synchronization

    rhs = wp.sqrt(beta_sq[0])

    # Apply 2x2 rotations to H so as to remove lower diagonal,
    # and apply similar rotations to right-hand-side
    max_k = int(k)
    for i in range(k):
        Ha = H[i]
        Hb = H[i + 1]

        # Givens rotation [[c s], [-s c]]
        a = Ha[i]
        b = Hb[i]
        abn_sq = a * a + b * b

        if abn_sq < type(abn_sq)(pivot_tolerance):
            # Arnoldi iteration finished early
            max_k = i
            break

        abn = wp.sqrt(abn_sq)
        c = a / abn
        s = b / abn

        # Rotate H
        for j in range(i, k):
            a = Ha[j]
            b = Hb[j]
            Ha[j] = c * a + s * b
            Hb[j] = c * b - s * a

        # Rotate rhs
        y[i] = c * rhs
        rhs = -s * rhs

    for i in range(max_k, k):
        y[i] = y.dtype(0.0)

    # Triangular back-solve for y
    for ii in range(max_k, 0, -1):
        i = ii - 1
        Hi = H[i]
        yi = y[i]
        for j in range(ii, max_k):
            yi -= Hi[j] * y[j]
        y[i] = yi / Hi[i]


@wp.kernel
def _gmres_arnoldi_axpy_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
):
    tid = wp.tid()
    y[tid] -= x[tid] * alpha[0]


@wp.kernel
def _gmres_arnoldi_normalize_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
):
    tid = wp.tid()
    y[tid] = wp.where(alpha[0] == alpha.dtype(0.0), x[tid], x[tid] / wp.sqrt(alpha[0]))


@wp.kernel
def _gmres_update_x_kernel(k: int, beta: Any, y: wp.array(dtype=Any), V: wp.array2d(dtype=Any), x: wp.array(dtype=Any)):
    tid = wp.tid()

    xi = beta * x[tid]
    for j in range(k):
        xi += V[j, tid] * y[j]

    x[tid] = xi
