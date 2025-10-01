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

import functools
import math
from typing import Any, Callable, Optional, Tuple, Union

import warp as wp
import warp.sparse as sparse
from warp.types import type_length, type_scalar_type

__all__ = ["LinearOperator", "aslinearoperator", "bicgstab", "cg", "cr", "gmres", "preconditioner"]

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
            '''Perform a generalized matrix-vector product.

            This function computes the operation z = alpha * (A @ x) + beta * y, where 'A'
            is the linear operator represented by this class.
            '''
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
        alpha = A.dtype(alpha)
        beta = A.dtype(beta)
        if A.device.is_cuda:
            tile_size = 1 << min(10, max(5, math.ceil(math.log2(A.shape[1]))))
        else:
            tile_size = 1
        wp.launch(
            _dense_mv_kernel,
            dim=(A.shape[0], tile_size),
            block_dim=tile_size,
            device=A.device,
            inputs=[A, x, y, z, alpha, beta],
        )

    def diag_mv_impl(A, x, y, z, alpha, beta):
        scalar_type = type_scalar_type(A.dtype)
        alpha = scalar_type(alpha)
        beta = scalar_type(beta)
        wp.launch(_diag_mv_kernel, dim=A.shape, device=A.device, inputs=[A, x, y, z, alpha, beta])

    def diag_mv(x, y, z, alpha, beta):
        return diag_mv_impl(A, x, y, z, alpha, beta)

    def diag_mv_vec(x, y, z, alpha, beta):
        return diag_mv_impl(
            _as_scalar_array(A), _as_scalar_array(x), _as_scalar_array(y), _as_scalar_array(z), alpha, beta
        )

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


def _as_scalar_array(x: wp.array):
    scalar_type = type_scalar_type(x.dtype)
    if scalar_type == x.dtype:
        return x

    dlen = type_length(x.dtype)
    arr = wp.array(
        ptr=x.ptr,
        shape=(*x.shape[:-1], x.shape[-1] * dlen),
        strides=(*x.strides[:-1], x.strides[-1] // dlen),
        dtype=scalar_type,
        device=x.device,
        grad=None if x.grad is None else _as_scalar_array(x.grad),
    )
    arr._ref = x
    return arr


class TiledDot:
    """
    Computes the dot product of two arrays in a way that is compatible with CUDA sub-graphs.
    """

    def __init__(self, max_length: int, scalar_type: type, tile_size=512, device=None, max_column_count: int = 1):
        self.tile_size = tile_size
        self.device = device
        self.max_column_count = max_column_count

        num_blocks = (max_length + self.tile_size - 1) // self.tile_size
        scratch = wp.empty(
            shape=(2, max_column_count, num_blocks),
            dtype=scalar_type,
            device=self.device,
        )
        self.partial_sums_a = scratch[0]
        self.partial_sums_b = scratch[1]

        self.dot_kernel, self.sum_kernel = _create_tiled_dot_kernels(self.tile_size)

        rounds = 0
        length = num_blocks
        while length > 1:
            length = (length + self.tile_size - 1) // self.tile_size
            rounds += 1

        self.rounds = rounds

        self._output = self.partial_sums_a if rounds % 2 == 0 else self.partial_sums_b

        self.dot_launch: wp.Launch = wp.launch(
            self.dot_kernel,
            dim=(max_column_count, num_blocks, self.tile_size),
            inputs=(self.partial_sums_a, self.partial_sums_b),
            outputs=(self.partial_sums_a,),
            block_dim=self.tile_size,
            record_cmd=True,
        )
        self.sum_launch: wp.Launch = wp.launch(
            self.sum_kernel,
            dim=(max_column_count, num_blocks, self.tile_size),
            inputs=(self.partial_sums_a,),
            outputs=(self.partial_sums_b,),
            block_dim=self.tile_size,
            record_cmd=True,
        )

    # Result contains a single value, the sum of the array (will get updated by this function)
    def compute(self, a: wp.array, b: wp.array, col_offset: int = 0):
        a = _as_scalar_array(a)
        b = _as_scalar_array(b)
        if a.ndim == 1:
            a = a.reshape((1, -1))
        if b.ndim == 1:
            b = b.reshape((1, -1))

        column_count = a.shape[0]
        num_blocks = (a.shape[1] + self.tile_size - 1) // self.tile_size

        data_out = self.partial_sums_a[col_offset : col_offset + column_count]
        data_in = self.partial_sums_b[col_offset : col_offset + column_count]

        self.dot_launch.set_param_at_index(0, a)
        self.dot_launch.set_param_at_index(1, b)
        self.dot_launch.set_param_at_index(2, data_out)
        self.dot_launch.set_dim((column_count, num_blocks, self.tile_size))
        self.dot_launch.launch()

        for _r in range(self.rounds):
            array_length = num_blocks
            num_blocks = (array_length + self.tile_size - 1) // self.tile_size
            data_in, data_out = data_out, data_in

            self.sum_launch.set_param_at_index(0, data_in)
            self.sum_launch.set_param_at_index(1, data_out)
            self.sum_launch.set_dim((column_count, num_blocks, self.tile_size))
            self.sum_launch.launch()

        return data_out

    def col(self, col: int = 0):
        return self._output[col][:1]

    def cols(self, count, start: int = 0):
        return self._output[start : start + count, :1]


@functools.lru_cache(maxsize=None)
def _create_tiled_dot_kernels(tile_size):
    @wp.kernel
    def block_dot_kernel(
        a: wp.array2d(dtype=Any),
        b: wp.array2d(dtype=Any),
        partial_sums: wp.array2d(dtype=Any),
    ):
        column, block_id, tid_block = wp.tid()

        start = block_id * tile_size

        a_block = wp.tile_load(a[column], shape=tile_size, offset=start)
        b_block = wp.tile_load(b[column], shape=tile_size, offset=start)
        t = wp.tile_map(wp.mul, a_block, b_block)

        tile_sum = wp.tile_sum(t)
        wp.tile_store(partial_sums[column], tile_sum, offset=block_id)

    @wp.kernel
    def block_sum_kernel(
        data: wp.array2d(dtype=Any),
        partial_sums: wp.array2d(dtype=Any),
    ):
        column, block_id, tid_block = wp.tid()
        start = block_id * tile_size

        t = wp.tile_load(data[column], shape=tile_size, offset=start)

        tile_sum = wp.tile_sum(t)
        wp.tile_store(partial_sums[column], tile_sum, offset=block_id)

    return block_dot_kernel, block_sum_kernel


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
) -> Union[Tuple[int, float, float], Tuple[wp.array, wp.array, wp.array]]:
    """Computes an approximate solution to a symmetric, positive-definite linear system
    using the Conjugate Gradient algorithm.

    Args:
        A: the linear system's left-hand-side
        b: the linear system's right-hand-side
        x: initial guess and solution vector
        tol: relative tolerance for the residual, as a ratio of the right-hand-side norm
        atol: absolute tolerance for the residual
        maxiter: maximum number of iterations to perform before aborting. Defaults to the system size.
        M: optional left-preconditioner, ideally chosen such that ``M A`` is close to identity.
        callback: function to be called every `check_every` iteration with the current iteration number, residual and tolerance.
            If `check_every` is 0, the callback should be a Warp kernel.
        check_every: number of iterations every which to call `callback`, check the residual against the tolerance and possibility terminate the algorithm.
            Setting `check_every` to 0 disables host-side residual checks, making the solver fully CUDA-graph capturable.
            If conditional CUDA graphs are supported, convergence checks are performed device-side; otherwise, the solver will always run
            to the maximum number of iterations.
        use_cuda_graph: If true and when run on a CUDA device, capture the solver iteration as a CUDA graph for reduced launch overhead.
            The linear operator and preconditioner must only perform graph-friendly operations.

    Returns:
        If `check_every` > 0: Tuple (final_iteration, residual_norm, absolute_tolerance)
            - final_iteration: The number of iterations performed before convergence or reaching maxiter
            - residual_norm: The final residual norm ||b - Ax||
            - absolute_tolerance: The absolute tolerance used for convergence checking

        If `check_every` is 0: Tuple (final_iteration_array, residual_norm_squared_array, absolute_tolerance_squared_array)
            - final_iteration_array: Device array containing the number of iterations performed
            - residual_norm_squared_array: Device array containing the squared residual norm ||b - Ax||²
            - absolute_tolerance_squared_array: Device array containing the squared absolute tolerance

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """
    A = aslinearoperator(A)
    M = aslinearoperator(M)

    if maxiter == 0:
        maxiter = A.shape[0]

    device = A.device
    scalar_type = A.scalar_type

    # Temp storage
    r_and_z = wp.empty((2, b.shape[0]), dtype=b.dtype, device=device)
    p_and_Ap = wp.empty_like(r_and_z)
    residuals = wp.empty(2, dtype=scalar_type, device=device)

    tiled_dot = TiledDot(max_length=A.shape[0], device=device, scalar_type=scalar_type, max_column_count=2)

    # named views

    # (r, r) -- so we can compute r.z and r.r at once
    r_repeated = _repeat_first(r_and_z)
    if M is None:
        # without preconditioner r == z
        r_and_z = r_repeated
        rz_new = tiled_dot.col(0)
    else:
        rz_new = tiled_dot.col(1)

    r, z = r_and_z[0], r_and_z[1]
    r_norm_sq = tiled_dot.col(0)

    p, Ap = p_and_Ap[0], p_and_Ap[1]
    rz_old, atol_sq = residuals[0:1], residuals[1:2]

    # Not strictly necessary, but makes it more robust to user-provided LinearOperators
    Ap.zero_()
    z.zero_()

    # Initialize tolerance from right-hand-side norm
    _initialize_absolute_tolerance(b, tol, atol, tiled_dot, atol_sq)
    # Initialize residual
    A.matvec(x, b, r, alpha=-1.0, beta=1.0)

    def update_rr_rz():
        # z = M r
        if M is None:
            tiled_dot.compute(r, r)
        else:
            M.matvec(r, z, z, alpha=1.0, beta=0.0)
            tiled_dot.compute(r_repeated, r_and_z)

    update_rr_rz()
    p.assign(z)

    def do_iteration():
        rz_old.assign(rz_new)

        # Ap = A * p;
        A.matvec(p, Ap, Ap, alpha=1, beta=0)
        tiled_dot.compute(p, Ap, col_offset=1)
        p_Ap = tiled_dot.col(1)

        wp.launch(
            kernel=_cg_kernel_1,
            dim=x.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, rz_old, p_Ap, x, r, p, Ap],
        )

        update_rr_rz()

        wp.launch(
            kernel=_cg_kernel_2,
            dim=z.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, rz_old, rz_new, z, p],
        )

    return _run_capturable_loop(do_iteration, r_norm_sq, maxiter, atol_sq, callback, check_every, use_cuda_graph)


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
        callback: function to be called every `check_every` iteration with the current iteration number, residual and tolerance.
            If `check_every` is 0, the callback should be a Warp kernel.
        check_every: number of iterations every which to call `callback`, check the residual against the tolerance and possibility terminate the algorithm.
            Setting `check_every` to 0 disables host-side residual checks, making the solver fully CUDA-graph capturable.
            If conditional CUDA graphs are supported, convergence checks are performed device-side; otherwise, the solver will always run
            to the maximum number of iterations.
        use_cuda_graph: If true and when run on a CUDA device, capture the solver iteration as a CUDA graph for reduced launch overhead.
          The linear operator and preconditioner must only perform graph-friendly operations.

    Returns:
        If `check_every` > 0: Tuple (final_iteration, residual_norm, absolute_tolerance)
            - final_iteration: The number of iterations performed before convergence or reaching maxiter
            - residual_norm: The final residual norm ||b - Ax||
            - absolute_tolerance: The absolute tolerance used for convergence checking

        If `check_every` is 0: Tuple (final_iteration_array, residual_norm_squared_array, absolute_tolerance_squared_array)
            - final_iteration_array: Device array containing the number of iterations performed
            - residual_norm_squared_array: Device array containing the squared residual norm ||b - Ax||²
            - absolute_tolerance_squared_array: Device array containing the squared absolute tolerance

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """

    A = aslinearoperator(A)
    M = aslinearoperator(M)

    if maxiter == 0:
        maxiter = A.shape[0]

    device = A.device
    scalar_type = wp.types.type_scalar_type(A.dtype)

    # Notations below follow roughly pseudo-code from https://en.wikipedia.org/wiki/Conjugate_residual_method
    # with z := M^-1 r and y := M^-1 Ap

    # Temp storage
    r_and_z = wp.empty((2, b.shape[0]), dtype=b.dtype, device=device)
    r_and_Az = wp.empty_like(r_and_z)
    y_and_Ap = wp.empty_like(r_and_z)
    p = wp.empty_like(b)
    residuals = wp.empty(2, dtype=scalar_type, device=device)

    tiled_dot = TiledDot(max_length=A.shape[0], device=device, scalar_type=scalar_type, max_column_count=2)

    if M is None:
        r_and_z = _repeat_first(r_and_z)
        y_and_Ap = _repeat_first(y_and_Ap)

    # named views
    r, z = r_and_z[0], r_and_z[1]
    r_copy, Az = r_and_Az[0], r_and_Az[1]

    y, Ap = y_and_Ap[0], y_and_Ap[1]

    r_norm_sq = tiled_dot.col(0)
    zAz_new = tiled_dot.col(1)
    zAz_old, atol_sq = residuals[0:1], residuals[1:2]

    # Initialize tolerance from right-hand-side norm
    _initialize_absolute_tolerance(b, tol, atol, tiled_dot, atol_sq)
    # Initialize residual
    A.matvec(x, b, r, alpha=-1.0, beta=1.0)

    # Not strictly necessary, but makes it more robust to user-provided LinearOperators
    y_and_Ap.zero_()

    # z = M r
    if M is not None:
        z.zero_()
        M.matvec(r, z, z, alpha=1.0, beta=0.0)

    def update_rr_zAz():
        A.matvec(z, Az, Az, alpha=1, beta=0)
        r_copy.assign(r)
        tiled_dot.compute(r_and_z, r_and_Az)

    update_rr_zAz()

    p.assign(z)
    Ap.assign(Az)

    def do_iteration():
        zAz_old.assign(zAz_new)

        if M is not None:
            M.matvec(Ap, y, y, alpha=1.0, beta=0.0)
        tiled_dot.compute(Ap, y, col_offset=1)
        y_Ap = tiled_dot.col(1)

        if M is None:
            # In non-preconditioned case, first kernel is same as CG
            wp.launch(
                kernel=_cg_kernel_1,
                dim=x.shape[0],
                device=device,
                inputs=[atol_sq, r_norm_sq, zAz_old, y_Ap, x, r, p, Ap],
            )
        else:
            # In preconditioned case, we have one more vector to update
            wp.launch(
                kernel=_cr_kernel_1,
                dim=x.shape[0],
                device=device,
                inputs=[atol_sq, r_norm_sq, zAz_old, y_Ap, x, r, z, p, Ap, y],
            )

        update_rr_zAz()
        wp.launch(
            kernel=_cr_kernel_2,
            dim=z.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, zAz_old, zAz_new, z, p, Az, Ap],
        )

    return _run_capturable_loop(
        do_iteration,
        cycle_size=1,
        r_norm_sq=r_norm_sq,
        maxiter=maxiter,
        atol_sq=atol_sq,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
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
        callback: function to be called every `check_every` iteration with the current iteration number, residual and tolerance.
            If `check_every` is 0, the callback should be a Warp kernel.
        check_every: number of iterations every which to call `callback`, check the residual against the tolerance and possibility terminate the algorithm.
            Setting `check_every` to 0 disables host-side residual checks, making the solver fully CUDA-graph capturable.
            If conditional CUDA graphs are supported, convergence checks are performed device-side; otherwise, the solver will always run
            to the maximum number of iterations.
        use_cuda_graph: If true and when run on a CUDA device, capture the solver iteration as a CUDA graph for reduced launch overhead.
            The linear operator and preconditioner must only perform graph-friendly operations.
        is_left_preconditioner: whether `M` should be used as a left- or right- preconditioner.

    Returns:
        If `check_every` > 0: Tuple (final_iteration, residual_norm, absolute_tolerance)
            - final_iteration: The number of iterations performed before convergence or reaching maxiter
            - residual_norm: The final residual norm ||b - Ax||
            - absolute_tolerance: The absolute tolerance used for convergence checking

        If `check_every` is 0: Tuple (final_iteration_array, residual_norm_squared_array, absolute_tolerance_squared_array)
            - final_iteration_array: Device array containing the number of iterations performed
            - residual_norm_squared_array: Device array containing the squared residual norm ||b - Ax||²
            - absolute_tolerance_squared_array: Device array containing the squared absolute tolerance

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """
    A = aslinearoperator(A)
    M = aslinearoperator(M)

    if maxiter == 0:
        maxiter = A.shape[0]

    device = A.device
    scalar_type = wp.types.type_scalar_type(A.dtype)

    # Notations below follow pseudo-code from biconjugate https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    # Temp storage
    r_and_r0 = wp.empty((2, b.shape[0]), dtype=b.dtype, device=device)
    p = wp.empty_like(b)
    v = wp.empty_like(b)
    t = wp.empty_like(b)

    r, r0 = r_and_r0[0], r_and_r0[1]
    r_repeated = _repeat_first(r_and_r0)

    if M is not None:
        y = wp.zeros_like(p)
        z = wp.zeros_like(r)
        if is_left_preconditioner:
            Mt = wp.zeros_like(t)
    else:
        y = p
        z = r
        Mt = t

    tiled_dot = TiledDot(max_length=A.shape[0], device=device, scalar_type=scalar_type, max_column_count=5)
    r_norm_sq = tiled_dot.col(0)
    rho = tiled_dot.col(1)

    atol_sq = wp.empty(1, dtype=scalar_type, device=device)

    # Initialize tolerance from right-hand-side norm
    _initialize_absolute_tolerance(b, tol, atol, tiled_dot, atol_sq)
    # Initialize residual
    A.matvec(x, b, r, alpha=-1.0, beta=1.0)
    tiled_dot.compute(r, r, col_offset=0)

    p.assign(r)
    r0.assign(r)
    rho.assign(r_norm_sq)

    # Not strictly necessary, but makes it more robust to user-provided LinearOperators
    v.zero_()
    t.zero_()

    def do_iteration():
        # y = M p
        if M is not None:
            M.matvec(p, y, y, alpha=1.0, beta=0.0)

        # v = A * y;
        A.matvec(y, v, v, alpha=1, beta=0)

        # alpha = rho / <r0 . v>
        tiled_dot.compute(r0, v, col_offset=2)
        r0v = tiled_dot.col(2)

        #  x += alpha y
        #  r -= alpha v
        wp.launch(
            kernel=_bicgstab_kernel_1,
            dim=x.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, rho, r0v, x, r, y, v],
        )
        tiled_dot.compute(r, r, col_offset=0)

        # z = M r
        if M is not None:
            M.matvec(r, z, z, alpha=1.0, beta=0.0)

        # t = A z
        A.matvec(z, t, t, alpha=1, beta=0)

        if M is not None and is_left_preconditioner:
            # Mt = M t
            M.matvec(t, Mt, Mt, alpha=1.0, beta=0.0)

            # omega = <Mt, Ms> / <Mt, Mt>
            tiled_dot.compute(z, Mt, col_offset=3)
            tiled_dot.compute(Mt, Mt, col_offset=4)
        else:
            tiled_dot.compute(r, t, col_offset=3)
            tiled_dot.compute(t, t, col_offset=4)
        st = tiled_dot.col(3)
        tt = tiled_dot.col(4)

        # x += omega z
        # r -= omega t
        wp.launch(
            kernel=_bicgstab_kernel_2,
            dim=z.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, st, tt, z, t, x, r],
        )

        # r = <r,r>, rho = <r0, r>
        tiled_dot.compute(r_and_r0, r_repeated, col_offset=0)

        # beta = (rho / rho_old) * alpha / omega = (rho / r0v) / omega
        # p = r + beta (p - omega v)
        wp.launch(
            kernel=_bicgstab_kernel_3,
            dim=z.shape[0],
            device=device,
            inputs=[atol_sq, r_norm_sq, rho, r0v, st, tt, p, r, v],
        )

    return _run_capturable_loop(
        do_iteration,
        r_norm_sq=r_norm_sq,
        maxiter=maxiter,
        atol_sq=atol_sq,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
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
        callback: function to be called every `check_every` iteration with the current iteration number, residual and tolerance.
            If `check_every` is 0, the callback should be a Warp kernel.
        check_every: number of iterations every which to call `callback`, check the residual against the tolerance and possibility terminate the algorithm.
            Setting `check_every` to 0 disables host-side residual checks, making the solver fully CUDA-graph capturable.
            If conditional CUDA graphs are supported, convergence checks are performed device-side; otherwise, the solver will always run
            to the maximum number of iterations.
        use_cuda_graph: If true and when run on a CUDA device, capture the solver iteration as a CUDA graph for reduced launch overhead.
          The linear operator and preconditioner must only perform graph-friendly operations.
        is_left_preconditioner: whether `M` should be used as a left- or right- preconditioner.

    Returns:
        If `check_every` > 0: Tuple (final_iteration, residual_norm, absolute_tolerance)
            - final_iteration: The number of iterations performed before convergence or reaching maxiter
            - residual_norm: The final residual norm ||b - Ax||
            - absolute_tolerance: The absolute tolerance used for convergence checking

        If `check_every` is 0: Tuple (final_iteration_array, residual_norm_squared_array, absolute_tolerance_squared_array)
            - final_iteration_array: Device array containing the number of iterations performed
            - residual_norm_squared_array: Device array containing the squared residual norm ||b - Ax||²
            - absolute_tolerance_squared_array: Device array containing the squared absolute tolerance

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """

    A = aslinearoperator(A)
    M = aslinearoperator(M)

    if maxiter == 0:
        maxiter = A.shape[0]

    restart = min(restart, maxiter)

    if check_every > 0:
        check_every = max(restart, check_every)

    device = A.device
    scalar_dtype = wp.types.type_scalar_type(A.dtype)

    pivot_tolerance = _get_dtype_epsilon(scalar_dtype) ** 2

    r = wp.empty_like(b)
    w = wp.empty_like(r)

    H = wp.empty(shape=(restart + 1, restart), dtype=scalar_dtype, device=device)
    y = wp.empty(shape=restart + 1, dtype=scalar_dtype, device=device)

    V = wp.zeros(shape=(restart + 1, r.shape[0]), dtype=r.dtype, device=device)

    residuals = wp.empty(2, dtype=scalar_dtype, device=device)
    beta, atol_sq = residuals[0:1], residuals[1:2]

    tiled_dot = TiledDot(max_length=A.shape[0], device=device, scalar_type=scalar_dtype, max_column_count=restart + 1)
    r_norm_sq = tiled_dot.col(0)

    w_repeated = wp.array(
        ptr=w.ptr, shape=(restart + 1, w.shape[0]), strides=(0, w.strides[0]), dtype=w.dtype, device=w.device
    )

    # tile size for least square solve
    # (need to fit in a CUDA block, so 1024 max)
    if device.is_cuda and 4 < restart <= 1024:
        tile_size = 1 << math.ceil(math.log2(restart))
        least_squares_kernel = make_gmres_solve_least_squares_kernel_tiled(tile_size)
    else:
        tile_size = 1
        least_squares_kernel = _gmres_solve_least_squares

    # recorded launches
    least_squares_solve = wp.launch(
        least_squares_kernel,
        dim=(1, tile_size),
        block_dim=tile_size if tile_size > 1 else 256,
        device=device,
        inputs=[restart, pivot_tolerance, beta, H, y],
        record_cmd=True,
    )

    normalize_anorldi_vec = wp.launch(
        _gmres_arnoldi_normalize_kernel,
        dim=r.shape,
        device=r.device,
        inputs=[r, w, tiled_dot.col(0), beta],
        record_cmd=True,
    )

    arnoldi_axpy = wp.launch(
        _gmres_arnoldi_axpy_kernel,
        dim=(w.shape[0], tile_size),
        block_dim=tile_size,
        device=w.device,
        inputs=[V, w, H],
        record_cmd=True,
    )

    # Initialize tolerance from right-hand-side norm
    _initialize_absolute_tolerance(b, tol, atol, tiled_dot, atol_sq)
    # Initialize residual
    A.matvec(x, b, r, alpha=-1.0, beta=1.0)
    tiled_dot.compute(r, r, col_offset=0)

    # Not strictly necessary, but makes it more robust to user-provided LinearOperators
    w.zero_()

    def array_coeff(H, i, j):
        return H[i][j : j + 1]

    def array_col(H, j):
        return H[: j + 1, j : j + 1]

    def do_arnoldi_iteration(j: int):
        # w = A * v[j];
        if M is not None:
            tmp = V[j + 1]

            if is_left_preconditioner:
                A.matvec(V[j], tmp, tmp, alpha=1, beta=0)
                M.matvec(tmp, w, w, alpha=1, beta=0)
            else:
                M.matvec(V[j], tmp, tmp, alpha=1, beta=0)
                A.matvec(tmp, w, w, alpha=1, beta=0)
        else:
            A.matvec(V[j], w, w, alpha=1, beta=0)

        # compute and apply dot products in rappel,
        # since Hj columns are orthogonal
        Hj = array_col(H, j)
        tiled_dot.compute(w_repeated, V[: j + 1])
        wp.copy(src=tiled_dot.cols(j + 1), dest=Hj)

        # w -= w.vi vi
        arnoldi_axpy.set_params([V[: j + 1], w, Hj])
        arnoldi_axpy.launch()

        # H[j+1, j] = |w.w|
        tiled_dot.compute(w, w)
        normalize_anorldi_vec.set_params([w, V[j + 1], tiled_dot.col(0), array_coeff(H, j + 1, j)])

        normalize_anorldi_vec.launch()

    def do_restart_cycle():
        if M is not None and is_left_preconditioner:
            M.matvec(r, w, w, alpha=1, beta=0)
            rh = w
        else:
            rh = r

        # beta^2 = rh.rh
        tiled_dot.compute(rh, rh)

        # v[0] = r / beta
        normalize_anorldi_vec.set_params([rh, V[0], tiled_dot.col(0), beta])
        normalize_anorldi_vec.launch()

        for j in range(restart):
            do_arnoldi_iteration(j)

        least_squares_solve.launch()

        # update x
        if M is None or is_left_preconditioner:
            wp.launch(_gmres_update_x_kernel, dim=x.shape, device=device, inputs=[restart, scalar_dtype(1.0), y, V, x])
        else:
            wp.launch(_gmres_update_x_kernel, dim=x.shape, device=device, inputs=[restart, scalar_dtype(0.0), y, V, w])
            M.matvec(w, x, x, alpha=1, beta=1)

        # update r and residual
        wp.copy(src=b, dest=r)
        A.matvec(x, b, r, alpha=-1.0, beta=1.0)
        tiled_dot.compute(r, r)

    return _run_capturable_loop(
        do_restart_cycle,
        cycle_size=restart,
        r_norm_sq=r_norm_sq,
        maxiter=maxiter,
        atol_sq=atol_sq,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
    )


def _repeat_first(arr: wp.array):
    # returns a view of the first element repeated arr.shape[0] times
    view = wp.array(
        ptr=arr.ptr,
        shape=arr.shape,
        dtype=arr.dtype,
        strides=(0, *arr.strides[1:]),
        device=arr.device,
    )
    view._ref = arr
    return view


def _get_dtype_epsilon(dtype):
    if dtype == wp.float64:
        return 1.0e-16
    elif dtype == wp.float16:
        return 1.0e-4

    return 1.0e-8


def _get_tolerances(dtype, tol, atol):
    eps_tol = _get_dtype_epsilon(dtype)
    default_tol = eps_tol ** (3 / 4)
    min_tol = eps_tol ** (9 / 4)

    if tol is None and atol is None:
        tol = atol = default_tol
    elif tol is None:
        tol = atol
    elif atol is None:
        atol = tol

    atol = max(atol, min_tol)
    return tol, atol


@wp.kernel
def _initialize_tolerance(
    rtol: Any,
    atol: Any,
    r_norm_sq: wp.array(dtype=Any),
    atol_sq: wp.array(dtype=Any),
):
    atol = wp.max(rtol * wp.sqrt(r_norm_sq[0]), atol)
    atol_sq[0] = atol * atol


def _initialize_absolute_tolerance(
    b: wp.array,
    tol: float,
    atol: float,
    tiled_dot: TiledDot,
    atol_sq: wp.array,
):
    scalar_type = atol_sq.dtype

    # Compute b norm to define absolute tolerance
    tiled_dot.compute(b, b)
    b_norm_sq = tiled_dot.col(0)

    rtol, atol = _get_tolerances(scalar_type, tol, atol)
    wp.launch(
        kernel=_initialize_tolerance,
        dim=1,
        device=b.device,
        inputs=[scalar_type(rtol), scalar_type(atol), b_norm_sq, atol_sq],
    )


@wp.kernel
def _update_condition(
    maxiter: int,
    cycle_size: int,
    cur_iter: wp.array(dtype=int),
    r_norm_sq: wp.array(dtype=Any),
    atol_sq: wp.array(dtype=Any),
    condition: wp.array(dtype=int),
):
    cur_iter[0] += cycle_size
    condition[0] = wp.where(r_norm_sq[0] <= atol_sq[0] or cur_iter[0] >= maxiter, 0, 1)


def _run_capturable_loop(
    do_cycle: Callable,
    r_norm_sq: wp.array,
    maxiter: int,
    atol_sq: wp.array,
    callback: Optional[Callable],
    check_every: int,
    use_cuda_graph: bool,
    cycle_size: int = 1,
):
    device = atol_sq.device

    if check_every > 0:
        atol = math.sqrt(atol_sq.numpy()[0])
        return _run_solver_loop(
            do_cycle, cycle_size, r_norm_sq, maxiter, atol, callback, check_every, use_cuda_graph, device
        )

    cur_iter_and_condition = wp.full((2,), value=-1, dtype=int, device=device)
    cur_iter = cur_iter_and_condition[0:1]
    condition = cur_iter_and_condition[1:2]

    update_condition_launch = wp.launch(
        _update_condition,
        dim=1,
        device=device,
        inputs=[int(maxiter), cycle_size, cur_iter, r_norm_sq, atol_sq, condition],
        record_cmd=True,
    )

    if isinstance(callback, wp.Kernel):
        callback_launch = wp.launch(
            callback, dim=1, device=device, inputs=[cur_iter, r_norm_sq, atol_sq], record_cmd=True
        )
    else:
        callback_launch = None

    update_condition_launch.launch()
    if callback_launch is not None:
        callback_launch.launch()

    def do_cycle_with_condition():
        do_cycle()
        update_condition_launch.launch()
        if callback_launch is not None:
            callback_launch.launch()

    if use_cuda_graph and device.is_cuda:
        if device.is_capturing:
            wp.capture_while(condition, do_cycle_with_condition)
        else:
            with wp.ScopedCapture() as capture:
                wp.capture_while(condition, do_cycle_with_condition)
            wp.capture_launch(capture.graph)
    else:
        for _ in range(0, maxiter, cycle_size):
            do_cycle_with_condition()

    return cur_iter, r_norm_sq, atol_sq


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
    check_every = max(check_every, cycle_size)

    cur_iter = 0

    err_sq = r_norm_sq.numpy()[0]
    err = math.sqrt(err_sq)
    if callback is not None:
        callback(cur_iter, err, atol)

    if err_sq <= atol_sq:
        return cur_iter, err, atol

    graph = None

    while True:
        # Do not do graph capture at first iteration -- modules may not be loaded yet
        if device.is_cuda and use_cuda_graph and cur_iter > 0:
            if graph is None:
                with wp.ScopedCapture(force_module_load=False) as capture:
                    do_cycle()
                graph = capture.graph
            wp.capture_launch(graph)
        else:
            do_cycle()

        cur_iter += cycle_size

        if cur_iter >= maxiter:
            break

        if (cur_iter % check_every) < cycle_size:
            err_sq = r_norm_sq.numpy()[0]

            if err_sq <= atol_sq:
                break

            if callback is not None:
                callback(cur_iter, math.sqrt(err_sq), atol)

    err_sq = r_norm_sq.numpy()[0]
    err = math.sqrt(err_sq)
    if callback is not None:
        callback(cur_iter, err, atol)

    return cur_iter, err, atol


@wp.kernel
def _dense_mv_kernel(
    A: wp.array2d(dtype=Any),
    x: wp.array1d(dtype=Any),
    y: wp.array1d(dtype=Any),
    z: wp.array1d(dtype=Any),
    alpha: Any,
    beta: Any,
):
    row, lane = wp.tid()

    zero = type(alpha)(0)
    s = zero
    if alpha != zero:
        for col in range(lane, A.shape[1], wp.block_dim()):
            s += A[row, col] * x[col]

    row_tile = wp.tile_sum(wp.tile(s * alpha))

    if beta != zero:
        row_tile += wp.tile_load(y, shape=1, offset=row) * beta

    wp.tile_store(z, row_tile, offset=row)


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
    zero = type(alpha)(0)
    s = z.dtype(zero)
    if alpha != zero:
        s += alpha * (A[i] * x[i])
    if beta != zero:
        s += beta * y[i]
    z[i] = s


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
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    rz_old: wp.array(dtype=Any),
    p_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
):
    i = wp.tid()

    alpha = wp.where(resid[0] > tol[0], rz_old[0] / p_Ap[0], rz_old.dtype(0.0))

    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]


@wp.kernel
def _cg_kernel_2(
    tol: wp.array(dtype=Any),
    resid_new: wp.array(dtype=Any),
    rz_old: wp.array(dtype=Any),
    rz_new: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
):
    #    p = r + (rz_new / rz_old) * p;
    i = wp.tid()

    cond = resid_new[0] > tol[0]
    beta = wp.where(cond, rz_new[0] / rz_old[0], rz_old.dtype(0.0))

    p[i] = z[i] + beta * p[i]


@wp.kernel
def _cr_kernel_1(
    tol: wp.array(dtype=Any),
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

    alpha = wp.where(resid[0] > tol[0] and y_Ap[0] > 0.0, zAz_old[0] / y_Ap[0], zAz_old.dtype(0.0))

    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]
    z[i] = z[i] - alpha * y[i]


@wp.kernel
def _cr_kernel_2(
    tol: wp.array(dtype=Any),
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

    beta = wp.where(resid[0] > tol[0] and zAz_old[0] > 0.0, zAz_new[0] / zAz_old[0], zAz_old.dtype(0.0))

    p[i] = z[i] + beta * p[i]
    Ap[i] = Az[i] + beta * Ap[i]


@wp.kernel
def _bicgstab_kernel_1(
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    rho_old: wp.array(dtype=Any),
    r0v: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    v: wp.array(dtype=Any),
):
    i = wp.tid()

    alpha = wp.where(resid[0] > tol[0], rho_old[0] / r0v[0], rho_old.dtype(0.0))

    x[i] += alpha * y[i]
    r[i] -= alpha * v[i]


@wp.kernel
def _bicgstab_kernel_2(
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    st: wp.array(dtype=Any),
    tt: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    t: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
):
    i = wp.tid()

    omega = wp.where(resid[0] > tol[0], st[0] / tt[0], st.dtype(0.0))

    x[i] += omega * z[i]
    r[i] -= omega * t[i]


@wp.kernel
def _bicgstab_kernel_3(
    tol: wp.array(dtype=Any),
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

    beta = wp.where(resid[0] > tol[0], rho_new[0] * tt[0] / (r0v[0] * st[0]), st.dtype(0.0))
    beta_omega = wp.where(resid[0] > tol[0], rho_new[0] / r0v[0], st.dtype(0.0))

    p[i] = r[i] + beta * p[i] - beta_omega * v[i]


@wp.kernel
def _gmres_solve_least_squares(
    k: int, pivot_tolerance: float, beta: wp.array(dtype=Any), H: wp.array2d(dtype=Any), y: wp.array(dtype=Any)
):
    # Solve H y = (beta, 0, ..., 0)
    # H Hessenberg matrix of shape (k+1, k)
    # so would not fit in registers

    rhs = beta[0]

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


@functools.lru_cache(maxsize=None)
def make_gmres_solve_least_squares_kernel_tiled(K: int):
    @wp.kernel(module="unique")
    def gmres_solve_least_squares_tiled(
        k: int, pivot_tolerance: float, beta: wp.array(dtype=Any), H: wp.array2d(dtype=Any), y: wp.array(dtype=Any)
    ):
        # Assumes tiles of size K, and K at least as large as highest number of columns
        # Limits the max restart cycle length to the max block size of 1024, but using
        # larger restarts would be very inefficient anyway (default is ~30)

        # Solve H y = (beta, 0, ..., 0)
        # H Hessenberg matrix of shape (k+1, k)

        i, lane = wp.tid()

        rhs = beta[0]

        zero = H.dtype(0.0)
        one = H.dtype(1.0)
        yi = zero

        Ha = wp.tile_load(H[0], shape=(K))

        # Apply 2x2 rotations to H so as to remove lower diagonal,
        # and apply similar rotations to right-hand-side
        max_k = int(k)
        for i in range(k):
            # Ha = H[i]
            # Hb = H[i + 1]
            Hb = wp.tile_load(H[i + 1], shape=(K))

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
            a = wp.untile(Ha)
            b = wp.untile(Hb)
            a_rot = c * a + s * b
            b_rot = c * b - s * a

            # Rotate rhs
            if lane == i:
                yi = c * rhs
            rhs = -s * rhs

            wp.tile_store(H[i], wp.tile(a_rot))
            Ha[lane] = b_rot

        y_tile = wp.tile(yi)

        # Triangular back-solve for y
        for ii in range(max_k, 0, -1):
            i = ii - 1

            Hi = wp.tile_load(H[i], shape=(K))

            il = lane + i
            if lane == 0:
                yl = y_tile[i]
            elif il < max_k:
                yl = -y_tile[il] * Hi[il]
            else:
                yl = zero

            yit = wp.tile_sum(wp.tile(yl)) * (one / Hi[i])
            yit[0]  # no-op, movs yit to shared
            wp.tile_assign(y_tile, yit, offset=(i,))

        wp.tile_store(y, y_tile)

    return gmres_solve_least_squares_tiled


@wp.kernel
def _gmres_arnoldi_axpy_kernel(
    V: wp.array2d(dtype=Any),
    w: wp.array(dtype=Any),
    Vw: wp.array2d(dtype=Any),
):
    tid, lane = wp.tid()

    s = w.dtype(Vw.dtype(0))

    tile_size = wp.block_dim()
    for k in range(lane, Vw.shape[0], tile_size):
        s += Vw[k, 0] * V[k, tid]

    wi = wp.tile_load(w, shape=1, offset=tid)
    wi -= wp.tile_sum(wp.tile(s, preserve_type=True))

    wp.tile_store(w, wi, offset=tid)


@wp.kernel
def _gmres_arnoldi_normalize_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    alpha_copy: wp.array(dtype=Any),
):
    tid = wp.tid()
    norm = wp.sqrt(alpha[0])
    y[tid] = wp.where(alpha[0] == alpha.dtype(0.0), x[tid], x[tid] / norm)

    if tid == 0:
        alpha_copy[0] = norm


@wp.kernel
def _gmres_update_x_kernel(k: int, beta: Any, y: wp.array(dtype=Any), V: wp.array2d(dtype=Any), x: wp.array(dtype=Any)):
    tid = wp.tid()

    xi = beta * x[tid]
    for j in range(k):
        xi += V[j, tid] * y[j]

    x[tid] = xi
