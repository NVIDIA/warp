# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
from collections.abc import Callable
from typing import Any

import warp as wp
import warp.sparse as sparse
from warp._src.types import type_is_matrix, type_is_vector, type_length, type_scalar_type

_wp_module_name_ = "warp.optim.linear"

__all__ = [
    "CG",
    "CR",
    "GMRES",
    "BiCGSTAB",
    "LinearOperator",
    "LinearSolverState",
    "aslinearoperator",
    "bicgstab",
    "cg",
    "cr",
    "gmres",
    "preconditioner",
]

# No need to auto-generate adjoint code for linear solvers
wp.set_module_options({"enable_backward": False})


class LinearOperator:
    """Linear operator to be used as left-hand-side of linear iterative solvers.

    Args:
        shape: Tuple containing the number of rows and columns of the operator
        dtype: Type of the operator elements
        device: Device on which computations involving the operator should be performed
        matvec: Matrix-vector multiplication routine
        batch_offsets: Optional array of shape ``(B+1,)`` partitioning scalar degrees of freedom into
            ``B`` independent subproblems. ``batch_offsets[i]`` is the first scalar degree of freedom of
            subproblem ``i``. For vector-valued arrays, offsets must be aligned to the vector length.
            When ``None`` (default) the operator represents a single subproblem.

    The matrix-vector multiplication routine should have the following signature:

    .. code-block:: python

        def matvec(x: wp.array, y: wp.array, z: wp.array, alpha: Scalar, beta: Scalar):
            '''Perform a generalized matrix-vector product.

            This function computes the operation z = alpha * (A @ x) + beta * y, where 'A'
            is the linear operator represented by this class.
            '''
            ...

    For performance reasons, by default the iterative linear solvers in this module will try to capture the calls
    for one or more iterations in CUDA graphs. If the ``matvec`` routine of a custom :class:`LinearOperator`
    cannot be graph-captured, the ``use_cuda_graph=False`` parameter should be passed to the solver function.

    """

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: type,
        device: wp._src.context.Device,
        matvec: Callable,
        batch_offsets: wp.array | None = None,
    ):
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._matvec = matvec
        self._batch_offsets = batch_offsets

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def device(self) -> wp.Device:
        return self._device

    @property
    def matvec(self) -> Callable:
        return self._matvec

    @property
    def scalar_type(self):
        return type_scalar_type(self.dtype)

    @property
    def batch_offsets(self) -> wp.array | None:
        """Array of length ``batch_count + 1`` partitioning scalar degrees of freedom, or ``None``."""
        return self._batch_offsets

    @property
    def batch_count(self) -> int:
        """Number of independent subproblems. ``1`` when :attr:`batch_offsets` is ``None``."""
        return 1 if self._batch_offsets is None else self._batch_offsets.shape[0] - 1


_Matrix = wp.array | sparse.BsrMatrix | LinearOperator


def aslinearoperator(A: _Matrix, batch_offsets: wp.array | None = None) -> LinearOperator:
    """Cast the dense or sparse matrix ``A`` as a :class:`LinearOperator`.

    ``A`` must be of one of the following types:

        - :class:`warp.sparse.BsrMatrix`
        - two-dimensional ``warp.array``; then ``A`` is assumed to be a dense matrix
        - one-dimensional ``warp.array``; then ``A`` is assumed to be a diagonal matrix
        - :class:`warp.optim.linear.LinearOperator`; no casting necessary, ``batch_offsets`` is ignored

    Args:
        A: The matrix to wrap.
        batch_offsets: Optional array of shape ``(B+1,)`` partitioning scalar degrees of freedom into
            ``B`` independent subproblems (see :class:`LinearOperator`).
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
            return LinearOperator(A.shape, A.dtype, A.device, matvec=dense_mv, batch_offsets=batch_offsets)
        if A.ndim == 1:
            if type_is_vector(A.dtype):
                return LinearOperator(A.shape, A.dtype, A.device, matvec=diag_mv_vec, batch_offsets=batch_offsets)
            return LinearOperator(A.shape, A.dtype, A.device, matvec=diag_mv, batch_offsets=batch_offsets)
    if isinstance(A, sparse.BsrMatrix):
        return LinearOperator(A.shape, A.dtype, A.device, matvec=bsr_mv, batch_offsets=batch_offsets)

    raise ValueError(f"Unable to create LinearOperator from {A}")


def preconditioner(A: _Matrix, ptype: str = "diag") -> LinearOperator:
    """Construct and return a preconditioner for an input matrix.

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
        return _make_jacobi_preconditioner(A, use_abs=ptype == "diag_abs")

    raise ValueError(f"Unsupported preconditioner type '{ptype}'")


def _make_jacobi_preconditioner(A: _Matrix, use_abs: bool) -> LinearOperator:
    use_abs_int = 1 if use_abs else 0
    if isinstance(A, sparse.BsrMatrix):
        A_diag = sparse.bsr_get_diag(A)
        if type_is_matrix(A.dtype):
            inv_diag = wp.empty(
                shape=A.nrow, dtype=wp.types.vector(length=A.block_shape[0], dtype=A.scalar_type), device=A.device
            )
            kernel = _extract_inverse_diagonal_blocked
        else:
            inv_diag = wp.empty(shape=A.shape[0], dtype=A.scalar_type, device=A.device)
            kernel = _extract_inverse_diagonal_scalar
        wp.launch(kernel, dim=inv_diag.shape, device=inv_diag.device, inputs=[A_diag, inv_diag, use_abs_int])
    elif isinstance(A, wp.array) and A.ndim == 2:
        inv_diag = wp.empty(shape=A.shape[0], dtype=A.dtype, device=A.device)
        wp.launch(
            _extract_inverse_diagonal_dense,
            dim=inv_diag.shape,
            device=inv_diag.device,
            inputs=[A, inv_diag, use_abs_int],
        )
    else:
        raise ValueError("Unsupported source matrix type for building diagonal preconditioner")

    return aslinearoperator(inv_diag)


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


def _dofs_per_entry(dtype: type):
    return type_length(dtype) if type_scalar_type(dtype) != dtype else 1


def _scalar_dof_count(x: wp.array):
    return x.shape[0] * _dofs_per_entry(x.dtype)


class TiledDot:
    """Compute the dot product of two arrays in a way that is compatible with CUDA sub-graphs.

    Args:
        max_length: Total length of the arrays to dot (sum of all subproblem lengths).
        scalar_type: Scalar data type of the arrays.
        tile_size: Number of threads per tile/block.
        device: Device on which to allocate scratch memory and launch kernels.
        max_column_count: Maximum number of simultaneous dot products.
        batch_offsets: Optional array of shape ``(B+1,)`` partitioning scalar degrees of freedom into
            ``B`` independent subproblems. When provided, :meth:`compute` returns ``B`` independent
            dot products rather than a single global one.
    """

    def __init__(
        self,
        max_length: int,
        scalar_type: type,
        tile_size=512,
        device=None,
        max_column_count: int = 1,
        batch_offsets: wp.array | None = None,
    ):
        self.tile_size = tile_size
        self.device = device
        self.max_column_count = max_column_count
        self.batch_offsets = batch_offsets
        self.batch_count = 1 if batch_offsets is None else batch_offsets.shape[0] - 1

        if self.batch_count == 1:
            num_blocks = (max_length + self.tile_size - 1) // self.tile_size
            # Scratch must hold at least batch_count result slots per column (one per subproblem)
            scratch_size = max(num_blocks, self.batch_count)
            scratch = wp.zeros(
                shape=(2, max_column_count, scratch_size),
                dtype=scalar_type,
                device=self.device,
            )
            self.partial_sums_a = scratch[0]
            self.partial_sums_b = scratch[1]

            # Non-batched: tiled tree reduction
            self.dot_kernel, self.sum_kernel = _create_tiled_dot_kernels(self.tile_size)

            rounds = 0
            length = (max_length + self.tile_size - 1) // self.tile_size
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
            self.batch_dot_launch = None
        else:
            # Batched: direct per-subproblem reduction in one kernel
            # Each subproblem's threads loop over their DOF range and reduce cooperatively.
            # (Assumes all subproblems are small compared to total length)

            if not self.device.is_cuda:
                self.tile_size = 1

            self.partial_sums_a = wp.zeros(
                shape=(max_column_count, self.batch_count),
                dtype=scalar_type,
                device=self.device,
            )
            self.partial_sums_b = self.partial_sums_a

            self.rounds = 0
            self._output = self.partial_sums_a  # rounds=0 -> always written to partial_sums_a
            self.dot_launch = None
            self.sum_launch = None

            self.batch_dot_launch: wp.Launch = wp.launch(
                batch_dot_kernel,
                dim=(max_column_count, self.batch_count, self.tile_size),
                inputs=[self.partial_sums_a, self.partial_sums_b, self.partial_sums_a, batch_offsets],
                block_dim=self.tile_size,
                device=self.device,
                record_cmd=True,
            )

    def compute(self, a: wp.array, b: wp.array, col_offset: int = 0):
        """Compute dot products, updating results accessible via :meth:`col` and :meth:`cols`.

        Args:
            a: First array operand (1-D or 2-D with leading column dimension).
            b: Second array operand, same shape as ``a``.
            col_offset: Column index in the scratch at which to write results.
        """
        a = _as_scalar_array(a)
        b = _as_scalar_array(b)
        if a.ndim == 1:
            a = a.reshape((1, -1))
        if b.ndim == 1:
            b = b.reshape((1, -1))

        column_count = a.shape[0]
        data_out = self.partial_sums_a[col_offset : col_offset + column_count]
        data_in = self.partial_sums_b[col_offset : col_offset + column_count]

        if self.batch_dot_launch is not None:
            # Batched path: one block per (col, batch_id) with tile_size lanes
            self.batch_dot_launch.set_param_at_index(0, a)
            self.batch_dot_launch.set_param_at_index(1, b)
            self.batch_dot_launch.set_param_at_index(2, data_out)
            self.batch_dot_launch.set_dim((column_count, self.batch_count, self.tile_size))
            self.batch_dot_launch.launch()
        else:
            # Non-batched path: tiled tree reduction
            num_blocks = (a.shape[1] + self.tile_size - 1) // self.tile_size

            self.dot_launch.set_param_at_index(0, a)
            self.dot_launch.set_param_at_index(1, b)
            self.dot_launch.set_param_at_index(2, data_out)
            self.dot_launch.set_dim((column_count, num_blocks, self.tile_size))
            self.dot_launch.launch()

            for _r in range(self.rounds):
                array_length = num_blocks
                num_blocks = (array_length + self.tile_size - 1) // self.tile_size
                data_in, data_out = data_out, data_in

                self.sum_launch.set_param_at_index(0, data_in[:, :array_length])
                self.sum_launch.set_param_at_index(1, data_out)
                self.sum_launch.set_dim((column_count, num_blocks, self.tile_size))
                self.sum_launch.launch()

        return data_out

    def col(self, col: int = 0) -> wp.array:
        """Return a view of the result for column ``col``, shape ``(batch_count,)``."""
        return self._output[col][: self.batch_count]

    def cols(self, count: int, start: int = 0) -> wp.array:
        """Return a view of results for columns ``[start, start+count)``, shape ``(count, batch_count)``."""
        return self._output[start : start + count, : self.batch_count]


@functools.cache
def _create_tiled_dot_kernels(tile_size):
    @wp.kernel(module="unique")
    def block_dot_kernel(
        a: wp.array2d(dtype=Any),
        b: wp.array2d(dtype=Any),
        partial_sums: wp.array2d(dtype=Any),
    ):
        column, block_id, _tid_block = wp.tid()

        start = block_id * tile_size

        a_block = wp.tile_load(a[column], shape=tile_size, offset=start)
        b_block = wp.tile_load(b[column], shape=tile_size, offset=start)
        t = wp.tile_map(wp.mul, a_block, b_block)

        tile_sum = wp.tile_sum(t)
        wp.tile_store(partial_sums[column], tile_sum, offset=block_id)

    @wp.kernel(module="unique")
    def block_sum_kernel(
        data: wp.array2d(dtype=Any),
        partial_sums: wp.array2d(dtype=Any),
    ):
        column, block_id, _tid_block = wp.tid()
        start = block_id * tile_size

        t = wp.tile_load(data[column], shape=tile_size, offset=start)

        tile_sum = wp.tile_sum(t)
        wp.tile_store(partial_sums[column], tile_sum, offset=block_id)

    return block_dot_kernel, block_sum_kernel


@wp.kernel(module="unique")
def batch_dot_kernel(
    a: wp.array2d(dtype=Any),
    b: wp.array2d(dtype=Any),
    result: wp.array2d(dtype=Any),
    batch_offsets: wp.array1d(dtype=int),
):
    col, batch_id, lane = wp.tid()

    batch_start = batch_offsets[batch_id] + lane
    batch_end = batch_offsets[batch_id + 1]

    # Each lane strides over the subproblem range and accumulates its share
    acc = a.dtype(0.0)
    for i in range(batch_start, batch_end, wp.block_dim()):
        acc += a[col, i] * b[col, i]

    # Cooperative reduction across all lanes in this block
    total = wp.tile_sum(wp.tile(acc))
    wp.tile_store(result[col], total, offset=batch_id)


class LinearSolverState:
    """Pre-allocated state for a linear iterative solver.

    Holds all temporary buffers required by the solver plus a reference to the original
    system (``A``, ``b``, ``x``, optional ``M``). Calling the state runs the solver,
    optionally substituting a new compatible matrix, right-hand-side, solution vector,
    or preconditioner. This avoids repeated buffer allocation when the same solver is
    applied many times to systems that share the same shape, batch count, dtype, and
    device.

    Args:
        A: the linear system's left-hand-side
        b: the linear system's right-hand-side
        x: initial guess and solution vector
        tol: relative tolerance for the residual, as a ratio of the right-hand-side norm
        atol: absolute tolerance for the residual
        maxiter: maximum number of iterations to perform before aborting. Defaults to the system size.
        M: optional preconditioner.
        callback: function to be called every ``check_every`` iteration with the current iteration number, residual and tolerance.
        check_every: number of iterations every which to call ``callback`` and check the residual against the tolerance.
        use_cuda_graph: whether to capture the solver iteration as a CUDA graph for reduced launch overhead.

    Instances are not directly constructible — use the solver-specific subclasses
    :class:`~warp.optim.linear.CG`, :class:`~warp.optim.linear.CR`, :class:`~warp.optim.linear.BiCGSTAB`, and :class:`~warp.optim.linear.GMRES`, or obtain one by
    calling the corresponding solver function (:func:`cg`, :func:`cr`, :func:`bicgstab`,
    :func:`gmres`) with ``run=False``.
    """

    def __init__(
        self,
        A: _Matrix,
        b: wp.array,
        x: wp.array,
        tol: float | None = None,
        atol: float | None = None,
        maxiter: float | None = 0,
        M: _Matrix | None = None,
        callback: Callable | None = None,
        check_every: int = 10,
        use_cuda_graph: bool = True,
    ):
        self._A = aslinearoperator(A)
        self._M = aslinearoperator(M)
        self._b = b
        self._x = x
        self._tol = tol
        self._atol = atol
        self._callback = callback
        self._check_every = check_every
        self._use_cuda_graph = use_cuda_graph

        self._device = self._A.device
        self._scalar_type = self._A.scalar_type
        self._batch_count = self._A.batch_count
        self._dofs_per_entry = _dofs_per_entry(b.dtype)

        if maxiter is None or maxiter == 0:
            maxiter = _scalar_dof_count(b) // self._batch_count
        self._maxiter = int(maxiter)

        self._allocate()

    def _allocate(self):
        """Allocate solver-specific temporary buffers. Implemented by subclasses."""
        raise NotImplementedError

    def _run(self, A: "LinearOperator", b: wp.array, x: wp.array, M: "LinearOperator | None"):
        """Run one solve with the given (possibly substituted) operands. Implemented by subclasses."""
        raise NotImplementedError

    def _check_compatible(self, A: "LinearOperator", b: wp.array, x: wp.array, M: "LinearOperator | None"):
        """Validate that ``A``, ``b``, ``x``, ``M`` are compatible with the allocated state."""
        if A is not self._A:
            if A.shape != self._A.shape:
                raise ValueError(f"Incompatible A.shape: expected {self._A.shape}, got {A.shape}")
            if A.dtype != self._A.dtype:
                raise ValueError(f"Incompatible A.dtype: expected {self._A.dtype}, got {A.dtype}")
            if A.device != self._A.device:
                raise ValueError(f"Incompatible A.device: expected {self._A.device}, got {A.device}")
            if A.batch_count != self._A.batch_count:
                raise ValueError(f"Incompatible A.batch_count: expected {self._A.batch_count}, got {A.batch_count}")
            if self._A.batch_offsets is not None and A.batch_offsets is not self._A.batch_offsets:
                raise ValueError("For batched systems, A.batch_offsets must be the same array as at construction")
        if b is not self._b:
            if b.shape != self._b.shape:
                raise ValueError(f"Incompatible b.shape: expected {self._b.shape}, got {b.shape}")
            if b.dtype != self._b.dtype:
                raise ValueError(f"Incompatible b.dtype: expected {self._b.dtype}, got {b.dtype}")
        if x is not self._x:
            if x.shape != self._x.shape:
                raise ValueError(f"Incompatible x.shape: expected {self._x.shape}, got {x.shape}")
            if x.dtype != self._x.dtype:
                raise ValueError(f"Incompatible x.dtype: expected {self._x.dtype}, got {x.dtype}")
        if M is not None and self._M is not None and M is not self._M:
            if M.shape != self._M.shape:
                raise ValueError(f"Incompatible M.shape: expected {self._M.shape}, got {M.shape}")
            if M.dtype != self._M.dtype:
                raise ValueError(f"Incompatible M.dtype: expected {self._M.dtype}, got {M.dtype}")

    def __call__(
        self,
        A: _Matrix | None = None,
        b: wp.array | None = None,
        x: wp.array | None = None,
        M: _Matrix | None = None,
    ):
        """Run the solver, optionally substituting a new compatible matrix, right-hand-side,
        solution vector, or preconditioner. Any argument left as ``None`` uses the value that
        was passed at construction."""
        A_op = aslinearoperator(A) if A is not None else self._A
        M_op = aslinearoperator(M) if M is not None else self._M
        b_arr = b if b is not None else self._b
        x_arr = x if x is not None else self._x

        self._check_compatible(A_op, b_arr, x_arr, M_op)
        return self._run(A_op, b_arr, x_arr, M_op)


class CG(LinearSolverState):
    """Pre-allocated state for the Conjugate Gradient solver.

    See :class:`LinearSolverState` for the constructor parameters. The preconditioner
    ``M`` may be freely changed (or toggled between ``None`` and a valid operator)
    between calls as long as the matrix shape, batch count, dtype, and device remain
    the same.
    """

    def _allocate(self):
        A = self._A
        b = self._b
        device = self._device
        scalar_type = self._scalar_type
        batch_count = self._batch_count

        # Temp storage — residuals are per-subproblem
        self._r_and_z_buf = wp.empty((2, b.shape[0]), dtype=b.dtype, device=device)
        self._p_and_Ap = wp.empty_like(self._r_and_z_buf)
        self._residuals = wp.empty((2, batch_count), dtype=scalar_type, device=device)

        self._tiled_dot = TiledDot(
            max_length=_scalar_dof_count(b),
            device=device,
            scalar_type=scalar_type,
            max_column_count=2,
            batch_offsets=A.batch_offsets,
        )

        # (r, r) view — so we can compute r.z and r.r at once
        self._r_repeated = _repeat_first(self._r_and_z_buf)

    def _run(self, A, b, x, M):
        device = self._device
        batch_offsets = A.batch_offsets
        dofs_per_entry = self._dofs_per_entry
        tiled_dot = self._tiled_dot
        p_and_Ap = self._p_and_Ap
        r_and_z_buf = self._r_and_z_buf
        r_repeated = self._r_repeated

        if M is None:
            # without preconditioner r == z
            r_and_z = r_repeated
            rz_new = tiled_dot.col(0)
        else:
            r_and_z = r_and_z_buf
            rz_new = tiled_dot.col(1)

        r, z = r_and_z[0], r_and_z[1]
        r_norm_sq = tiled_dot.col(0)

        p, Ap = p_and_Ap[0], p_and_Ap[1]
        rz_old, atol_sq = self._residuals[0], self._residuals[1]

        # Not strictly necessary, but makes it more robust to user-provided LinearOperators
        Ap.zero_()
        z.zero_()

        # Initialize tolerance from right-hand-side norm
        _initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
        # Initialize residual
        A.matvec(x, b, r, alpha=-1.0, beta=1.0)

        def update_rr_rz():
            # z = M r
            if M is None:
                tiled_dot.compute(r, r)
            else:
                M.matvec(r, z, z, alpha=1.0, beta=0.0)
                tiled_dot.compute(r_repeated, r_and_z_buf)

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
                inputs=[atol_sq, r_norm_sq, rz_old, p_Ap, x, r, p, Ap, batch_offsets, dofs_per_entry],
            )

            update_rr_rz()

            wp.launch(
                kernel=_cg_kernel_2,
                dim=z.shape[0],
                device=device,
                inputs=[atol_sq, r_norm_sq, rz_old, rz_new, z, p, batch_offsets, dofs_per_entry],
            )

        return _run_capturable_loop(
            do_iteration,
            r_norm_sq,
            self._maxiter,
            atol_sq,
            self._callback,
            self._check_every,
            self._use_cuda_graph,
        )


def cg(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: float | None = None,
    atol: float | None = None,
    maxiter: float | None = 0,
    M: _Matrix | None = None,
    callback: Callable | None = None,
    check_every=10,
    use_cuda_graph=True,
    run: bool = True,
) -> tuple[int, float, float] | tuple[wp.array, wp.array, wp.array] | CG:
    """Compute an approximate solution to a symmetric, positive-definite linear system
    using the Conjugate Gradient algorithm.

    Supports batched systems when ``A`` is a :class:`LinearOperator` with ``batch_offsets`` set;
    all subproblems iterate together and convergence uses the worst-case residual.

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
        run: If ``True`` (default), allocate temporary buffers and immediately run the solver, returning the iteration count,
            residual norm, and absolute tolerance. If ``False``, return a pre-allocated :class:`~warp.optim.linear.CG` functor that can be
            called repeatedly on compatible systems (same shape, batch count, dtype, device) without re-allocating.

    Returns:
        If ``run`` is ``True`` and ``check_every`` > 0: Tuple (final_iteration, residual_norm, absolute_tolerance)
            - final_iteration: The number of iterations performed before convergence or reaching maxiter
            - residual_norm: The final residual norm ||b - Ax||
            - absolute_tolerance: The absolute tolerance used for convergence checking

        If ``run`` is ``True`` and ``check_every`` is 0: Tuple (final_iteration_array, residual_norm_squared_array, absolute_tolerance_squared_array)
            - final_iteration_array: Device array containing the number of iterations performed
            - residual_norm_squared_array: Device array containing the squared residual norm ||b - Ax||²
            - absolute_tolerance_squared_array: Device array containing the squared absolute tolerance

        If ``run`` is ``False``: a :class:`~warp.optim.linear.CG` functor with all temporary buffers pre-allocated.

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """
    state = CG(
        A,
        b,
        x,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=M,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
    )
    if run:
        return state()
    return state


class CR(LinearSolverState):
    """Pre-allocated state for the Conjugate Residual solver.

    See :class:`LinearSolverState` for the constructor parameters. The preconditioner
    ``M`` may be freely changed (or toggled between ``None`` and a valid operator)
    between calls as long as the matrix shape, batch count, dtype, and device remain
    the same.
    """

    def _allocate(self):
        A = self._A
        b = self._b
        device = self._device
        scalar_type = self._scalar_type
        batch_count = self._batch_count

        # Notations follow pseudo-code from https://en.wikipedia.org/wiki/Conjugate_residual_method
        # with z := M^-1 r and y := M^-1 Ap
        self._r_and_z_buf = wp.empty((2, b.shape[0]), dtype=b.dtype, device=device)
        self._r_and_Az = wp.empty_like(self._r_and_z_buf)
        self._y_and_Ap_buf = wp.empty_like(self._r_and_z_buf)
        self._p = wp.empty_like(b)
        self._residuals = wp.empty((2, batch_count), dtype=scalar_type, device=device)

        self._tiled_dot = TiledDot(
            max_length=_scalar_dof_count(b),
            device=device,
            scalar_type=scalar_type,
            max_column_count=2,
            batch_offsets=A.batch_offsets,
        )

        self._r_and_z_repeated = _repeat_first(self._r_and_z_buf)
        self._y_and_Ap_repeated = _repeat_first(self._y_and_Ap_buf)

    def _run(self, A, b, x, M):
        device = self._device
        batch_offsets = A.batch_offsets
        dofs_per_entry = self._dofs_per_entry
        tiled_dot = self._tiled_dot
        r_and_z_buf = self._r_and_z_buf
        y_and_Ap_buf = self._y_and_Ap_buf
        r_and_Az = self._r_and_Az
        p = self._p

        if M is None:
            r_and_z = self._r_and_z_repeated
            y_and_Ap = self._y_and_Ap_repeated
        else:
            r_and_z = r_and_z_buf
            y_and_Ap = y_and_Ap_buf

        r, z = r_and_z[0], r_and_z[1]
        r_copy, Az = r_and_Az[0], r_and_Az[1]
        y, Ap = y_and_Ap[0], y_and_Ap[1]

        r_norm_sq = tiled_dot.col(0)
        zAz_new = tiled_dot.col(1)
        zAz_old, atol_sq = self._residuals[0], self._residuals[1]

        # Initialize tolerance from right-hand-side norm
        _initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
        # Initialize residual
        A.matvec(x, b, r, alpha=-1.0, beta=1.0)

        # Not strictly necessary, but makes it more robust to user-provided LinearOperators
        y_and_Ap_buf.zero_()

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
                    inputs=[atol_sq, r_norm_sq, zAz_old, y_Ap, x, r, p, Ap, batch_offsets, dofs_per_entry],
                )
            else:
                # In preconditioned case, we have one more vector to update
                wp.launch(
                    kernel=_cr_kernel_1,
                    dim=x.shape[0],
                    device=device,
                    inputs=[atol_sq, r_norm_sq, zAz_old, y_Ap, x, r, z, p, Ap, y, batch_offsets, dofs_per_entry],
                )

            update_rr_zAz()
            wp.launch(
                kernel=_cr_kernel_2,
                dim=z.shape[0],
                device=device,
                inputs=[atol_sq, r_norm_sq, zAz_old, zAz_new, z, p, Az, Ap, batch_offsets, dofs_per_entry],
            )

        return _run_capturable_loop(
            do_iteration,
            cycle_size=1,
            r_norm_sq=r_norm_sq,
            maxiter=self._maxiter,
            atol_sq=atol_sq,
            callback=self._callback,
            check_every=self._check_every,
            use_cuda_graph=self._use_cuda_graph,
        )


def cr(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: float | None = None,
    atol: float | None = None,
    maxiter: float | None = 0,
    M: _Matrix | None = None,
    callback: Callable | None = None,
    check_every=10,
    use_cuda_graph=True,
    run: bool = True,
) -> tuple[int, float, float] | tuple[wp.array, wp.array, wp.array] | CR:
    """Compute an approximate solution to a symmetric, positive-definite linear system
    using the Conjugate Residual algorithm.

    Supports batched systems when ``A`` is a :class:`LinearOperator` with ``batch_offsets`` set;
    all subproblems iterate together and convergence uses the worst-case residual.

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
        run: If ``True`` (default), allocate temporary buffers and immediately run the solver. If ``False``, return a
            pre-allocated :class:`~warp.optim.linear.CR` functor that can be called repeatedly on compatible systems (same shape, batch
            count, dtype, device) without re-allocating.

    Returns:
        If ``run`` is ``True`` and ``check_every`` > 0: Tuple (final_iteration, residual_norm, absolute_tolerance)
            - final_iteration: The number of iterations performed before convergence or reaching maxiter
            - residual_norm: The final residual norm ||b - Ax||
            - absolute_tolerance: The absolute tolerance used for convergence checking

        If ``run`` is ``True`` and ``check_every`` is 0: Tuple (final_iteration_array, residual_norm_squared_array, absolute_tolerance_squared_array)
            - final_iteration_array: Device array containing the number of iterations performed
            - residual_norm_squared_array: Device array containing the squared residual norm ||b - Ax||²
            - absolute_tolerance_squared_array: Device array containing the squared absolute tolerance

        If ``run`` is ``False``: a :class:`~warp.optim.linear.CR` functor with all temporary buffers pre-allocated.

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """
    state = CR(
        A,
        b,
        x,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=M,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
    )
    if run:
        return state()
    return state


class BiCGSTAB(LinearSolverState):
    """Pre-allocated state for the BiConjugate Gradient Stabilized solver.

    See :class:`LinearSolverState` for the shared constructor parameters, plus:

    Args:
        is_left_preconditioner: whether ``M`` should be used as a left- or right- preconditioner.

    Unlike :class:`~warp.optim.linear.CG` and :class:`~warp.optim.linear.CR`, the presence of ``M`` is fixed at construction:
    if ``M`` was ``None``, subsequent calls must also have ``M`` ``None``; otherwise a
    compatible preconditioner must be supplied.
    """

    def __init__(
        self,
        A: _Matrix,
        b: wp.array,
        x: wp.array,
        tol: float | None = None,
        atol: float | None = None,
        maxiter: float | None = 0,
        M: _Matrix | None = None,
        callback: Callable | None = None,
        check_every: int = 10,
        use_cuda_graph: bool = True,
        is_left_preconditioner: bool = False,
    ):
        self._is_left_preconditioner = is_left_preconditioner
        super().__init__(
            A,
            b,
            x,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            M=M,
            callback=callback,
            check_every=check_every,
            use_cuda_graph=use_cuda_graph,
        )

    def _allocate(self):
        A = self._A
        b = self._b
        M = self._M
        device = self._device
        scalar_type = self._scalar_type
        batch_count = self._batch_count

        # Notations follow https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
        self._r_and_r0 = wp.empty((2, b.shape[0]), dtype=b.dtype, device=device)
        self._p = wp.empty_like(b)
        self._v = wp.empty_like(b)
        self._t = wp.empty_like(b)
        self._r_repeated = _repeat_first(self._r_and_r0)

        # Preconditioner-dependent buffers are fixed at construction
        if M is not None:
            self._y = wp.zeros_like(self._p)
            self._z = wp.zeros_like(self._r_and_r0[0])
            if self._is_left_preconditioner:
                self._Mt = wp.zeros_like(self._t)
            else:
                self._Mt = self._t
        else:
            self._y = self._p
            self._z = self._r_and_r0[0]
            self._Mt = self._t

        self._tiled_dot = TiledDot(
            max_length=_scalar_dof_count(b),
            device=device,
            scalar_type=scalar_type,
            max_column_count=5,
            batch_offsets=A.batch_offsets,
        )

        self._atol_sq = wp.empty(batch_count, dtype=scalar_type, device=device)

    def _check_compatible(self, A, b, x, M):
        super()._check_compatible(A, b, x, M)
        # BiCGSTAB allocation depends on whether M is provided — require consistency
        if (M is None) != (self._M is None):
            raise ValueError(
                "BiCGSTAB requires M to be consistently provided between construction and call "
                "(both None or both non-None)"
            )

    def _run(self, A, b, x, M):
        device = self._device
        batch_offsets = A.batch_offsets
        dofs_per_entry = self._dofs_per_entry
        tiled_dot = self._tiled_dot
        is_left_preconditioner = self._is_left_preconditioner

        r_and_r0 = self._r_and_r0
        p = self._p
        v = self._v
        t = self._t
        r_repeated = self._r_repeated

        r, r0 = r_and_r0[0], r_and_r0[1]
        y = self._y
        z = self._z
        Mt = self._Mt

        r_norm_sq = tiled_dot.col(0)
        rho = tiled_dot.col(1)
        atol_sq = self._atol_sq

        # Initialize tolerance from right-hand-side norm
        _initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
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
                inputs=[atol_sq, r_norm_sq, rho, r0v, x, r, y, v, batch_offsets, dofs_per_entry],
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
                inputs=[atol_sq, r_norm_sq, st, tt, z, t, x, r, batch_offsets, dofs_per_entry],
            )

            # r = <r,r>, rho = <r0, r>
            tiled_dot.compute(r_and_r0, r_repeated, col_offset=0)

            # beta = (rho / rho_old) * alpha / omega = (rho / r0v) / omega
            # p = r + beta (p - omega v)
            wp.launch(
                kernel=_bicgstab_kernel_3,
                dim=z.shape[0],
                device=device,
                inputs=[atol_sq, r_norm_sq, rho, r0v, st, tt, p, r, v, batch_offsets, dofs_per_entry],
            )

        return _run_capturable_loop(
            do_iteration,
            r_norm_sq=r_norm_sq,
            maxiter=self._maxiter,
            atol_sq=atol_sq,
            callback=self._callback,
            check_every=self._check_every,
            use_cuda_graph=self._use_cuda_graph,
        )


def bicgstab(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: float | None = None,
    atol: float | None = None,
    maxiter: float | None = 0,
    M: _Matrix | None = None,
    callback: Callable | None = None,
    check_every=10,
    use_cuda_graph=True,
    is_left_preconditioner=False,
    run: bool = True,
):
    """Compute an approximate solution to a linear system using the Biconjugate Gradient Stabilized method (BiCGSTAB).

    Supports batched systems when ``A`` is a :class:`LinearOperator` with ``batch_offsets`` set;
    all subproblems iterate together and convergence uses the worst-case residual.

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
        run: If ``True`` (default), allocate temporary buffers and immediately run the solver. If ``False``, return a
            pre-allocated :class:`~warp.optim.linear.BiCGSTAB` functor that can be called repeatedly on compatible systems (same shape,
            batch count, dtype, device) without re-allocating. Whether ``M`` was provided at construction must match
            subsequent calls.

    Returns:
        If ``run`` is ``True`` and ``check_every`` > 0: Tuple (final_iteration, residual_norm, absolute_tolerance)
            - final_iteration: The number of iterations performed before convergence or reaching maxiter
            - residual_norm: The final residual norm ||b - Ax||
            - absolute_tolerance: The absolute tolerance used for convergence checking

        If ``run`` is ``True`` and ``check_every`` is 0: Tuple (final_iteration_array, residual_norm_squared_array, absolute_tolerance_squared_array)
            - final_iteration_array: Device array containing the number of iterations performed
            - residual_norm_squared_array: Device array containing the squared residual norm ||b - Ax||²
            - absolute_tolerance_squared_array: Device array containing the squared absolute tolerance

        If ``run`` is ``False``: a :class:`~warp.optim.linear.BiCGSTAB` functor with all temporary buffers pre-allocated.

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """
    state = BiCGSTAB(
        A,
        b,
        x,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=M,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
        is_left_preconditioner=is_left_preconditioner,
    )
    if run:
        return state()
    return state


class GMRES(LinearSolverState):
    """Pre-allocated state for the restarted Generalized Minimum Residual solver.

    See :class:`LinearSolverState` for the shared constructor parameters, plus:

    Args:
        restart: the ``k`` in ``GMRES[k]``. Determines the size of the Krylov subspace and the
            corresponding Hessenberg/basis allocations. Larger values reduce iteration count at
            the cost of memory.
        is_left_preconditioner: whether ``M`` should be used as a left- or right- preconditioner.

    The preconditioner ``M`` may be freely changed (or toggled between ``None`` and a valid
    operator) between calls.
    """

    def __init__(
        self,
        A: _Matrix,
        b: wp.array,
        x: wp.array,
        tol: float | None = None,
        atol: float | None = None,
        restart: int = 31,
        maxiter: float | None = 0,
        M: _Matrix | None = None,
        callback: Callable | None = None,
        check_every: int = 31,
        use_cuda_graph: bool = True,
        is_left_preconditioner: bool = False,
    ):
        self._restart = restart
        self._is_left_preconditioner = is_left_preconditioner
        super().__init__(
            A,
            b,
            x,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            M=M,
            callback=callback,
            check_every=check_every,
            use_cuda_graph=use_cuda_graph,
        )

    def _allocate(self):
        A = self._A
        b = self._b
        device = self._device
        scalar_dtype = self._scalar_type
        batch_count = self._batch_count

        # Cap restart at maxiter; align check_every with restart
        restart = min(self._restart, self._maxiter)
        self._restart = restart
        if self._check_every > 0:
            self._check_every = max(restart, self._check_every)

        self._pivot_tolerance = _get_dtype_epsilon(scalar_dtype) ** 2

        self._r = wp.empty_like(b)
        self._w = wp.empty_like(self._r)

        # Per-batch Hessenberg + LS solution. Batch-major so H[bid] is a contiguous
        # (restart+1, restart) slice usable by the LS kernels.
        self._H = wp.empty(shape=(batch_count, restart + 1, restart), dtype=scalar_dtype, device=device)
        self._y = wp.empty(shape=(batch_count, restart + 1), dtype=scalar_dtype, device=device)

        self._V = wp.zeros(shape=(restart + 1, self._r.shape[0]), dtype=self._r.dtype, device=device)

        self._residuals = wp.empty((2, batch_count), dtype=scalar_dtype, device=device)
        self._beta = self._residuals[0]
        self._atol_sq = self._residuals[1]

        self._tiled_dot = TiledDot(
            max_length=_scalar_dof_count(b),
            device=device,
            scalar_type=scalar_dtype,
            max_column_count=restart + 1,
            batch_offsets=A.batch_offsets,
        )

        w = self._w
        self._w_repeated = wp.array(
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

        self._tile_size = tile_size

        # recorded launches — all reference internal buffers, safe to reuse across calls
        # Scalar LS uses a (batch, 1) grid; tiled LS uses (batch, tile_size) with one block per batch.
        self._least_squares_solve = wp.launch(
            least_squares_kernel,
            dim=(batch_count, tile_size),
            block_dim=tile_size if tile_size > 1 else 256,
            device=device,
            inputs=[restart, self._pivot_tolerance, self._beta, self._H, self._y],
            record_cmd=True,
        )

        self._normalize_arnoldi_vec = wp.launch(
            _gmres_arnoldi_normalize_kernel,
            dim=self._r.shape,
            device=self._r.device,
            inputs=[self._r, self._w, self._tiled_dot.col(0), self._beta, A.batch_offsets, self._dofs_per_entry],
            record_cmd=True,
        )

        self._arnoldi_axpy = wp.launch(
            _gmres_arnoldi_axpy_kernel,
            dim=(self._w.shape[0], tile_size),
            block_dim=tile_size,
            device=self._w.device,
            inputs=[0, self._V, self._w, self._H, A.batch_offsets, self._dofs_per_entry],
            record_cmd=True,
        )

        # Transpose-copy of tiled_dot output (j+1, batch) into H[:, :j+1, j] (batch, j+1).
        # Launched with a fixed grid of (restart+1, batch) so CUDA graphs see a stable
        # launch size; the kernel guards against k >= k_count.
        self._copy_hessenberg_col = wp.launch(
            _gmres_copy_hessenberg_column,
            dim=(restart + 1, batch_count),
            device=device,
            inputs=[0, self._tiled_dot.cols(restart + 1), self._H],
            record_cmd=True,
        )

    def _run(self, A, b, x, M):
        device = self._device
        scalar_dtype = self._scalar_type
        restart = self._restart
        is_left_preconditioner = self._is_left_preconditioner
        batch_offsets = A.batch_offsets
        dofs_per_entry = self._dofs_per_entry

        tiled_dot = self._tiled_dot
        r = self._r
        w = self._w
        H = self._H
        y = self._y
        V = self._V
        beta = self._beta
        atol_sq = self._atol_sq
        w_repeated = self._w_repeated
        r_norm_sq = tiled_dot.col(0)

        least_squares_solve = self._least_squares_solve
        normalize_arnoldi_vec = self._normalize_arnoldi_vec
        arnoldi_axpy = self._arnoldi_axpy
        copy_hessenberg_col = self._copy_hessenberg_col

        # Initialize tolerance from right-hand-side norm
        _initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
        # Initialize residual
        A.matvec(x, b, r, alpha=-1.0, beta=1.0)
        tiled_dot.compute(r, r, col_offset=0)

        # Not strictly necessary, but makes it more robust to user-provided LinearOperators
        w.zero_()

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

            # compute and apply dot products in parallel,
            # since the Hj column entries are independent per batch
            tiled_dot.compute(w_repeated, V[: j + 1])
            copy_hessenberg_col.set_param_at_index(0, j)
            copy_hessenberg_col.launch()

            # w -= sum_k H[:, k, j] * v[k]
            arnoldi_axpy.set_param_at_index(0, j)
            arnoldi_axpy.launch()

            # H[:, j+1, j] = ||w||; normalize w into v[j+1]
            tiled_dot.compute(w, w)
            normalize_arnoldi_vec.set_params(
                [w, V[j + 1], tiled_dot.col(0), H[:, j + 1, j], batch_offsets, dofs_per_entry]
            )
            normalize_arnoldi_vec.launch()

        def do_restart_cycle():
            if M is not None and is_left_preconditioner:
                M.matvec(r, w, w, alpha=1, beta=0)
                rh = w
            else:
                rh = r

            # beta^2 = rh.rh
            tiled_dot.compute(rh, rh)

            # v[0] = rh / beta
            normalize_arnoldi_vec.set_params([rh, V[0], tiled_dot.col(0), beta, batch_offsets, dofs_per_entry])
            normalize_arnoldi_vec.launch()

            for j in range(restart):
                do_arnoldi_iteration(j)

            least_squares_solve.launch()

            # update x
            if M is None or is_left_preconditioner:
                wp.launch(
                    _gmres_update_x_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[restart, scalar_dtype(1.0), y, V, x, batch_offsets, dofs_per_entry],
                )
            else:
                wp.launch(
                    _gmres_update_x_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[restart, scalar_dtype(0.0), y, V, w, batch_offsets, dofs_per_entry],
                )
                M.matvec(w, x, x, alpha=1, beta=1)

            # update r and residual
            wp.copy(src=b, dest=r)
            A.matvec(x, b, r, alpha=-1.0, beta=1.0)
            tiled_dot.compute(r, r)

        return _run_capturable_loop(
            do_restart_cycle,
            cycle_size=restart,
            r_norm_sq=r_norm_sq,
            maxiter=self._maxiter,
            atol_sq=atol_sq,
            callback=self._callback,
            check_every=self._check_every,
            use_cuda_graph=self._use_cuda_graph,
        )


def gmres(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: float | None = None,
    atol: float | None = None,
    restart=31,
    maxiter: float | None = 0,
    M: _Matrix | None = None,
    callback: Callable | None = None,
    check_every=31,
    use_cuda_graph=True,
    is_left_preconditioner=False,
    run: bool = True,
):
    """Compute an approximate solution to a linear system using the restarted Generalized Minimum Residual method (GMRES[k]).

    Supports batched systems when ``A`` is a :class:`LinearOperator` with ``batch_offsets`` set;
    all subproblems iterate together and convergence uses the worst-case residual.

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
        run: If ``True`` (default), allocate temporary buffers and immediately run the solver. If ``False``, return a
            pre-allocated :class:`~warp.optim.linear.GMRES` functor that can be called repeatedly on compatible systems (same shape,
            dtype, device) without re-allocating.

    Returns:
        If ``run`` is ``True`` and ``check_every`` > 0: Tuple (final_iteration, residual_norm, absolute_tolerance)
            - final_iteration: The number of iterations performed before convergence or reaching maxiter
            - residual_norm: The final residual norm ||b - Ax||
            - absolute_tolerance: The absolute tolerance used for convergence checking

        If ``run`` is ``True`` and ``check_every`` is 0: Tuple (final_iteration_array, residual_norm_squared_array, absolute_tolerance_squared_array)
            - final_iteration_array: Device array containing the number of iterations performed
            - residual_norm_squared_array: Device array containing the squared residual norm ||b - Ax||²
            - absolute_tolerance_squared_array: Device array containing the squared absolute tolerance

        If ``run`` is ``False``: a :class:`~warp.optim.linear.GMRES` functor with all temporary buffers pre-allocated.

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """
    state = GMRES(
        A,
        b,
        x,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        M=M,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
        is_left_preconditioner=is_left_preconditioner,
    )
    if run:
        return state()
    return state


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


@wp.func
def _find_batch(dof: int, batch_offsets: wp.array(dtype=int)) -> int:
    """Binary search for the batch containing ``dof`` in ``batch_offsets``."""

    if not batch_offsets:
        return 0

    batch_count = batch_offsets.shape[0] - 1
    return wp.where(dof < batch_offsets[batch_count], wp.lower_bound(batch_offsets, 0, batch_count + 1, dof + 1), 0) - 1


@wp.func
def _find_entry_batch(entry: int, batch_offsets: wp.array(dtype=int), dofs_per_entry: int) -> int:
    return _find_batch(entry * dofs_per_entry, batch_offsets)


@wp.kernel
def _initialize_tolerance(
    rtol: Any,
    atol: Any,
    r_norm_sq: wp.array(dtype=Any),
    atol_sq: wp.array(dtype=Any),
):
    i = wp.tid()
    a = wp.max(rtol * wp.sqrt(r_norm_sq[i]), atol)
    atol_sq[i] = a * a


def _initialize_absolute_tolerance(
    b: wp.array,
    tol: float,
    atol: float,
    tiled_dot: TiledDot,
    atol_sq: wp.array,
):
    scalar_type = atol_sq.dtype
    batch_count = atol_sq.shape[0]

    # Compute per-subproblem b norm to define absolute tolerances
    tiled_dot.compute(b, b)
    b_norm_sq = tiled_dot.col(0)

    rtol, atol = _get_tolerances(scalar_type, tol, atol)
    wp.launch(
        kernel=_initialize_tolerance,
        dim=batch_count,
        device=b.device,
        inputs=[scalar_type(rtol), scalar_type(atol), b_norm_sq, atol_sq],
    )


@functools.cache
def _create_update_condition_kernel(batch_count: int):
    tile_size = max(32, min(512, 1 << math.ceil(math.log2(max(batch_count, 1)))))

    @wp.kernel(module="unique")
    def _update_condition(
        maxiter: int,
        cycle_size: int,
        cur_iter: wp.array(dtype=int),
        r_norm_sq: wp.array(dtype=Any),
        atol_sq: wp.array(dtype=Any),
        condition: wp.array(dtype=int),
    ):
        _, lane = wp.tid()

        max_diff_tile = wp.tile_zeros(dtype=r_norm_sq.dtype, shape=(tile_size,))

        for i in range(0, batch_count, wp.block_dim()):
            r_norm_tile = wp.tile_load(r_norm_sq, shape=tile_size, offset=i)
            atol_tile = wp.tile_load(atol_sq, shape=tile_size, offset=i)
            diff_tile = wp.tile_map(wp.sub, r_norm_tile, atol_tile)
            max_diff_tile = wp.tile_map(wp.max, max_diff_tile, diff_tile)

        max_diff = wp.tile_max(max_diff_tile)
        converged = max_diff[0] <= r_norm_sq.dtype(0.0)
        if lane == 0:
            cur_iter[0] += cycle_size
            condition[0] = wp.where(converged or cur_iter[0] >= maxiter, 0, 1)

    return _update_condition, tile_size


def _run_capturable_loop(
    do_cycle: Callable,
    r_norm_sq: wp.array,
    maxiter: int,
    atol_sq: wp.array,
    callback: Callable | None,
    check_every: int,
    use_cuda_graph: bool,
    cycle_size: int = 1,
):
    device = atol_sq.device
    batch_count = atol_sq.shape[0]

    if check_every > 0:
        return _run_solver_loop(
            do_cycle, cycle_size, r_norm_sq, maxiter, atol_sq, callback, check_every, use_cuda_graph, device
        )

    cur_iter_and_condition = wp.full((2,), value=-1, dtype=int, device=device)
    cur_iter = cur_iter_and_condition[0:1]
    condition = cur_iter_and_condition[1:2]

    update_condition_kernel, update_condition_tile_size = _create_update_condition_kernel(batch_count)
    update_condition_launch = wp.launch(
        update_condition_kernel,
        dim=(1, update_condition_tile_size),
        block_dim=update_condition_tile_size,
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
            with wp.ScopedCapture(device=device) as capture:
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
    atol_sq: wp.array,
    callback: Callable,
    check_every: int,
    use_cuda_graph: bool,
    device,
):
    atol_sq_host = atol_sq.numpy()
    atol = math.sqrt(float(atol_sq_host.max()))
    check_every = max(check_every, cycle_size)

    cur_iter = 0

    # For batched solves r_norm_sq has shape (batch_count,); convergence requires each
    # batch to satisfy its own tolerance.
    r_norm_sq_host = r_norm_sq.numpy()
    err_sq = float(r_norm_sq_host.max())
    err = math.sqrt(err_sq)
    if callback is not None:
        callback(cur_iter, err, atol)

    if (r_norm_sq_host <= atol_sq_host).all():
        return cur_iter, err, atol

    graph = None

    while True:
        # Do not do graph capture at first iteration -- modules may not be loaded yet
        if device.is_cuda and use_cuda_graph and cur_iter > 0:
            if graph is None:
                with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                    do_cycle()
                graph = capture.graph
            wp.capture_launch(graph)
        else:
            do_cycle()

        cur_iter += cycle_size

        if cur_iter >= maxiter:
            break

        if (cur_iter % check_every) < cycle_size:
            r_norm_sq_host = r_norm_sq.numpy()
            err_sq = float(r_norm_sq_host.max())

            if (r_norm_sq_host <= atol_sq_host).all():
                break

            if callback is not None:
                callback(cur_iter, math.sqrt(err_sq), atol)

    err_sq = float(r_norm_sq.numpy().max())
    err = math.sqrt(err_sq)
    if callback is not None:
        callback(cur_iter, err, atol)

    return cur_iter, err, atol


@wp.kernel(module="unique")
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


@wp.kernel(module="unique")
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


@wp.kernel(module="unique")
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


@wp.kernel(module="unique")
def _extract_inverse_diagonal_scalar(
    diag_array: wp.array(dtype=Any),
    inv_diag: wp.array(dtype=Any),
    use_abs: int,
):
    i = wp.tid()
    inv_diag[i] = _inverse_diag_coefficient(diag_array[i], use_abs != 0)


@wp.kernel(module="unique")
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
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    i = wp.tid()
    bid = _find_entry_batch(i, batch_offsets, dofs_per_entry)

    alpha = wp.where(resid[bid] > tol[bid], rz_old[bid] / p_Ap[bid], rz_old.dtype(0.0))

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
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    #    p = r + (rz_new / rz_old) * p;
    i = wp.tid()
    bid = _find_entry_batch(i, batch_offsets, dofs_per_entry)

    beta = wp.where(resid_new[bid] > tol[bid], rz_new[bid] / rz_old[bid], rz_old.dtype(0.0))

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
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    i = wp.tid()
    bid = _find_entry_batch(i, batch_offsets, dofs_per_entry)

    alpha = wp.where(resid[bid] > tol[bid] and y_Ap[bid] > 0.0, zAz_old[bid] / y_Ap[bid], zAz_old.dtype(0.0))

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
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    #    p = r + (rz_new / rz_old) * p;
    i = wp.tid()
    bid = _find_entry_batch(i, batch_offsets, dofs_per_entry)

    beta = wp.where(resid[bid] > tol[bid] and zAz_old[bid] > 0.0, zAz_new[bid] / zAz_old[bid], zAz_old.dtype(0.0))

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
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    i = wp.tid()
    bid = _find_entry_batch(i, batch_offsets, dofs_per_entry)

    alpha = wp.where(resid[bid] > tol[bid], rho_old[bid] / r0v[bid], rho_old.dtype(0.0))

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
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    i = wp.tid()
    bid = _find_entry_batch(i, batch_offsets, dofs_per_entry)

    omega = wp.where(resid[bid] > tol[bid], st[bid] / tt[bid], st.dtype(0.0))

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
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    i = wp.tid()
    bid = _find_entry_batch(i, batch_offsets, dofs_per_entry)

    beta = wp.where(resid[bid] > tol[bid], rho_new[bid] * tt[bid] / (r0v[bid] * st[bid]), st.dtype(0.0))
    beta_omega = wp.where(resid[bid] > tol[bid], rho_new[bid] / r0v[bid], st.dtype(0.0))

    p[i] = r[i] + beta * p[i] - beta_omega * v[i]


@wp.kernel(enable_backward=False)
def _gmres_solve_least_squares(
    k: int,
    pivot_tolerance: float,
    beta: wp.array(dtype=Any),
    H: wp.array3d(dtype=Any),
    y: wp.array2d(dtype=Any),
):
    # Per-batch QR-by-Givens + back-solve of H y = (beta, 0, ..., 0).
    # H is Hessenberg of shape (batch, k+1, k); one thread per batch.

    bid, _lane = wp.tid()

    rhs = beta[bid]

    # Apply 2x2 rotations to H so as to remove lower diagonal,
    # and apply similar rotations to right-hand-side
    max_k = int(k)
    for i in range(k):
        Ha = H[bid, i]
        Hb = H[bid, i + 1]

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
        y[bid, i] = c * rhs
        rhs = -s * rhs

    for i in range(max_k, k):
        y[bid, i] = y.dtype(0.0)

    # Triangular back-solve for y
    for ii in range(max_k, 0, -1):
        i = ii - 1
        Hi = H[bid, i]
        yi = y[bid, i]
        for j in range(ii, max_k):
            yi -= Hi[j] * y[bid, j]
        y[bid, i] = yi / Hi[i]


@functools.cache
def make_gmres_solve_least_squares_kernel_tiled(K: int):
    @wp.kernel(module="unique", enable_backward=False)
    def gmres_solve_least_squares_tiled(
        k: int,
        pivot_tolerance: float,
        beta: wp.array(dtype=Any),
        H: wp.array3d(dtype=Any),
        y: wp.array2d(dtype=Any),
    ):
        # One CUDA block per batch, tile_size threads cooperate on that batch's LS.
        # Assumes tiles of size K, and K at least as large as highest number of columns.
        # Limits the max restart cycle length to the max block size of 1024.
        #
        # Solve H[bid] y[bid] = (beta[bid], 0, ..., 0) with H of shape (batch, k+1, k).

        bid, lane = wp.tid()

        rhs = beta[bid]

        zero = H.dtype(0.0)
        one = H.dtype(1.0)
        yi = zero

        Ha = wp.tile_load(H[bid, 0], shape=(K))

        # Apply 2x2 rotations to H so as to remove lower diagonal,
        # and apply similar rotations to right-hand-side
        max_k = int(k)
        for i in range(k):
            Hb = wp.tile_load(H[bid, i + 1], shape=(K))

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

            wp.tile_store(H[bid, i], wp.tile(a_rot))
            Ha[lane] = b_rot

        y_tile = wp.tile(yi)

        # Triangular back-solve for y
        for ii in range(max_k, 0, -1):
            i = ii - 1

            Hi = wp.tile_load(H[bid, i], shape=(K))

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

        wp.tile_store(y[bid], y_tile)

    return gmres_solve_least_squares_tiled


@wp.kernel(enable_backward=False)
def _gmres_arnoldi_axpy_kernel(
    j: int,
    V: wp.array2d(dtype=Any),
    w: wp.array(dtype=Any),
    H: wp.array3d(dtype=Any),
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    tid, lane = wp.tid()
    bid = _find_entry_batch(tid, batch_offsets, dofs_per_entry)

    s = w.dtype(H.dtype(0))

    tile_size = wp.block_dim()
    for k in range(lane, j + 1, tile_size):
        s += H[bid, k, j] * V[k, tid]

    wi = wp.tile_load(w, shape=1, offset=tid)
    wi -= wp.tile_sum(wp.tile(s, preserve_type=True))

    wp.tile_store(w, wi, offset=tid)


@wp.kernel(enable_backward=False)
def _gmres_arnoldi_normalize_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    alpha_copy: wp.array(dtype=Any),
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    tid = wp.tid()
    scalar_dof = tid * dofs_per_entry
    bid = _find_batch(scalar_dof, batch_offsets)
    a = alpha[bid]
    norm = wp.sqrt(a)
    y[tid] = wp.where(a == alpha.dtype(0.0), x[tid], x[tid] / norm)

    if not batch_offsets:
        if tid == 0:
            alpha_copy[0] = norm
    elif scalar_dof == batch_offsets[bid]:
        alpha_copy[bid] = norm


@wp.kernel(enable_backward=False)
def _gmres_copy_hessenberg_column(
    j: int,
    src: wp.array2d(dtype=Any),
    H: wp.array3d(dtype=Any),
):
    k, bid = wp.tid()
    if k <= j:
        H[bid, k, j] = src[k, bid]


@wp.kernel(enable_backward=False)
def _gmres_update_x_kernel(
    k: int,
    scale: Any,
    y: wp.array2d(dtype=Any),
    V: wp.array2d(dtype=Any),
    x: wp.array(dtype=Any),
    batch_offsets: wp.array(dtype=int),
    dofs_per_entry: int,
):
    tid = wp.tid()
    bid = _find_entry_batch(tid, batch_offsets, dofs_per_entry)

    xi = scale * x[tid]
    for j in range(k):
        xi += V[j, tid] * y[bid, j]

    x[tid] = xi


def _register_overloads():
    # Pre-register float32 and float64 overloads so the module can be AOT-compiled
    # without requiring a prior runtime launch.
    for dtype in (wp.float32, wp.float64):
        a = wp.array(dtype=dtype)
        a2 = wp.array2d(dtype=dtype)
        a3 = wp.array3d(dtype=dtype)

        wp.overload(_initialize_tolerance, [dtype, dtype, a, a])
        wp.overload(_cg_kernel_1, {"tol": a, "resid": a, "rz_old": a, "p_Ap": a, "x": a, "r": a, "p": a, "Ap": a})
        wp.overload(_cg_kernel_2, {"tol": a, "resid_new": a, "rz_old": a, "rz_new": a, "z": a, "p": a})
        wp.overload(
            _cr_kernel_1,
            {"tol": a, "resid": a, "zAz_old": a, "y_Ap": a, "x": a, "r": a, "z": a, "p": a, "Ap": a, "y": a},
        )
        wp.overload(_cr_kernel_2, {"tol": a, "resid": a, "zAz_old": a, "zAz_new": a, "z": a, "p": a, "Az": a, "Ap": a})
        wp.overload(_bicgstab_kernel_1, {"tol": a, "resid": a, "rho_old": a, "r0v": a, "x": a, "r": a, "y": a, "v": a})
        wp.overload(_bicgstab_kernel_2, {"tol": a, "resid": a, "st": a, "tt": a, "z": a, "t": a, "x": a, "r": a})
        wp.overload(
            _bicgstab_kernel_3, {"tol": a, "resid": a, "rho_new": a, "r0v": a, "st": a, "tt": a, "p": a, "r": a, "v": a}
        )
        wp.overload(_gmres_solve_least_squares, {"beta": a, "H": a3, "y": a2})
        wp.overload(_gmres_arnoldi_axpy_kernel, {"V": a2, "w": a, "H": a3})
        wp.overload(_gmres_arnoldi_normalize_kernel, {"x": a, "y": a, "alpha": a, "alpha_copy": a})
        wp.overload(_gmres_copy_hessenberg_column, {"src": a2, "H": a3})
        wp.overload(_gmres_update_x_kernel, {"scale": dtype, "y": a2, "V": a2, "x": a})


_register_overloads()
