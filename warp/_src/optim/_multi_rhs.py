# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-operator multiple-RHS actions using upstream CG."""

import copy
import functools
from operator import index
from typing import Any

import warp as wp
from warp._src import sparse as sp
from warp._src.context import is_conditional_graph_supported
from warp._src.optim._multi_rhs_kernels import csr_block_product, csr_eight_rhs, transpose_axpby, transpose_rows
from warp._src.optim._solve_action import (
    ResidualMode,
    _positive_int,
    _resolve_tolerances,
    _scalar_shape,
    _SolveAction,
    _StreamOrder,
)
from warp._src.optim.linear import LinearOperator, aslinearoperator, cg
from warp._src.types import type_scalar_type


@functools.lru_cache(None)
def bsr_eight_rhs(dtype):
    vector = wp.types.vector(length=3, dtype=dtype)
    matrix = wp.types.matrix(shape=(3, 3), dtype=dtype)

    @wp.kernel(module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[matrix],
        x: wp.array2d[dtype],
        y: wp.array2d[dtype],
    ):
        group, row, lane = wp.tid()
        rhs, slot = group * 8 + lane % 8, lane // 8
        value = vector(dtype(0.0))
        if rhs < x.shape[1]:
            for e in range(offsets[row] + slot, offsets[row + 1], 4):
                col = columns[e] * 3
                value += values[e] * vector(x[col, rhs], x[col + 1, rhs], x[col + 2, rhs])
        a = wp.tile_sum(wp.tile_reshape(wp.tile(value[0]), shape=(4, 8)), axis=0)
        b = wp.tile_sum(wp.tile_reshape(wp.tile(value[1]), shape=(4, 8)), axis=0)
        c = wp.tile_sum(wp.tile_reshape(wp.tile(value[2]), shape=(4, 8)), axis=0)
        if rhs < x.shape[1] and slot == 0:
            y[row * 3, rhs] = wp.tile_extract(a, lane % 8)
            y[row * 3 + 1, rhs] = wp.tile_extract(b, lane % 8)
            y[row * 3 + 2, rhs] = wp.tile_extract(c, lane % 8)

    return compute


@functools.lru_cache(None)
def _bsr_row_product(dtype):
    vector = wp.types.vector(length=3, dtype=dtype)
    matrix = wp.types.matrix(shape=(3, 3), dtype=dtype)

    @wp.kernel(module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[matrix],
        x: wp.array2d[dtype],
        y: wp.array2d[dtype],
    ):
        group, row, lane = wp.tid()
        rhs, slot = group * 4 + lane // 8, lane % 8
        value = vector(dtype(0.0))
        if rhs < x.shape[0]:
            for e in range(offsets[row] + slot, offsets[row + 1], 8):
                col = columns[e] * 3
                value += values[e] * vector(x[rhs, col], x[rhs, col + 1], x[rhs, col + 2])
        a = wp.tile_sum(wp.tile_reshape(wp.tile(value[0]), shape=(4, 8)), axis=1)
        b = wp.tile_sum(wp.tile_reshape(wp.tile(value[1]), shape=(4, 8)), axis=1)
        c = wp.tile_sum(wp.tile_reshape(wp.tile(value[2]), shape=(4, 8)), axis=1)
        if rhs < x.shape[0] and slot == 0:
            y[rhs, row * 3] = wp.tile_extract(a, lane // 8)
            y[rhs, row * 3 + 1] = wp.tile_extract(b, lane // 8)
            y[rhs, row * 3 + 2] = wp.tile_extract(c, lane // 8)

    return compute


@wp.kernel(enable_backward=False)
def _offsets(offsets: wp.array[int], n: int):
    i = wp.tid()
    offsets[i] = i * n


@wp.kernel(enable_backward=False)
def _cast(x: wp.array[Any], y: wp.array[Any]):
    i = wp.tid()
    y[i] = y.dtype(x[i])


@wp.kernel(enable_backward=False)
def _combine(x: wp.array[Any], y: wp.array[Any], z: wp.array[Any], alpha: wp.float64, beta: wp.float64):
    i = wp.tid()
    v = z.dtype(0)
    if alpha != wp.float64(0):
        v = z.dtype(alpha) * x[i]
    if beta != wp.float64(0):
        v += z.dtype(beta) * y[i]
    z[i] = v


@functools.lru_cache(None)
def _cpu_product(block_type):
    scalar = type_scalar_type(block_type)
    block = 1 if block_type == scalar else block_type._shape_[0]

    @wp.kernel(enable_backward=False, module="unique")
    def product(
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[block_type],
        x: wp.array2d[scalar],
        y: wp.array2d[scalar],
    ):
        rhs, row = wp.tid()
        value = scalar(0)
        for e in range(offsets[row // wp.static(block)], offsets[row // wp.static(block) + 1]):
            if wp.static(block == 1):
                value += values[e] * x[rhs, columns[e]]
            else:
                for j in range(wp.static(block)):
                    value += values[e][row % wp.static(block), j] * x[rhs, columns[e] * wp.static(block) + j]
        y[rhs, row] = value

    return product


class MultiRHSOperator(LinearOperator):
    """Repeat one sparse matrix or linear operator over independent RHS rows.

    This is a logical block-diagonal operator, with no duplicated matrix storage.
    Inputs to ``matvec`` are contiguous flat scalar arrays representing ``(nrhs,n)``.
    CSR and 3x3 BSR use shared sparse-data CUDA products; arbitrary operators use
    a serial RHS loop. CPU uses a scalar-row product. Exact aliases are supported;
    shifted output overlap and strided storage are rejected before launching.

    Each object owns its scratch. ``with_nrhs`` creates smaller capacities sharing
    it. Eager applications issued from different CUDA streams are ordered with an
    event; simultaneously executing graphs that share scratch require caller
    ordering. Value updates in existing buffers are observed on replay;
    topology/address changes require recapture. No adjoint is supplied.

    Args:
        A: Square sparse matrix or a single-system :class:`LinearOperator`.
        nrhs: Positive RHS count.
        layout: ``"columns"`` enables coalesced sparse CUDA products. ``"rows"``
            retains RHS-row storage; compare both for narrow batches. CPU and
            non-sparse operators always use ``"rows"``.
    """

    def __init__(self, A, nrhs: int, *, layout: str = "columns"):
        if isinstance(nrhs, bool):
            raise TypeError("nrhs must be an integer, not bool")
        if index(nrhs) < 1:
            raise ValueError("nrhs must be a positive integer")
        self.source = A
        self.single = aslinearoperator(A)
        self.n, self.nrhs = _scalar_shape(self.single)[0], index(nrhs)
        if (
            self.n < 1
            or _scalar_shape(self.single) not in ((self.n,), (self.n, self.n))
            or self.single.batch_offsets is not None
        ):
            raise ValueError("Expected one nonempty square system")
        if layout not in ("rows", "columns"):
            raise ValueError("layout must be 'rows' or 'columns'")
        dtype = self.single.scalar_type
        if dtype not in (wp.float32, wp.float64):
            raise ValueError("A must have float32 or float64 scalar type")
        self.sparse = isinstance(A, sp.BsrMatrix)
        if self.sparse and (A.row_counts is not None or A.block_shape not in ((1, 1), (3, 3))):
            raise ValueError("Sparse inputs must be compressed scalar or 3x3 BSR matrices")
        self.layout = layout if self.single.device.is_cuda and self.sparse else "rows"
        shape = (self.n, self.nrhs) if self.layout == "columns" else (self.nrhs, self.n)
        self._first = wp.empty(shape, dtype=dtype, device=self.single.device)
        self._second = wp.empty_like(self._first) if self.layout == "columns" else None
        offsets = wp.empty(self.nrhs + 1, dtype=int, device=self.single.device)
        wp.launch(_offsets, self.nrhs + 1, [offsets, self.n], device=self.single.device)
        stream = self.single.device.stream if self.single.device.is_cuda else None
        if stream is not None and not stream.is_capturing:
            # Solver setup reads batch offsets before any matvec joins the stream ordering,
            # so finish writing them before another stream can use this operator.
            wp.synchronize_stream(stream)
        self._stream_order = _StreamOrder(self.single.device)
        super().__init__(
            (self.n * self.nrhs, self.n * self.nrhs),
            dtype,
            self.single.device,
            self._apply,
            batch_offsets=offsets,
            max_batch_length=self.n,
        )

    def with_nrhs(self, nrhs: int):
        """Prepare a smaller capacity sharing scratch and its stream ordering."""
        if isinstance(nrhs, bool):
            raise TypeError("nrhs must be an integer, not bool")
        if not 1 <= index(nrhs) <= self.nrhs:
            raise ValueError("nrhs must be positive and at most the prepared capacity")
        result = copy.copy(self)
        result.nrhs = index(nrhs)
        count = result.nrhs * self.n
        shape = (self.n, result.nrhs) if self.layout == "columns" else (result.nrhs, self.n)
        result._first = self._first.flatten()[:count].reshape(shape)
        if self._second is not None:
            result._second = self._second.flatten()[:count].reshape(shape)
        result._shape = (count, count)
        result._batch_offsets = self.batch_offsets[: result.nrhs + 1]
        result._matvec = result._apply
        return result

    def _apply(self, x, y, z, alpha, beta):
        for a in (x, y, z):
            if (
                a.shape != (self.n * self.nrhs,)
                or a.dtype != self.dtype
                or a.device != self.device
                or not a.is_contiguous
            ):
                raise ValueError("Expected flat contiguous arrays matching the prepared size, dtype and device")
        size = x.size * wp.types.type_size_in_bytes(x.dtype)
        for a in (x, y):
            if a.ptr != z.ptr and max(a.ptr, z.ptr) < min(a.ptr + size, z.ptr + size):
                raise ValueError("Partially overlapping output views are unsupported")
        self._stream_order.acquire(self.device)
        if alpha == 0:
            wp.launch(_combine, z.size, [x, y, z, wp.float64(0), wp.float64(beta)], device=self.device)
            return
        if not self.sparse:
            for j in range(self.nrhs):
                sl = slice(j * self.n, (j + 1) * self.n)
                self.single.matvec(x[sl], self._first.flatten()[sl], self._first.flatten()[sl], 1.0, 0.0)
        elif self.layout == "rows":
            output = z if alpha == 1 and beta == 0 and z.ptr != x.ptr else self._first.flatten()
            if self.device.is_cuda and self.source.block_shape == (1, 1):
                csr_block_product(self.source, x.reshape((self.nrhs, self.n)), output.reshape((self.nrhs, self.n)))
            elif self.device.is_cuda:
                wp.launch_tiled(
                    _bsr_row_product(self.dtype),
                    dim=((self.nrhs + 3) // 4, self.source.nrow),
                    inputs=[
                        self.source.offsets,
                        self.source.columns,
                        self.source.values,
                        x.reshape((self.nrhs, self.n)),
                        output.reshape((self.nrhs, self.n)),
                    ],
                    block_dim=32,
                    device=self.device,
                )
            else:
                wp.launch(
                    _cpu_product(self.source.dtype),
                    (self.nrhs, self.n),
                    [
                        self.source.offsets,
                        self.source.columns,
                        self.source.values,
                        x.reshape((self.nrhs, self.n)),
                        output.reshape((self.nrhs, self.n)),
                    ],
                    device=self.device,
                )
            if output.ptr == z.ptr:
                return
        else:
            wp.launch_tiled(
                transpose_rows,
                dim=((self.nrhs + 31) // 32, (self.n + 31) // 32),
                inputs=[x.reshape((self.nrhs, self.n)), self._first],
                block_dim=128,
                device=self.device,
            )
            a = self.source
            kernel = csr_eight_rhs if a.block_shape == (1, 1) else bsr_eight_rhs(a.scalar_type)
            wp.launch_tiled(
                kernel,
                dim=((self.nrhs + 7) // 8, a.nrow),
                inputs=[a.offsets, a.columns, a.values, self._first, self._second],
                block_dim=32,
                device=self.device,
            )
            wp.launch_tiled(
                transpose_axpby,
                dim=((self.n + 31) // 32, (self.nrhs + 31) // 32),
                inputs=[
                    self._second,
                    y.reshape((self.nrhs, self.n)),
                    z.reshape((self.nrhs, self.n)),
                    wp.float64(alpha),
                    wp.float64(beta),
                ],
                block_dim=128,
                device=self.device,
            )
            return
        wp.launch(
            _combine, z.size, [self._first.flatten(), y, z, wp.float64(alpha), wp.float64(beta)], device=self.device
        )


class CGSolve(_SolveAction):
    """Prepared independent multiple-RHS CG solves sharing one sparse SPD matrix.

    This wraps :func:`cg`; it is not a coupled block-CG recurrence. Inputs and
    outputs use the source scalar dtype; inner arithmetic may use another working
    dtype. Every call starts at zero and refreshes any working-precision matrix
    copy on device. Truncated or tolerance-stopped CG is not a fixed linear
    inverse. Matrix topology and storage must stay fixed after preparation; a
    reallocated source raises on eager calls. Construction and warmup precede
    capture. Replay performs no host convergence readbacks or work-array
    allocations. Eager calls from different CUDA streams are ordered
    automatically; captured graphs sharing one workspace must not execute
    concurrently. Reverse-mode differentiation is not supported.

    The RHSs iterate together, so every RHS reports the batch's scheduled
    iteration count, including iterations after it individually converged.

    Args:
        A: Compressed square SPD scalar or 3x3 BSR matrix, float32 or float64.
        nrhs: Positive number of independent right-hand sides.
        M: Fixed SPD preconditioner, either a single-system linear operator or a
            prepared multiple-RHS operator with ``nrhs`` equal partitions of length
            ``n``. The latter retains optimized application, such as coalesced
            FSAI. Preconditioner arithmetic must use ``working_dtype``.
        tol: Nonnegative relative per-RHS tolerance. Defaults to ``eps ** 0.75``
            for ``working_dtype``: 1e-6 for float32 and 1e-12 for float64.
        atol: Nonnegative absolute per-RHS tolerance. At least one tolerance must
            be positive.
        maxiter: Positive CG iteration budget; truncation is allowed.
        working_dtype: float32 or float64; defaults to A's scalar type.
        layout: Sparse application layout, ``"columns"`` or ``"rows"``.
        residual: Residual reported and used for acceptance. The default
            :attr:`ResidualMode.RECURRENCE` adds no operator application.
            :attr:`ResidualMode.FRESH` recomputes ``||A x - b||`` with the source
            matrix and precision, which validates mixed-precision or truncated
            solves at the cost of one batched product per call.

    Attributes:
        iterations: Device array of scheduled batch iterations per RHS from the
            latest call.
        absolute_residual: Device array of residual norms from the latest call.
            Nonfinite norms are reported as infinity.
        relative_residual: Device array of residual norms relative to ``||b||``.
            A zero RHS gives zero for a zero residual and infinity otherwise.
        tolerance: Device array of effective absolute tolerances from the latest call.
        converged: Device array of per-RHS acceptance flags, one for success.
        stats: Device array of cumulative iterations summed over RHSs, maximum
            iterations, failed RHSs and solved RHSs since ``reset_stats()``.
        residual_maxima: Device array of the maximum absolute and relative
            residuals since the last ``reset_stats()``.
    """

    def __init__(
        self,
        A,
        nrhs: int,
        *,
        M=None,
        tol: float | None = None,
        atol: float = 0.0,
        maxiter: int = 1000,
        working_dtype: type | None = None,
        layout: str = "columns",
        residual: ResidualMode = ResidualMode.RECURRENCE,
    ):
        maxiter = _positive_int("maxiter", maxiter)
        if not isinstance(A, sp.BsrMatrix) or A.shape[0] != A.shape[1] or A.shape[0] < 1:
            raise ValueError("A must be a nonempty square Warp sparse matrix")
        if (
            A.scalar_type not in (wp.float32, wp.float64)
            or A.block_shape not in ((1, 1), (3, 3))
            or A.row_counts is not None
        ):
            raise ValueError("A must be compressed scalar or 3x3 BSR in float32 or float64")
        working_dtype = A.scalar_type if working_dtype is None else working_dtype
        if working_dtype not in (wp.float32, wp.float64):
            raise ValueError("working_dtype must be float32 or float64")
        tol, atol = _resolve_tolerances(tol, atol, working_dtype)
        norm_dtype = wp.float64 if wp.float64 in (A.scalar_type, working_dtype) else wp.float32
        self._init_action(A.shape[0], nrhs, A.scalar_type, A.device, residual, norm_dtype)
        self.source = A
        self.working_dtype = working_dtype
        self.tol, self.atol, self.maxiter, self.layout = tol, atol, maxiter, layout
        self.working_matrix = A if self.working_dtype == self.dtype else sp.bsr_copy(A, scalar_type=self.working_dtype)
        self._storage = self._storage_identity()
        self.operator = MultiRHSOperator(self.working_matrix, self.nrhs, layout=layout)
        self.operator._stream_order = self._stream_order
        self._b = wp.zeros(self.n * self.nrhs, dtype=self.working_dtype, device=self.device)
        self._x = wp.zeros_like(self._b)
        self._source_operator = self._x_source = self._ax = None
        if residual == ResidualMode.FRESH:
            if self.working_dtype == self.dtype:
                self._source_operator, self._x_source = self.operator, self._x
            else:
                self._source_operator = MultiRHSOperator(A, self.nrhs, layout=layout)
                self._source_operator._stream_order = self._stream_order
                self._x_source = wp.empty(self.n * self.nrhs, dtype=self.dtype, device=self.device)
            self._ax = wp.empty(self.n * self.nrhs, dtype=self.dtype, device=self.device)
        self._prepare_state(M)

    def _prepare_state(self, M):
        self._M = M
        M = aslinearoperator(M)
        if M is not None:
            if M.device != self.device:
                raise ValueError("M must be on the system device")
            if M.batch_offsets is None:
                if M.scalar_type != self.working_dtype:
                    raise ValueError("Single-system M must use the working dtype; supply a prepared operator otherwise")
                if _scalar_shape(M) not in ((self.n,), (self.n, self.n)):
                    raise ValueError("Single-system M must match the matrix dimension")
                M = MultiRHSOperator(M, self.nrhs)
            elif M.shape != self.operator.shape or M.batch_count != self.nrhs or M.max_batch_length != self.n:
                raise ValueError("Prepared M must have nrhs equal partitions matching A")
            if M.scalar_type != self.working_dtype:
                raise ValueError("Prepared M must use the working dtype")
        self.state = cg(
            self.operator,
            self._b,
            self._x,
            M=M,
            tol=self.tol,
            atol=self.atol,
            maxiter=self.maxiter,
            check_every=0,
            use_cuda_graph=is_conditional_graph_supported(),
            run=False,
        )

    def with_nrhs(self, nrhs: int, *, M=None) -> "CGSolve":
        """Prepare a smaller capacity sharing matrix storage and staging buffers.

        The returned action owns its Krylov workspace and diagnostics but shares
        stream ordering, so it must not run concurrently with this action.

        Args:
            nrhs: Positive capacity no greater than this action's ``nrhs``.
            M: Preconditioner for the smaller capacity. Required when the original
                preconditioner was a prepared multiple-RHS operator; otherwise the
                original single-system preconditioner is reused by default.

        Returns:
            A :class:`CGSolve` that solves ``nrhs`` right-hand sides.
        """
        if M is None:
            M = self._M
            if M is not None and aslinearoperator(M).batch_offsets is not None:
                raise ValueError("Supply a matching prepared M for the smaller capacity")
        child = self._child(nrhs)
        count = self.n * child.nrhs
        child.operator = self.operator.with_nrhs(child.nrhs)
        child._b = self._b[:count]
        child._x = self._x[:count]
        if self._source_operator is not None:
            if self._source_operator is self.operator:
                child._source_operator, child._x_source = child.operator, child._x
            else:
                child._source_operator = self._source_operator.with_nrhs(child.nrhs)
                child._x_source = self._x_source[:count]
            child._ax = self._ax[:count]
        child._prepare_state(M)
        return child

    def _storage_identity(self):
        a = self.source
        return (a.offsets.ptr, a.columns.ptr, a.values.ptr, a.values.size)

    def update(self):
        """Copy current source matrix values into the working-precision matrix.

        :meth:`solve` also does this on every call, so explicit calls are only
        needed by solver-agnostic code. It is graph capturable.

        Raises:
            ValueError: If the source matrix storage was reallocated after preparation.
        """
        if self._storage_identity() != self._storage:
            raise ValueError("A's sparse storage changed after preparation; prepare a new CGSolve")
        if self.working_dtype == self.dtype:
            return
        self._stream_order.acquire(self.device)
        source_flat = self.source.values.view(self.dtype).flatten()
        target_flat = self.working_matrix.values.view(self.working_dtype).flatten()
        # bsr_copy may size the copy to nnz while the source keeps extra capacity.
        wp.launch(_cast, min(source_flat.size, target_flat.size), [source_flat, target_flat], device=self.device)

    def solve(self, rhs: wp.array, out: wp.array):
        """Solve from zero into caller-owned storage and refresh device diagnostics.

        In-place and aliased arrays are safe because all RHSs are staged before
        any output is written.

        Args:
            rhs: Contiguous array of shape ``(nrhs, n)``, or ``(n,)`` for one RHS,
                with the source matrix dtype and device.
            out: Matching output. Exact in-place aliasing is supported, but
                shifted overlapping views are rejected before any output mutation.

        Raises:
            ValueError: If arrays do not match, or if the source matrix storage
                was reallocated after preparation.
        """
        rows, result = self._prepare_solve(rhs, out)
        self.update()
        if self.working_dtype != self.dtype:
            wp.launch(_cast, self._b.size, [rows.flatten(), self._b], device=self.device)
        else:
            wp.copy(self._b, rows.flatten())
        self._x.zero_()
        iterations, residual_sq, tolerance_sq = self.state()
        solution = self._x
        if self.residual == ResidualMode.FRESH:
            if self._x_source is not self._x:
                wp.launch(_cast, self._x.size, [self._x, self._x_source], device=self.device)
            solution = self._x_source
            # The caller's RHS is still intact because no output has been written yet.
            self._source_operator.matvec(solution, self._ax, self._ax, 1.0, 0.0)
            self._squared_norms(self._ax.reshape((self.nrhs, self.n)), rows, 0)
            residual_sq = self._residual_sq
        else:
            self._squared_norms(None, rows, 0)
        self._record(iterations, residual_sq, tolerance_sq, 0, self.nrhs)
        if solution.dtype != self.dtype:
            wp.launch(_cast, solution.size, [solution, result.flatten()], device=self.device)
        else:
            wp.copy(result.flatten(), solution)
