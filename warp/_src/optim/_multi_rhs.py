# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Portions adapted from warpack; the original MIT notice is retained below.

# MIT License
#
# Copyright (c) 2026 warpack contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Shared-operator multiple-RHS actions using upstream CG."""

import copy
import functools
import math
from operator import index
from typing import Any

import warp as wp
from warp._src import sparse as sp
from warp._src.optim._multi_rhs_kernels import csr_block_product, csr_eight_rhs, transpose_axpby, transpose_rows
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


@wp.kernel(enable_backward=False)
def _statistics(iterations: wp.array[int], residual: wp.array[Any], tolerance: wp.array[Any], stats: wp.array[int]):
    rhs = wp.tid()
    wp.atomic_add(stats, 0, iterations[0])
    wp.atomic_max(stats, 1, iterations[0])
    failed = not wp.isfinite(residual[rhs]) or not wp.isfinite(tolerance[rhs]) or residual[rhs] > tolerance[rhs]
    wp.atomic_add(stats, 2, int(failed))
    wp.atomic_add(stats, 3, 1)


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
    it. All uses of shared scratch must be serialized, including across streams
    and captured graphs. Value updates in existing buffers are observed on replay;
    topology/address changes require recapture. No adjoint is supplied.

    Args:
        A: Square sparse matrix or a single-system :class:`LinearOperator`.
        nrhs: Positive RHS count.
        layout: ``"columns"`` enables coalesced sparse CUDA products. ``"rows"``
            retains RHS-row storage; compare both for narrow batches.
        dtype: Scalar arithmetic type for a custom operator; defaults to A's
            scalar type. Sparse matrices must already have this scalar type.
    """

    def __init__(self, A, nrhs: int, *, layout: str = "columns", dtype: type | None = None):
        if isinstance(nrhs, bool) or index(nrhs) < 1:
            raise ValueError("nrhs must be a positive integer")
        self.source = A
        self.single = aslinearoperator(A)
        self.n, self.nrhs = self.single.shape[0], index(nrhs)
        if (
            self.n < 1
            or self.single.shape not in ((self.n,), (self.n, self.n))
            or self.single.batch_offsets is not None
        ):
            raise ValueError("Expected one nonempty square system")
        if layout not in ("rows", "columns"):
            raise ValueError("layout must be 'rows' or 'columns'")
        dtype = self.single.scalar_type if dtype is None else dtype
        if dtype not in (wp.float32, wp.float64):
            raise ValueError("Working dtype must be float32 or float64")
        self.sparse = isinstance(A, sp.BsrMatrix)
        if self.sparse and (
            A.row_counts is not None or A.scalar_type != dtype or A.block_shape not in ((1, 1), (3, 3))
        ):
            raise ValueError("Sparse inputs must be compressed scalar or 3x3 BSR in the working dtype")
        self.layout = layout if self.single.device.is_cuda and self.sparse else "rows"
        shape = (self.n, self.nrhs) if self.layout == "columns" else (self.nrhs, self.n)
        self._first = wp.empty(shape, dtype=dtype, device=self.single.device)
        self._second = wp.empty_like(self._first) if self.layout == "columns" else None
        offsets = wp.empty(self.nrhs + 1, dtype=int, device=self.single.device)
        wp.launch(_offsets, self.nrhs + 1, [offsets, self.n], device=self.single.device)
        super().__init__(
            (self.n * self.nrhs, self.n * self.nrhs),
            dtype,
            self.single.device,
            self._apply,
            batch_offsets=offsets,
            max_batch_length=self.n,
        )

    def with_nrhs(self, nrhs: int):
        """Prepare a smaller capacity sharing scratch; execution must be serialized."""
        if isinstance(nrhs, bool) or not 1 <= index(nrhs) <= self.nrhs:
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


class CGSolve:
    """Prepared independent multiple-RHS CG solves sharing one sparse SPD matrix.

    This wraps :func:`cg`; it is not a coupled block-CG recurrence. Inputs/outputs
    use the source scalar dtype; inner arithmetic may explicitly use float32.
    Every call starts at zero and refreshes any working-precision matrix copy on
    device. Truncated or tolerance-stopped CG is not a fixed linear inverse.
    Matrix topology and addresses must stay fixed during replay; factors remain
    caller-owned and updates are explicit. Construction/warmup precede capture.
    No host convergence readbacks or work-array allocations occur during replay.

    ``stats`` accumulates scheduled vector iterations, maximum batch iterations,
    failed recurrence residual checks, and RHS calls. Masked converged vectors
    still contribute scheduled iterations; these are not individual convergence
    iteration counts. Recurrence diagnostics use working precision and do not
    certify source-precision residuals for an approximate solve.

    Args:
        A: Compressed square SPD scalar or 3x3 BSR matrix, float32 or float64.
        nrhs: Positive number of independent right-hand sides.
        M: Fixed SPD preconditioner, either a single-system linear operator or a
            prepared multiple-RHS operator with equal length n partitions. The
            latter retains optimized application, such as coalesced FSAI.
            Preconditioner metadata/arithmetic must use ``working_dtype``.
        tol: Nonnegative relative per-RHS tolerance.
        atol: Nonnegative absolute per-RHS tolerance.
        maxiter: Positive CG iteration budget; truncation is allowed.
        working_dtype: float32 or float64; defaults to A's scalar type.
        layout: Sparse application layout, ``"columns"`` or ``"rows"``.
    """

    def __init__(
        self,
        A,
        nrhs: int,
        *,
        M=None,
        tol: float = 1e-12,
        atol: float = 0.0,
        maxiter: int = 1000,
        working_dtype: type | None = None,
        layout: str = "columns",
    ):
        if isinstance(nrhs, bool) or index(nrhs) < 1 or isinstance(maxiter, bool) or index(maxiter) < 1:
            raise ValueError("nrhs and maxiter must be positive integers")
        if not math.isfinite(tol) or tol < 0 or not math.isfinite(atol) or atol < 0 or max(tol, atol) == 0:
            raise ValueError("Tolerances must be finite, nonnegative and not both zero")
        if not isinstance(A, sp.BsrMatrix) or A.shape[0] != A.shape[1] or A.shape[0] < 1:
            raise ValueError("A must be a nonempty square Warp sparse matrix")
        if (
            A.scalar_type not in (wp.float32, wp.float64)
            or A.block_shape not in ((1, 1), (3, 3))
            or A.row_counts is not None
        ):
            raise ValueError("A must be compressed scalar or 3x3 BSR in float32 or float64")
        self.source, self.n, self.nrhs, self.device, self.dtype = A, A.shape[0], index(nrhs), A.device, A.scalar_type
        self.working_dtype = A.scalar_type if working_dtype is None else working_dtype
        if self.working_dtype not in (wp.float32, wp.float64):
            raise ValueError("working_dtype must be float32 or float64")
        self.tol, self.atol, self.maxiter, self.layout = tol, atol, index(maxiter), layout
        self.working_matrix = A if self.working_dtype == self.dtype else sp.bsr_copy(A, scalar_type=self.working_dtype)
        self.operator = MultiRHSOperator(self.working_matrix, self.nrhs, layout=layout)
        self._b = wp.zeros(self.n * self.nrhs, dtype=self.working_dtype, device=self.device)
        self._x = wp.zeros_like(self._b)
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
                if M.shape not in ((self.n,), (self.n, self.n)):
                    raise ValueError("Single-system M must match the matrix dimension")
                M = MultiRHSOperator(M, self.nrhs, dtype=self.working_dtype)
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
            run=False,
        )
        self.stats = wp.zeros(4, dtype=int, device=self.device)
        """Cumulative scheduled iterations, maximum batch iterations, failures, RHSs."""

    def with_nrhs(self, nrhs: int, *, M=None):
        """Prepare a smaller solve sharing matrix and staging; use serially.

        Each capacity has independent Krylov workspaces/counters. Supply a matching
        prepared M when the original preconditioner was itself multiple-RHS.
        Otherwise the original single-system preconditioner is reused by default.
        """
        if isinstance(nrhs, bool) or not 1 <= index(nrhs) <= self.nrhs:
            raise ValueError("nrhs must be positive and no greater than the prepared capacity")
        if M is None:
            M = self._M
            if M is not None and aslinearoperator(M).batch_offsets is not None:
                raise ValueError("Supply a matching prepared M for the smaller capacity")
        child = copy.copy(self)
        child.nrhs = index(nrhs)
        child.operator = self.operator.with_nrhs(child.nrhs)
        child._b = self._b[: self.n * child.nrhs]
        child._x = self._x[: self.n * child.nrhs]
        child._prepare_state(M)
        return child

    def reset_stats(self):
        """Clear cumulative counters on device, including during capture."""
        self.stats.zero_()

    def solve(self, rhs: wp.array, out: wp.array):
        """Solve from zero into matching contiguous ``(nrhs,n)`` arrays.

        Input and output use the source matrix dtype/device. In-place and partial
        aliases are safe because all RHSs are staged before any output is written.
        The state and shared capacities must not execute concurrently.
        """
        if rhs.shape != (self.nrhs, self.n) or out.shape != rhs.shape:
            raise ValueError("Expected matching arrays of shape (nrhs,n)")
        for a in (rhs, out):
            if a.device != self.device or a.dtype != self.dtype or not a.is_contiguous:
                raise ValueError("Expected contiguous arrays with the source matrix dtype and device")
        if self.working_dtype != self.dtype:
            source, target = self.source.values, self.working_matrix.values
            source_flat = source.view(self.dtype).flatten()
            target_flat = target.view(self.working_dtype).flatten()
            wp.launch(_cast, min(source_flat.size, target_flat.size), [source_flat, target_flat], device=self.device)
            wp.launch(_cast, self._b.size, [rhs.flatten(), self._b], device=self.device)
        else:
            wp.copy(self._b, rhs.flatten())
        self._x.zero_()
        iterations, residual, tolerance = self.state()
        if self.working_dtype != self.dtype:
            wp.launch(_cast, self._x.size, [self._x, out.flatten()], device=self.device)
        else:
            wp.copy(out.flatten(), self._x)
        wp.launch(_statistics, self.nrhs, [iterations, residual, tolerance, self.stats], device=self.device)
        return iterations, residual, tolerance
