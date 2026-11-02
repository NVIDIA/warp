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

"""Reusable solve actions built from the existing linear solver states."""

import math
from operator import index
from typing import Any

import warp as wp
from warp._src.optim.linear import aslinearoperator, gmres


@wp.kernel(enable_backward=False)
def _residual_norms(ax: wp.array[Any], b: wp.array[Any], norms: wp.array[Any]):
    block = wp.tid()
    av = wp.tile_load(ax, shape=256, offset=block * 256)
    bv = wp.tile_load(b, shape=256, offset=block * 256)
    r = av - bv
    wp.tile_atomic_add(norms, wp.tile_sum(wp.tile_map(wp.mul, r, r)), offset=0)
    wp.tile_atomic_add(norms, wp.tile_sum(wp.tile_map(wp.mul, bv, bv)), offset=1)


@wp.kernel(enable_backward=False)
def _record(
    iterations: wp.array[int],
    norms: wp.array[Any],
    tolerance_sq: wp.array[Any],
    row: int,
    counts: wp.array[int],
    absolute: wp.array[Any],
    relative: wp.array[Any],
    tolerance: wp.array[Any],
    converged: wp.array[int],
    stats: wp.array[int],
    maxima: wp.array[Any],
):
    a = wp.sqrt(norms[0])
    r = norms.dtype(0)
    if norms[1] > norms.dtype(0):
        r = a / wp.sqrt(norms[1])
    elif a > norms.dtype(0):
        r = norms.dtype(wp.inf)
    valid = wp.isfinite(norms[0]) and wp.isfinite(norms[1]) and wp.isfinite(tolerance_sq[0])
    success = valid and norms[0] <= tolerance_sq[0]
    if not valid:
        a = norms.dtype(wp.inf)
        r = norms.dtype(wp.inf)
    counts[row] = iterations[0]
    absolute[row] = a
    relative[row] = r
    tolerance[row] = wp.sqrt(tolerance_sq[0])
    converged[row] = int(success)
    stats[0] += iterations[0]
    stats[1] = wp.max(stats[1], iterations[0])
    stats[2] += int(not success)
    stats[3] += 1
    maxima[0] = wp.max(maxima[0], a)
    maxima[1] = wp.max(maxima[1], r)


class GMRESSolve:
    """Prepared zero-start GMRES action with fresh original-equation diagnostics.

    Bind a square real operator and call :meth:`solve` with new RHS/output arrays.
    Existing :func:`gmres` supplies the recurrence and device stopping decisions.
    Multiple RHSs are solved serially with a shared workspace; this is not block
    GMRES. No matrix is constructed. A fixed linear right preconditioner can be
    supplied; a variable iterative action requires a flexible algorithm instead.

    Each solve resets the iterate to zero. Per-RHS diagnostics are overwritten on
    every call; ``stats`` and ``residual_maxima`` accumulate until ``reset_stats``.
    Fresh acceptance uses ``||A*x-b||`` and the effective upstream tolerance,
    including its arithmetic safety floor. It never trusts only the recurrence's
    estimate. Restart boundaries may exceed ``maxiter`` by up to ``restart-1``;
    ``iterations`` reports actual work. Zero RHSs have relative residual zero
    only for a zero residual, otherwise infinity; nonfinite norms fail acceptance.

    Prepare and warm up outside capture. With caller-provided graph-compatible
    operators, CUDA replay has no allocations, host convergence checks or Python
    numerical callbacks. Coefficients in existing buffers may change on replay;
    topology/address changes require recapture and preconditioner updates remain
    explicit. This state owns mutable scratch. Eager calls issued from different
    CUDA streams are ordered with an event, but simultaneously executing graphs
    that share this state require caller ordering.
    Reverse-mode differentiation through this action is not supported.

    Args:
        A: Square real sparse matrix or :class:`LinearOperator`, float32 or float64.
        nrhs: Positive number of independent RHSs prepared for :meth:`solve`.
        M: Optional fixed linear right preconditioner, with matching size/device.
        tol: Nonnegative relative tolerance.
        atol: Nonnegative absolute tolerance. At least one tolerance must be positive.
        maxiter: Positive iteration budget.
        restart: Positive Krylov restart dimension, clipped to the matrix size.
    """

    def __init__(
        self, A, nrhs: int = 1, *, M=None, tol: float = 1e-12, atol: float = 0.0, maxiter: int = 1000, restart: int = 64
    ):
        for name, value in (("nrhs", nrhs), ("maxiter", maxiter), ("restart", restart)):
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, not bool")
            if index(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("tol", tol), ("atol", atol)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if max(tol, atol) == 0:
            raise ValueError("At least one tolerance must be positive")
        self.operator = aslinearoperator(A)
        self.n = self.operator.shape[0]
        if self.operator.shape not in ((self.n,), (self.n, self.n)) or self.n < 1:
            raise ValueError("A must be nonempty and square")
        if self.operator.batch_offsets is not None:
            raise ValueError("A must describe one shared system, not partitioned systems")
        self.device, self.dtype, self.nrhs = self.operator.device, self.operator.scalar_type, index(nrhs)
        if self.dtype not in (wp.float32, wp.float64):
            raise ValueError("A must have float32 or float64 scalar type")
        M = aslinearoperator(M)
        if M is not None and (
            M.shape not in ((self.n,), (self.n, self.n))
            or M.device != self.device
            or M.scalar_type != self.dtype
            or M.batch_offsets is not None
        ):
            raise ValueError("M must match the system size, dtype and device")
        self._b = wp.zeros(self.n, dtype=self.dtype, device=self.device)
        self._x = wp.zeros_like(self._b)
        self._ax = wp.empty_like(self._b)
        self._norms = wp.zeros(2, dtype=self.dtype, device=self.device)
        self.stats = wp.zeros(4, dtype=int, device=self.device)
        """Cumulative total iterations, maximum per RHS, failures, and RHS calls."""
        self.residual_maxima = wp.zeros(2, dtype=self.dtype, device=self.device)
        """Maximum absolute/relative fresh residual since the last reset."""
        self.iterations = wp.zeros(self.nrhs, dtype=int, device=self.device)
        """Actual iterations per RHS from the latest call."""
        self.absolute_residual = wp.zeros(self.nrhs, dtype=self.dtype, device=self.device)
        """Fresh original-equation residual norms from the latest call."""
        self.relative_residual = wp.zeros_like(self.absolute_residual)
        """Fresh relative residual norms from the latest call."""
        self.tolerance = wp.zeros_like(self.absolute_residual)
        """Effective absolute acceptance tolerances from the latest call."""
        self.converged = wp.zeros(self.nrhs, dtype=int, device=self.device)
        """Per-RHS acceptance flags (one for success, zero for failure)."""
        self._event = wp.Event(device=self.device) if self.device.is_cuda else None
        # Seed ordering with the constructing stream so first use elsewhere waits for setup.
        self._last_stream = self.device.stream if self.device.is_cuda else None
        self.state = gmres(
            self.operator,
            self._b,
            self._x,
            M=M,
            tol=tol,
            atol=atol,
            restart=min(index(restart), self.n),
            maxiter=index(maxiter),
            check_every=0,
            run=False,
        )

    def reset_stats(self):
        """Reset cumulative counters/maxima on device; preserve last-call diagnostics."""
        self.stats.zero_()
        self.residual_maxima.zero_()

    def solve(self, rhs: wp.array, out: wp.array):
        """Solve from zero into caller-owned storage and refresh device diagnostics.

        Args:
            rhs: Contiguous scalar array of shape ``(nrhs,n)``. For one RHS,
                ``(n,)`` is also accepted. Dtype/device must match the system.
            out: Matching output. Exact in-place aliasing is supported, but
                shifted overlapping views are rejected before any output mutation.

        Returns:
            None. Results and diagnostics remain on device.
        """
        shapes = ((self.nrhs, self.n), (self.n,)) if self.nrhs == 1 else ((self.nrhs, self.n),)
        if rhs.shape not in shapes or out.shape != rhs.shape:
            raise ValueError("Expected matching arrays of the prepared RHS shape")
        for a in (rhs, out):
            if a.dtype != self.dtype or a.device != self.device or not a.is_contiguous:
                raise ValueError("RHS/output must be contiguous with the system dtype and device")
        size = rhs.size * wp.types.type_size_in_bytes(rhs.dtype)
        if rhs.ptr != out.ptr and max(rhs.ptr, out.ptr) < min(rhs.ptr + size, out.ptr + size):
            raise ValueError("Partially overlapping RHS/output views are unsupported")
        rows, result = rhs.reshape((self.nrhs, self.n)), out.reshape((self.nrhs, self.n))
        # Outside capture, order this call after the previous use of the shared scratch.
        stream = self.device.stream if self.device.is_cuda else None
        if stream is not None and not stream.is_capturing and stream is not self._last_stream:
            if self._last_stream is not None and not self._last_stream.is_capturing:
                self._last_stream.record_event(self._event)
                stream.wait_event(self._event)
            self._last_stream = stream
        for j in range(self.nrhs):
            wp.copy(self._b, rows[j])
            self._x.zero_()
            iterations, _, tolerance_sq = self.state()
            self.operator.matvec(self._x, self._ax, self._ax, 1.0, 0.0)
            self._norms.zero_()
            wp.launch_tiled(
                _residual_norms,
                dim=(self.n + 255) // 256,
                inputs=[self._ax, self._b, self._norms],
                block_dim=64,
                device=self.device,
            )
            wp.launch(
                _record,
                1,
                [
                    iterations,
                    self._norms,
                    tolerance_sq,
                    j,
                    self.iterations,
                    self.absolute_residual,
                    self.relative_residual,
                    self.tolerance,
                    self.converged,
                    self.stats,
                    self.residual_maxima,
                ],
                device=self.device,
            )
            wp.copy(result[j], self._x)
