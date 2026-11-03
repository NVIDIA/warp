# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared linear solve actions sharing one usage and diagnostics contract."""

import copy
import enum
import math
from operator import index
from typing import Any

import warp as wp
from warp._src.optim.linear import _get_dtype_epsilon, aslinearoperator, gmres


class ResidualMode(enum.IntEnum):
    """Residual that a prepared solve action reports and uses for acceptance."""

    RECURRENCE = 0
    """The iterative solver's own residual estimate, in working precision. Costs no extra operator application."""

    FRESH = 1
    """A recomputed ``||A x - b||`` using the original operator and precision. Costs one operator application per solve."""


class _StreamOrder:
    """Order eager uses of shared scratch issued from different CUDA streams."""

    def __init__(self, device):
        self.event = wp.Event(device=device) if device.is_cuda else None
        # Seed with the constructing stream so first use elsewhere waits for setup.
        self.last = device.stream if device.is_cuda else None

    def acquire(self, device):
        if self.event is None:
            return
        stream = device.stream
        # Same-stream calls, the common case, skip the native capture query.
        if stream is self.last or stream.is_capturing:
            return
        if self.last is not None and not self.last.is_capturing:
            self.last.record_event(self.event)
            stream.wait_event(self.event)
        self.last = stream


@wp.kernel(enable_backward=False)
def _squared_norms(
    ax: wp.array2d[Any],
    b: wp.array2d[Any],
    with_residual: int,
    first: int,
    residual_sq: wp.array[Any],
    rhs_sq: wp.array[Any],
):
    rhs, block = wp.tid()
    bv = wp.tile_load(b[rhs], shape=256, offset=block * 256)
    if with_residual != 0:
        r = wp.tile_load(ax[rhs], shape=256, offset=block * 256) - bv
        wp.tile_atomic_add(residual_sq, wp.tile_sum(wp.tile_map(wp.mul, r, r)), offset=first + rhs)
    wp.tile_atomic_add(rhs_sq, wp.tile_sum(wp.tile_map(wp.mul, bv, bv)), offset=first + rhs)


@wp.kernel(enable_backward=False)
def _record(
    iterations: wp.array[int],
    residual_sq: wp.array[Any],
    rhs_sq: wp.array[Any],
    tolerance_sq: wp.array[Any],
    first: int,
    counts: wp.array[int],
    absolute: wp.array[Any],
    relative: wp.array[Any],
    tolerance: wp.array[Any],
    converged: wp.array[int],
    stats: wp.array[int],
    maxima: wp.array[Any],
):
    k = wp.tid()
    row = first + k
    r_sq = absolute.dtype(residual_sq[k])
    b_sq = absolute.dtype(rhs_sq[k])
    t_sq = absolute.dtype(tolerance_sq[wp.min(k, tolerance_sq.shape[0] - 1)])
    a = wp.sqrt(r_sq)
    r = absolute.dtype(0)
    if b_sq > absolute.dtype(0):
        r = a / wp.sqrt(b_sq)
    elif a > absolute.dtype(0):
        r = absolute.dtype(wp.inf)
    valid = wp.isfinite(r_sq) and wp.isfinite(b_sq) and wp.isfinite(t_sq)
    success = valid and r_sq <= t_sq
    if not valid:
        a = absolute.dtype(wp.inf)
        r = absolute.dtype(wp.inf)
    counts[row] = iterations[0]
    absolute[row] = a
    relative[row] = r
    tolerance[row] = wp.sqrt(t_sq)
    converged[row] = int(success)
    wp.atomic_add(stats, 0, iterations[0])
    wp.atomic_max(stats, 1, iterations[0])
    wp.atomic_add(stats, 2, int(not success))
    wp.atomic_add(stats, 3, 1)
    wp.atomic_max(maxima, 0, a)
    wp.atomic_max(maxima, 1, r)


def _positive_int(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if index(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return index(value)


def _resolve_tolerances(tol, atol, dtype):
    """Validate tolerances and give an omitted relative tolerance a ``dtype`` default."""
    for name, value in (("tol", tol), ("atol", atol)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if tol is None:
        # The iterative solvers' default; reachable in the working precision.
        tol = _get_dtype_epsilon(dtype) ** 0.75
    if max(tol, atol) == 0:
        raise ValueError("At least one tolerance must be positive")
    return tol, atol


class _SolveAction:
    """Shared validation, stream ordering and device diagnostics for prepared solves."""

    def _init_action(self, n, nrhs, dtype, device, residual):
        if not isinstance(residual, ResidualMode):
            raise TypeError("residual must be a ResidualMode")
        self.n, self.nrhs, self.dtype, self.device = n, _positive_int("nrhs", nrhs), dtype, device
        self.residual = residual
        self._stream_order = _StreamOrder(device)
        self._allocate_diagnostics()

    def _allocate_diagnostics(self):
        self.stats = wp.zeros(4, dtype=int, device=self.device)
        self.residual_maxima = wp.zeros(2, dtype=self.dtype, device=self.device)
        self.iterations = wp.zeros(self.nrhs, dtype=int, device=self.device)
        self.absolute_residual = wp.zeros(self.nrhs, dtype=self.dtype, device=self.device)
        self.relative_residual = wp.zeros_like(self.absolute_residual)
        self.tolerance = wp.zeros_like(self.absolute_residual)
        self.converged = wp.zeros(self.nrhs, dtype=int, device=self.device)
        self._residual_sq = wp.zeros_like(self.absolute_residual)
        self._rhs_sq = wp.zeros_like(self.absolute_residual)

    def _child(self, nrhs):
        """Copy this action for a smaller capacity with its own diagnostics."""
        nrhs = _positive_int("nrhs", nrhs)
        if nrhs > self.nrhs:
            raise ValueError("nrhs must not exceed the prepared capacity")
        child = copy.copy(self)
        child.nrhs = nrhs
        # The child shares stream ordering, so its zeroed diagnostics follow earlier work.
        self._stream_order.acquire(self.device)
        child._allocate_diagnostics()
        return child

    def update(self):
        """Make value changes to the system's existing buffers visible to later solves.

        Call after changing matrix or operator values in place. Actions that read
        current values on every solve treat this as a no-op, so solver-agnostic
        callers can always call it. It is graph capturable.
        """

    def reset_stats(self):
        """Clear cumulative ``stats`` and ``residual_maxima`` on device.

        The latest per-RHS diagnostics are kept. Can be captured in a graph.
        """
        self._stream_order.acquire(self.device)
        self.stats.zero_()
        self.residual_maxima.zero_()

    def _prepare_solve(self, rhs, out):
        shapes = ((self.nrhs, self.n), (self.n,)) if self.nrhs == 1 else ((self.nrhs, self.n),)
        if rhs.shape not in shapes or out.shape != rhs.shape:
            raise ValueError("Expected matching arrays of shape (nrhs, n), or (n,) for one RHS")
        for a in (rhs, out):
            if a.dtype != self.dtype or a.device != self.device or not a.is_contiguous:
                raise ValueError("RHS/output must be contiguous with the system dtype and device")
        size = rhs.size * wp.types.type_size_in_bytes(rhs.dtype)
        if rhs.ptr != out.ptr and max(rhs.ptr, out.ptr) < min(rhs.ptr + size, out.ptr + size):
            raise ValueError("Partially overlapping RHS/output views are unsupported")
        self._stream_order.acquire(self.device)
        self._residual_sq.zero_()
        self._rhs_sq.zero_()
        return rhs.reshape((self.nrhs, self.n)), out.reshape((self.nrhs, self.n))

    def _squared_norms(self, ax, b, first):
        """Accumulate squared ``||ax - b||`` (unless ``ax`` is None) and ``||b||`` per row."""
        wp.launch_tiled(
            _squared_norms,
            dim=(b.shape[0], (self.n + 255) // 256),
            inputs=[b if ax is None else ax, b, int(ax is not None), first, self._residual_sq, self._rhs_sq],
            block_dim=64,
            device=self.device,
        )

    def _record(self, iterations, residual_sq, tolerance_sq, first, count):
        wp.launch(
            _record,
            count,
            [
                iterations,
                residual_sq,
                self._rhs_sq[first : first + count],
                tolerance_sq,
                first,
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


class GMRESSolve(_SolveAction):
    """Prepared zero-start GMRES action with device-resident per-RHS diagnostics.

    Bind a square real operator and call :meth:`solve` with new RHS/output arrays.
    Existing :func:`gmres` supplies the recurrence and device stopping decisions.
    Multiple RHSs are solved serially with a shared workspace; this is not block
    GMRES. A fixed linear right preconditioner can be supplied; a variable
    iterative action requires a flexible algorithm instead.

    Each solve starts from zero. Acceptance compares the selected residual with
    the effective upstream tolerance, including its arithmetic safety floor.
    Restart boundaries may exceed ``maxiter`` by up to ``restart - 1``;
    ``iterations`` reports the work actually done.

    Prepare and warm up outside capture. With graph-compatible operators, CUDA
    replay has no allocations or host convergence checks. Coefficients in
    existing buffers may change on replay; topology/address changes require
    recapture and preconditioner updates remain explicit. Eager calls issued
    from different CUDA streams are ordered with an event, but simultaneously
    executing graphs that share this state or its smaller capacities require
    caller ordering. Reverse-mode differentiation is not supported.

    Args:
        A: Square real sparse matrix or :class:`LinearOperator`, float32 or float64.
        nrhs: Positive number of RHSs prepared for :meth:`solve`.
        M: Optional fixed linear right preconditioner, with matching size/device.
        tol: Nonnegative relative tolerance. Defaults to ``eps ** 0.75`` for the
            operator's scalar type: 1e-6 for float32 and 1e-12 for float64.
        atol: Nonnegative absolute tolerance. At least one tolerance must be positive.
        maxiter: Positive iteration budget.
        restart: Positive Krylov restart dimension, clipped to the matrix size.
        residual: Residual reported and used for acceptance.
            :attr:`ResidualMode.FRESH` recomputes ``||A x - b||`` after each RHS.

    Attributes:
        iterations: Device array of iterations per RHS from the latest call.
        absolute_residual: Device array of residual norms from the latest call.
            Nonfinite norms are reported as infinity.
        relative_residual: Device array of residual norms relative to ``||b||``.
            A zero RHS gives zero for a zero residual and infinity otherwise.
        tolerance: Device array of effective absolute tolerances from the latest call.
        converged: Device array of per-RHS acceptance flags, one for success.
        stats: Device array of cumulative total iterations, maximum iterations,
            failed RHSs and solved RHSs since the last ``reset_stats()``.
        residual_maxima: Device array of the maximum absolute and relative
            residuals since the last ``reset_stats()``.
    """

    def __init__(
        self,
        A,
        nrhs: int = 1,
        *,
        M=None,
        tol: float | None = None,
        atol: float = 0.0,
        maxiter: int = 1000,
        restart: int = 64,
        residual: ResidualMode = ResidualMode.FRESH,
    ):
        maxiter, restart = _positive_int("maxiter", maxiter), _positive_int("restart", restart)
        self.operator = aslinearoperator(A)
        n = self.operator.shape[0]
        if self.operator.shape not in ((n,), (n, n)) or n < 1:
            raise ValueError("A must be nonempty and square")
        if self.operator.batch_offsets is not None:
            raise ValueError("A must describe one shared system, not partitioned systems")
        if self.operator.scalar_type not in (wp.float32, wp.float64):
            raise ValueError("A must have float32 or float64 scalar type")
        tol, atol = _resolve_tolerances(tol, atol, self.operator.scalar_type)
        self._init_action(n, nrhs, self.operator.scalar_type, self.operator.device, residual)
        M = aslinearoperator(M)
        if M is not None and (
            M.shape not in ((n,), (n, n))
            or M.device != self.device
            or M.scalar_type != self.dtype
            or M.batch_offsets is not None
        ):
            raise ValueError("M must match the system size, dtype and device")
        self._b = wp.zeros(n, dtype=self.dtype, device=self.device)
        self._x = wp.zeros_like(self._b)
        self._ax = wp.empty_like(self._b)
        self.state = gmres(
            self.operator,
            self._b,
            self._x,
            M=M,
            tol=tol,
            atol=atol,
            restart=min(restart, n),
            maxiter=maxiter,
            check_every=0,
            run=False,
        )

    def with_nrhs(self, nrhs: int) -> "GMRESSolve":
        """Prepare a smaller capacity sharing this action's workspace.

        The returned action owns its diagnostics but shares the Krylov workspace
        and stream ordering, so it must not run concurrently with this action.

        Args:
            nrhs: Positive capacity no greater than this action's ``nrhs``.

        Returns:
            A :class:`GMRESSolve` that solves ``nrhs`` right-hand sides.
        """
        return self._child(nrhs)

    def solve(self, rhs: wp.array, out: wp.array):
        """Solve from zero into caller-owned storage and refresh device diagnostics.

        Args:
            rhs: Contiguous array of shape ``(nrhs, n)``, or ``(n,)`` for one RHS,
                with the system dtype and device.
            out: Matching output. Exact in-place aliasing is supported, but
                shifted overlapping views are rejected before any output mutation.
        """
        rows, result = self._prepare_solve(rhs, out)
        for j in range(self.nrhs):
            wp.copy(self._b, rows[j])
            self._x.zero_()
            iterations, residual_sq, tolerance_sq = self.state()
            b = self._b.reshape((1, self.n))
            if self.residual == ResidualMode.FRESH:
                self.operator.matvec(self._x, self._ax, self._ax, 1.0, 0.0)
                self._squared_norms(self._ax.reshape((1, self.n)), b, j)
                residual_sq = self._residual_sq[j : j + 1]
            else:
                self._squared_norms(None, b, j)
            self._record(iterations, residual_sq, tolerance_sq, j, 1)
            wp.copy(result[j], self._x)
