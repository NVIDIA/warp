# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared linear solve actions sharing one usage and diagnostics contract."""

import copy
import math
from operator import index
from typing import Any, Literal

import warp as wp
from warp._src import sparse as sp
from warp._src.context import is_conditional_graph_supported
from warp._src.optim._multi_rhs import MultiRHSOperator, _scalar_shape, _StreamOrder
from warp._src.optim.linear import _get_dtype_epsilon, aslinearoperator, bicgstab, cg, cr, gmres
from warp._src.sparse_multi_rhs import validate as _validate_algorithm
from warp._src.types import type_to_warp

RESIDUAL_MODES = ("recurrence", "fresh")
"""Residuals a prepared solve action can report and use for acceptance."""

KRYLOV_METHODS = ("cg", "cr", "bicgstab", "gmres")
"""Iterative methods that :class:`KrylovSolve` can run."""


def _validate_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"Unsupported {name}: {value!r}; expected one of {choices}")


@wp.kernel(enable_backward=False)
def _squared_norms(
    ax: wp.array2d[Any],
    b: wp.array2d[Any],
    first: int,
    residual_sq: wp.array[Any],
    rhs_sq: wp.array[Any],
):
    rhs, block = wp.tid()
    # Square in the accumulator dtype so small entries of a narrower input cannot underflow.
    bv = wp.tile_astype(wp.tile_load(b[rhs], shape=256, offset=block * 256), rhs_sq.dtype)
    r = wp.tile_astype(wp.tile_load(ax[rhs], shape=256, offset=block * 256), rhs_sq.dtype) - bv
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
    # Compare and take roots in double precision; squares can underflow in the reported dtype.
    r_sq = wp.float64(residual_sq[k])
    b_sq = wp.float64(rhs_sq[k])
    t_sq = wp.float64(tolerance_sq[wp.min(k, tolerance_sq.shape[0] - 1)])
    zero = wp.float64(0.0)
    a = wp.sqrt(r_sq)
    r = zero
    if b_sq > zero:
        r = a / wp.sqrt(b_sq)
    elif a > zero:
        r = wp.float64(wp.inf)
    valid = wp.isfinite(r_sq) and wp.isfinite(b_sq) and wp.isfinite(t_sq)
    success = valid and r_sq <= t_sq
    if not valid:
        a = wp.float64(wp.inf)
        r = wp.float64(wp.inf)
    counts[row] = iterations[0]
    absolute[row] = absolute.dtype(a)
    relative[row] = absolute.dtype(r)
    tolerance[row] = absolute.dtype(wp.sqrt(t_sq))
    converged[row] = int(success)
    wp.atomic_add(stats, 0, iterations[0])
    wp.atomic_max(stats, 1, iterations[0])
    wp.atomic_add(stats, 2, int(not success))
    wp.atomic_add(stats, 3, 1)
    wp.atomic_max(maxima, 0, absolute.dtype(a))
    wp.atomic_max(maxima, 1, absolute.dtype(r))


def _positive_int(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if index(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return index(value)


def _resolve_tolerances(tol, atol, dtype):
    """Validate tolerances and give an omitted relative tolerance a ``dtype`` default."""
    if atol is None:
        # Upstream solvers read None as "same as tol"; require an explicit value instead of guessing.
        raise TypeError("atol must be a nonnegative number; use 0.0 for no absolute tolerance")
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

    def _init_action(self, n, nrhs, dtype, device, residual, norm_dtype=None):
        _validate_choice("residual mode", residual, RESIDUAL_MODES)
        self.n, self.nrhs, self.dtype, self.device = n, _positive_int("nrhs", nrhs), dtype, device
        # Squared norms accumulate in the wider of the reported and working dtypes.
        self._norm_dtype = dtype if norm_dtype is None else norm_dtype
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
        self._residual_sq = wp.zeros(self.nrhs, dtype=self._norm_dtype, device=self.device)
        self._rhs_sq = wp.zeros_like(self._residual_sq)

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

    def _prepare_solve(self, rhs, out, x0=None):
        shapes = ((self.nrhs, self.n), (self.n,)) if self.nrhs == 1 else ((self.nrhs, self.n),)
        if rhs.shape not in shapes or out.shape != rhs.shape or (x0 is not None and x0.shape != rhs.shape):
            raise ValueError("Expected matching arrays of shape (nrhs, n), or (n,) for one RHS")
        for a in (rhs, out) if x0 is None else (rhs, out, x0):
            if a.dtype != self.dtype or a.device != self.device or not a.is_contiguous:
                raise ValueError("RHS/output/initial guess must be contiguous with the system dtype and device")
        size = rhs.size * wp.types.type_size_in_bytes(rhs.dtype)
        if rhs.ptr != out.ptr and max(rhs.ptr, out.ptr) < min(rhs.ptr + size, out.ptr + size):
            raise ValueError("Partially overlapping RHS/output views are unsupported")
        self._stream_order.acquire(self.device)
        return rhs.reshape((self.nrhs, self.n)), out.reshape((self.nrhs, self.n))

    def _squared_norms(self, ax, b, first):
        """Accumulate squared ``||ax - b||`` and ``||b||`` per row.

        The caller zeroes the accumulators first.
        """
        wp.launch_tiled(
            _squared_norms,
            dim=(b.shape[0], (self.n + 255) // 256),
            inputs=[ax, b, first, self._residual_sq, self._rhs_sq],
            block_dim=64,
            device=self.device,
        )

    def _record(self, iterations, residual_sq, rhs_sq, tolerance_sq, first, count):
        wp.launch(
            _record,
            count,
            [
                iterations,
                residual_sq,
                rhs_sq,
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


_METHODS = {"cg": cg, "cr": cr, "bicgstab": bicgstab, "gmres": gmres}


@wp.kernel(enable_backward=False)
def _cast(x: wp.array[Any], y: wp.array[Any]):
    i = wp.tid()
    y[i] = y.dtype(x[i])


def _working_copy(A, scalar_type):
    """Return a matrix that shares ``A``'s topology arrays, with values of another scalar type.

    Sharing the topology keeps every value slot aligned with the source, so ``update()`` can
    copy values slot by slot whatever the storage layout or the column order within rows.
    """
    if A.values.dtype == A.scalar_type:
        block_type = scalar_type
    else:
        block_type = wp.types.matrix(shape=A.block_shape, dtype=scalar_type)
    copy = sp.bsr_matrix_t(block_type)()
    copy.nrow, copy.ncol, copy.nnz = A.nrow, A.ncol, A.nnz
    copy.offsets, copy.columns, copy.row_counts = A.offsets, A.columns, A.row_counts
    copy.values = wp.empty(A.values.shape, dtype=block_type, device=A.values.device)
    return copy


class KrylovSolve(_SolveAction):
    """Prepared Krylov solves of one system for several right-hand sides.

    Bind a square real system once, then call :meth:`solve` with new RHS/output
    arrays. The existing functional solver selected by ``method`` supplies the
    recurrence and device stopping decisions, so every option of :func:`cg`,
    :func:`cr`, :func:`bicgstab` and :func:`gmres` behaves as documented there.
    Several RHSs iterate together as independent systems with a shared operator,
    so each reports the batch's scheduled iteration count, including iterations
    after it individually converged. This is not a coupled block method.

    Each solve starts from zero unless given an initial guess. Acceptance
    compares the selected residual with the effective tolerance of the
    functional solver, ``max(tol * ||b||, atol)``, where ``atol`` is first raised
    to an arithmetic safety floor that depends on ``working_dtype``: 1e-18 for
    float32 and 1e-36 for float64. ``atol=0.0`` therefore means
    a relative tolerance only, except for a zero RHS, and the reported
    ``tolerance`` includes the floor. Restarted methods may exceed ``maxiter``
    by less than one cycle.

    Prepare and warm up outside capture. With graph-compatible operators, CUDA
    replay has no allocations or host convergence checks. Coefficients in
    existing buffers may change on replay; topology/address changes require
    recapture and preconditioner updates remain explicit. Eager calls issued
    from different CUDA streams are ordered with an event, but simultaneously
    executing graphs that share this state or its smaller capacities require
    caller ordering. Without conditional graph nodes (CUDA Toolkit and driver
    12.4+), CUDA solves run whole cycles up to ``maxiter`` and ``iterations``
    counts them. Reverse-mode differentiation is not supported.

    Args:
        A: Square real sparse matrix or :class:`LinearOperator`, float32 or float64.
            A one-dimensional vector-valued diagonal counts every component.
        nrhs: Positive number of RHSs prepared for :meth:`solve`.
        method: Iterative method to run:

            - ``"cg"``: conjugate gradient, :func:`cg`. Requires a symmetric positive
              definite system and preconditioner.
            - ``"cr"``: conjugate residual, :func:`cr`. Requires a symmetric system and
              a symmetric positive definite preconditioner.
            - ``"bicgstab"``: biconjugate gradient stabilized, :func:`bicgstab`.
              Handles nonsymmetric systems.
            - ``"gmres"``: restarted generalized minimal residual, :func:`gmres`.
              Handles nonsymmetric and indefinite systems.

        M: Optional fixed linear preconditioner: a single-system operator, or for
            ``nrhs > 1`` a prepared operator with ``nrhs`` equal partitions of
            length ``n``, such as a :class:`MultiRHSOperator`. Its arithmetic must
            use ``working_dtype``.
        tol: Nonnegative relative tolerance. Defaults to 1e-6 for a float32
            ``working_dtype`` and 1e-12 for float64.
        atol: Nonnegative absolute tolerance, raised to the safety floor above.
            Unlike the functional solvers, an omitted ``atol`` is ``0.0`` rather
            than the value of ``tol``, and ``None`` is rejected. At least one
            tolerance must be positive.
        maxiter: Positive iteration budget; truncation is allowed.
        restart: Krylov restart dimension for ``method="gmres"``, clipped to
            the system size. Defaults to 64. Other methods reject it.
        working_dtype: float32 or float64 inner arithmetic for a sparse ``A``;
            defaults to ``A``'s scalar type. RHSs and solutions keep ``A``'s type.
            Every solve refreshes the converted matrix values on device.
        algorithm: Batched product strategy for a sparse ``A``, resolved once at
            construction as in :class:`MultiRHSOperator`, which counts stored
            entries; explicit strategies pin a kernel.
        residual: Residual reported and used for acceptance. ``"recurrence"``
            uses the iterative solver's own estimate in working precision, at no
            extra cost. ``"fresh"`` recomputes ``||A x - b||`` with the source
            system and precision after every solve, at the cost of one operator
            application.

    Attributes:
        iterations: Device array of scheduled iterations per RHS from the latest call
            that recorded diagnostics; likewise for the per-RHS arrays below.
        absolute_residual: Device array of residual norms from the latest call.
            Nonfinite norms are reported as infinity.
        relative_residual: Device array of residual norms relative to ``||b||``.
            A zero RHS gives zero for a zero residual and infinity otherwise. With
            ``residual="recurrence"``, ``||b||`` is the norm the solver
            computed in ``working_dtype``, so a float32 working precision reports
            an infinite relative residual for an RHS whose squared norm underflows.
        tolerance: Device array of effective absolute tolerances from the latest call.
        converged: Device array of per-RHS acceptance flags, one for success.
        stats: Device array of cumulative iterations summed over RHSs, maximum
            iterations, failed RHSs and solved RHSs since the last ``reset_stats()``.
        residual_maxima: Device array of the maximum absolute and relative
            residuals since the last ``reset_stats()``.
    """

    def __init__(
        self,
        A,
        nrhs: int = 1,
        *,
        method: Literal["cg", "cr", "bicgstab", "gmres"],
        M=None,
        tol: float | None = None,
        atol: float = 0.0,
        maxiter: int = 1000,
        restart: int | None = None,
        working_dtype: type | None = None,
        algorithm: Literal["auto", "loop", "rows", "columns"] = "auto",
        residual: Literal["recurrence", "fresh"] = "recurrence",
    ):
        _validate_choice("Krylov method", method, KRYLOV_METHODS)
        maxiter = _positive_int("maxiter", maxiter)
        if method == "gmres":
            restart = _positive_int("restart", 64 if restart is None else restart)
        elif restart is not None:
            raise ValueError("restart applies only to method='gmres'")
        _validate_algorithm(algorithm)
        self.method = method
        self._sparse = isinstance(A, sp.BsrMatrix)
        single = aslinearoperator(A)
        n = _scalar_shape(single)[0]
        if _scalar_shape(single) not in ((n,), (n, n)) or n < 1:
            raise ValueError("A must be nonempty and square")
        if single.batch_offsets is not None:
            raise ValueError("A must describe one shared system, not partitioned systems")
        dtype = single.scalar_type
        if dtype not in (wp.float32, wp.float64):
            raise ValueError("A must have float32 or float64 scalar type")
        working_dtype = dtype if working_dtype is None else type_to_warp(working_dtype)
        if working_dtype not in (wp.float32, wp.float64):
            raise ValueError("working_dtype must be float32 or float64")
        if working_dtype != dtype and not self._sparse:
            raise ValueError("working_dtype requires a sparse matrix A")
        tol, atol = _resolve_tolerances(tol, atol, working_dtype)
        norm_dtype = wp.float64 if wp.float64 in (dtype, working_dtype) else wp.float32
        self._init_action(n, nrhs, dtype, single.device, residual, norm_dtype)
        self.source = A
        self.working_dtype = working_dtype
        self.tol, self.atol, self.maxiter, self.restart = tol, atol, maxiter, restart
        if working_dtype == dtype:
            self.working_matrix = A
        else:
            self.working_matrix = _working_copy(A, working_dtype)
        self._storage = self._storage_identity()
        if self.working_matrix is not A:
            self.update()
        # One RHS without a pinned strategy keeps the plain single-system operator.
        self._batched = self.nrhs > 1 or algorithm != "auto"
        self.operator = self._operator(self.working_matrix, algorithm)
        count = self.n * self.nrhs
        self._b = wp.zeros(count, dtype=working_dtype, device=self.device)
        self._x = wp.zeros_like(self._b)
        self._source_operator = self._x_source = self._ax = None
        if residual == "fresh":
            if working_dtype == dtype:
                self._source_operator, self._x_source = self.operator, self._x
            else:
                self._source_operator = self._operator(A, algorithm)
                self._x_source = wp.empty(count, dtype=dtype, device=self.device)
            self._ax = wp.empty(count, dtype=dtype, device=self.device)
        self._prepare_state(M)

    def _operator(self, A, algorithm):
        if not self._batched:
            return aslinearoperator(A)
        operator = MultiRHSOperator(A, self.nrhs, algorithm=algorithm)
        operator._stream_order = self._stream_order
        return operator

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
                    raise ValueError("Single-system M must match the system dimension")
                if self._batched:
                    M = MultiRHSOperator(M, self.nrhs)
            elif not self._batched:
                raise ValueError("A partitioned M requires nrhs > 1")
            elif M.shape != self.operator.shape or M.batch_count != self.nrhs or M.max_batch_length != self.n:
                raise ValueError("Prepared M must have nrhs equal partitions matching A")
            if M.scalar_type != self.working_dtype:
                raise ValueError("Prepared M must use the working dtype")
        options = {"restart": min(self.restart, self.n)} if self.method == "gmres" else {}
        self.state = _METHODS[self.method](
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
            **options,
        )

    def with_nrhs(self, nrhs: int, *, M=None) -> "KrylovSolve":
        """Prepare a smaller capacity sharing system storage and staging buffers.

        The returned action owns its Krylov workspace and diagnostics but shares
        stream ordering, so it must not run concurrently with this action.

        Args:
            nrhs: Positive capacity no greater than this action's ``nrhs``.
            M: Preconditioner for the smaller capacity. Required when the original
                preconditioner was a prepared partitioned operator; otherwise the
                original single-system preconditioner is reused by default.

        Returns:
            A :class:`KrylovSolve` that solves ``nrhs`` right-hand sides.
        """
        if M is None:
            M = self._M
            if M is not None and aslinearoperator(M).batch_offsets is not None:
                raise ValueError("Supply a matching prepared M for the smaller capacity")
        child = self._child(nrhs)
        count = self.n * child.nrhs
        if self._batched:
            child.operator = self.operator.with_nrhs(child.nrhs)
        child._b = self._b[:count]
        child._x = self._x[:count]
        if self._source_operator is not None:
            if self._source_operator is self.operator:
                child._source_operator, child._x_source = child.operator, child._x
            else:
                child._source_operator = (
                    self._source_operator.with_nrhs(child.nrhs) if self._batched else self._source_operator
                )
                child._x_source = self._x_source[:count]
            child._ax = self._ax[:count]
        child._prepare_state(M)
        return child

    def _storage_identity(self):
        if not self._sparse:
            return None
        a = self.source
        # The working copy shares these arrays, so replacing any of them invalidates it.
        row_counts = None if a.row_counts is None else a.row_counts.ptr
        return (a.offsets.ptr, a.columns.ptr, row_counts, a.values.ptr, a.values.size)

    def update(self):
        """Copy current source matrix values into the working-precision matrix.

        :meth:`solve` also does this on every call, so explicit calls are only
        needed by solver-agnostic code. Operators that are not sparse matrices are
        read directly, so this is a no-op for them. It is graph capturable.

        Raises:
            ValueError: If the source matrix's ``offsets``, ``columns``, ``row_counts`` or
                ``values`` array was replaced or reallocated after preparation.
        """
        if self._storage_identity() != self._storage:
            raise ValueError("A's sparse storage changed after preparation; prepare a new KrylovSolve")
        if self.working_dtype == self.dtype:
            return
        self._stream_order.acquire(self.device)
        source_flat = self.source.values.view(self.dtype).flatten()
        target_flat = self.working_matrix.values.view(self.working_dtype).flatten()
        wp.launch(_cast, target_flat.size, [source_flat, target_flat], device=self.device)

    def _stage(self, source, target):
        if source.dtype != target.dtype:
            wp.launch(_cast, target.size, [source, target], device=self.device)
        else:
            wp.copy(target, source)

    def solve(
        self, rhs: wp.array, out: wp.array, x0: wp.array | None = None, *, diagnostics: bool = True
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Solve into caller-owned storage, optionally refreshing device diagnostics.

        In-place and aliased arrays are safe because all RHSs and initial guesses
        are staged before any output is written. Passing ``out`` as ``x0`` warm
        starts from the previous solution.

        Args:
            rhs: Contiguous array of shape ``(nrhs, n)``, or ``(n,)`` for one RHS,
                with the source scalar dtype and device.
            out: Matching output. Exact in-place aliasing is supported, but
                shifted overlapping views are rejected before any output mutation.
            x0: Optional matching initial guess; the solve starts from zero when
                omitted. It may alias ``rhs`` or ``out``. The tolerance stays
                relative to ``||b||``, not to the initial residual.
            diagnostics: Whether to refresh the per-RHS diagnostic attributes,
                ``stats`` and ``residual_maxima``. ``False`` skips that bookkeeping
                for short, frequent solves; it requires
                ``residual="recurrence"``.

        Returns:
            Device arrays of the scheduled iteration count shared by the batch,
            the squared residual norm of each RHS used for acceptance, and the
            squared effective tolerance of each RHS. Tolerances use
            ``working_dtype``, and so do residuals with ``residual="recurrence"``.
            With ``residual="fresh"``, residuals use float64 if either the source
            or the working dtype is float64, so their dtype can differ from the
            tolerances'. The arrays belong to
            this action and are overwritten by its next solve.

        Raises:
            ValueError: If arrays do not match, if ``diagnostics`` is ``False``
                with ``residual="fresh"``, or if a source matrix storage array was
                replaced or reallocated after preparation.
        """
        if not diagnostics and self.residual == "fresh":
            raise ValueError("residual='fresh' recomputes residuals for diagnostics; use diagnostics=True")
        rows, result = self._prepare_solve(rhs, out, x0)
        self.update()
        self._stage(rows.flatten(), self._b)
        if x0 is None:
            self._x.zero_()
        else:
            self._stage(x0.flatten(), self._x)
        iterations, residual_sq, tolerance_sq = self.state()
        solution = self._x
        if self.residual == "fresh":
            if self._x_source is not self._x:
                wp.launch(_cast, self._x.size, [self._x, self._x_source], device=self.device)
            solution = self._x_source
            # The caller's RHS is still intact because no output has been written yet.
            self._source_operator.matvec(solution, self._ax, self._ax, 1.0, 0.0)
            self._residual_sq.zero_()
            self._rhs_sq.zero_()
            self._squared_norms(self._ax.reshape((self.nrhs, self.n)), rows, 0)
            residual_sq = self._residual_sq
            self._record(iterations, residual_sq, self._rhs_sq, tolerance_sq, 0, self.nrhs)
        elif diagnostics:
            # The solver stored ||b||^2 while setting its tolerances.
            self._record(iterations, residual_sq, self.state.rhs_norm_sq, tolerance_sq, 0, self.nrhs)
        self._stage(solution, result.flatten())
        return iterations, residual_sq, tolerance_sq
