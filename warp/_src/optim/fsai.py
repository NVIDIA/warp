# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive factorized sparse approximate inverse preconditioning in Warp.

Rows are built independently by growing lower-triangular supports on the graph
of the diagonally equilibrated matrix. For each support S, solve A[S,S] z = e_i
and set G[i,S] = z / sqrt(z[i]). Application uses two sparse products G.T G x;
there is no global sparse factorization or triangular solve.

See https://hypre.readthedocs.io/en/latest/solvers-fsai.html for the adaptive
FSAI algorithm. This implementation uses a row-size cap rather than a step cap.
"""

import types
from functools import lru_cache

import numpy as np

import warp as wp
from warp._src import sparse as sp
from warp._src.optim.linear import LinearOperator
from warp._src.utils import array_scan

__all__ = ["FSAI"]


@lru_cache(None)
def _block_kernels(block_type, dtype):
    """Create scalar-entry access and diagonal scaling that read BSR blocks in place."""
    block_size = 1 if block_type == dtype else block_type._shape_[0]

    if block_type == dtype:

        @wp.func
        def entry(
            offsets: wp.array[int],
            row_counts: wp.array[int],
            columns: wp.array[int],
            values: wp.array[block_type],
            r: int,
            c: int,
        ):
            slot = sp.bsr_block_index(r, c, offsets, columns, row_counts)
            result = dtype(0)
            if slot >= 0:
                result = values[slot]
            return result

    else:

        @wp.func
        def entry(
            offsets: wp.array[int],
            row_counts: wp.array[int],
            columns: wp.array[int],
            values: wp.array[block_type],
            r: int,
            c: int,
        ):
            slot = sp.bsr_block_index(
                r // wp.static(block_size), c // wp.static(block_size), offsets, columns, row_counts
            )
            result = dtype(0)
            if slot >= 0:
                result = dtype(values[slot][r % wp.static(block_size), c % wp.static(block_size)])
            return result

    @wp.kernel(enable_backward=False, module="unique")
    def diagonal(
        offsets: wp.array[int],
        row_counts: wp.array[int],
        columns: wp.array[int],
        values: wp.array[block_type],
        scale: wp.array[dtype],
        status: wp.array[int],
        counter: int,
    ):
        i = wp.tid()
        d = entry(offsets, row_counts, columns, values, i, i)
        if d > dtype(0) and wp.isfinite(d):
            scale[i] = dtype(1) / wp.sqrt(d)
        else:
            scale[i] = dtype(0)
            wp.atomic_add(status, counter, 1)

    return block_size, entry, diagonal


@lru_cache(None)
def _build_kernels(block_type, dtype, width, step_size=1):
    block_size, entry, _ = _block_kernels(block_type, dtype)
    best_indices = wp.types.vector(length=step_size, dtype=wp.int32)
    best_scores = wp.types.vector(length=step_size, dtype=dtype)
    indices = wp.types.vector(length=width, dtype=wp.int32)
    vector = wp.types.vector(length=width, dtype=dtype)
    matrix = wp.types.matrix(shape=(width, width), dtype=dtype)

    @wp.kernel(enable_backward=False, module="unique")
    def build(
        offsets: wp.array[int],
        row_counts: wp.array[int],
        columns: wp.array[int],
        values: wp.array[block_type],
        scale: wp.array[dtype],
        tolerance: dtype,
        pivot_floor: dtype,
        cols_out: wp.array[int],
        vals_out: wp.array[dtype],
        status: wp.array[int],
        ready: wp.array[int],
    ):
        i = wp.tid()
        if ready[0] != 0:
            return
        pattern = indices(-1)
        chol = matrix(dtype(0))
        z = vector(dtype(0))
        v = vector(dtype(0))
        pattern[0] = i
        chol[0, 0] = dtype(1)
        z[0] = dtype(1)
        v[0] = dtype(1)
        size = int(1)
        # Search the frontier of the current support. Duplicates are harmless:
        # candidate scores are evaluated exactly, without a truncated hash table.
        while size < wp.static(width):
            candidates = best_indices(-1)
            scores = best_scores(dtype(0))
            for p in range(size):
                row = pattern[p]
                block_row = row // wp.static(block_size)
                for e in range(offsets[block_row], sp._bsr_row_end(offsets, row_counts, block_row)):
                    for k in range(wp.static(block_size)):
                        c = columns[e] * wp.static(block_size) + k
                        selected = bool(c >= i)
                        for q in range(size):
                            if pattern[q] == c:
                                selected = True
                        for q in range(wp.static(step_size)):
                            if candidates[q] == c:
                                selected = True
                        if not selected:
                            residual = dtype(0)
                            for q in range(size):
                                j = pattern[q]
                                residual += (
                                    entry(offsets, row_counts, columns, values, c, j) * scale[c] * scale[j] * z[q]
                                )
                            score = wp.abs(residual)
                            for rank in range(wp.static(step_size)):
                                if score > scores[rank] or (score == scores[rank] and c < candidates[rank]):
                                    for rev in range(wp.static(step_size - 1)):
                                        dest = wp.static(step_size - 1) - rev
                                        if dest > rank:
                                            candidates[dest] = candidates[dest - 1]
                                            scores[dest] = scores[dest - 1]
                                    candidates[rank] = c
                                    scores[rank] = score
                                    break
            if candidates[0] < 0:
                break
            old_z0 = z[0]
            stopped = bool(False)
            for rank in range(wp.static(step_size)):
                best = candidates[rank]
                if best < 0 or size == wp.static(width):
                    break
                # Border the Cholesky factor of the selected principal submatrix.
                w = vector(dtype(0))
                pivot = dtype(1)
                for p in range(size):
                    a = entry(offsets, row_counts, columns, values, best, pattern[p]) * scale[best] * scale[pattern[p]]
                    for q in range(p):
                        a -= chol[p, q] * w[q]
                    w[p] = a / chol[p, p]
                    pivot -= w[p] * w[p]
                if pivot <= pivot_floor or not wp.isfinite(pivot):
                    wp.atomic_add(status, 1, 1)
                    stopped = True
                    break
                pattern[size] = best
                for p in range(size):
                    chol[size, p] = w[p]
                chol[size, size] = wp.sqrt(pivot)
                # Existing forward-solve entries are unchanged by bordering L.
                rhs = dtype(0)
                for p in range(size):
                    rhs -= w[p] * v[p]
                v[size] = rhs / chol[size, size]
                size += 1
            # A[S,S] z = e_0; no global factorization or triangular dependency.
            for rev in range(size):
                p = size - 1 - rev
                a = v[p]
                for q in range(p + 1, size):
                    a -= chol[q, p] * z[q]
                z[p] = a / chol[p, p]
            # psi=1/z[0]. Relative reduction is 1-old_z0/new_z0.
            if stopped or (tolerance > dtype(0) and dtype(1) - old_z0 / z[0] <= tolerance):
                break
        norm = wp.sqrt(z[0])
        valid = bool(True)
        for p in range(wp.static(width)):
            e = i * wp.static(width) + p
            cols_out[e] = -1
            vals_out[e] = dtype(0)
            if p < size:
                cols_out[e] = pattern[p]
                vals_out[e] = z[p] / norm * scale[pattern[p]]
                if not wp.isfinite(vals_out[e]):
                    valid = False
                if p == 0 and vals_out[e] <= dtype(0):
                    valid = False
        if not valid:
            wp.atomic_add(status, 2, 1)

    return build


@wp.kernel(enable_backward=False, module="unique")
def _validate_float32_factor(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.float32],
    status: wp.array[int],
):
    row = wp.tid()
    invalid = bool(False)
    for e in range(offsets[row], offsets[row + 1]):
        if not wp.isfinite(values[e]):
            invalid = True
        if columns[e] == row and values[e] <= wp.float32(0):
            invalid = True
    if invalid:
        wp.atomic_add(status, 0, 1)


class FSAI(LinearOperator):
    """Build an adaptive factorized sparse approximate inverse of an SPD BSR matrix.

    This :class:`warp.optim.linear.LinearOperator` applies ``G.T @ G`` and can be
    passed as ``M`` to :func:`warp.optim.linear.cg` or :func:`warp.optim.linear.cr`.
    Rows of the lower-triangular factor ``G`` are built independently on the matrix
    device. Square blocks are expanded into scalar entries, so this is scalar FSAI
    even when the input uses block storage. No host numerical backend is used.

    Args:
        A: Square :class:`warp.sparse.BsrMatrix` with square blocks and ``float16``,
            ``float32`` or ``float64`` scalar entries. Both triangles must be stored. Symmetry
            and positive definiteness are caller preconditions, not checked globally.
            Stored columns must be sorted and unique within each row, as produced
            by the standard BSR construction functions.
        max_row_size: Maximum scalar entries per factor row, including its diagonal,
            in [1, 64]. One entry gives Jacobi preconditioning. Larger supports use
            quadratic per-thread setup workspace and increase setup and apply costs.
        kap_tolerance: Relative local energy improvement below which support growth
            stops, in [0, 1). Zero disables this stopping test. It is checked after
            each batch, so changing ``max_step_size`` may require retuning it.
        pivot_floor: Smallest accepted pivot of the diagonally equilibrated local
            Cholesky factorization, in (0, 1). ``None`` selects 1e-3 for ``float16``,
            1e-6 for ``float32`` and 1e-12 for ``float64``. Setup truncates a row
            before a rejected pivot.
        factor_dtype: Factor storage type: the matrix scalar type, or ``wp.float32``
            to compress a ``float64`` factor. ``None`` uses the matrix scalar type.
            Products accumulate in the matrix scalar type; the transpose is formed
            from the same rounded factor, preserving the Gram form.
        reuse_pattern: Retain a plan for :meth:`update` on the original factor
            supports. Requires compact input storage; use :func:`warp.sparse.bsr_compress`
            with ``prune_numerical_zeros=False`` to compact a padded matrix first. Ordinary construction accepts either.
        max_step_size: Maximum distinct largest-residual frontier entries selected
            per growth step, in [1, 64]. Batching reduces searches but can change the
            selected supports. A short batch does not exhaust the row-size budget.
        regularize_refit: With ``reuse_pattern``, make :meth:`update` refit every row it can
            instead of rejecting the whole factor when one row fails. A row whose pivot
            falls to ``pivot_floor`` or below is refit with the pivot raised to the floor and
            counted as regularized. A row with a nonfinite pivot, an invalid diagonal, or a
            nonpositive result is replaced by its scaled diagonal and counted as failed. Only a
            topology mismatch rejects the update. Inspect :attr:`update_status` to see
            affected rows.
        validate_topology: With ``reuse_pattern``, have :meth:`update` compare the matrix
            topology with the original on the device and reject a mismatch. Pass ``False``
            to skip the comparison when the caller guarantees an unchanged topology; a
            changed topology then gives undefined results and may read out of bounds.
        defer_pattern: Construct from the topology of ``A`` alone, without reading its values,
            and select the supports on the device during the first successful :meth:`update`.
            Requires ``reuse_pattern``. Until then the preconditioner is the identity. Every
            kernel of that selection checks :attr:`pattern_ready` and does nothing once the
            pattern exists, so a CUDA graph that captured the first update selects the pattern
            on its first replay and only refits on later replays. ``G`` is then stored with
            padded rows (``row_counts``), ``GT`` has capacity for ``max_row_size`` entries per
            row.
            A row with an unusable diagonal keeps a diagonal-only support, which is reported
            in :attr:`update_status`, if ``regularize_refit`` is set; otherwise the update is
            rejected and the next one selects again.

    Attributes:
        G: Canonical scalar BSR factor with finite entries and a positive diagonal.
        GT: Stored transpose of ``G``, including any storage rounding.
        truncated_rows: Number of setup rows stopped by the pivot safeguard, or ``None``
            with ``defer_pattern`` (see :attr:`pattern_status`).
        pattern_ready: ``None`` without ``defer_pattern``. Otherwise an ``int`` array on the
            matrix device holding ``1`` once the supports have been selected and published.
        pattern_status: ``None`` without ``defer_pattern``. Otherwise an ``int`` array on the
            matrix device with the counts from the most recent support selection: the rows
            stopped by the pivot safeguard at index 1 and the rows whose factor lost finite
            entries or a positive diagonal at index 2. Index 0 is unused.
        source: Matrix used by the most recent successful construction or update.
        update_status: ``None`` without ``reuse_pattern``. Otherwise an ``int`` array
            on the matrix device with four counts from the most recent update: the
            topology mismatches against the original matrix, the invalid diagonals,
            the local factorizations that failed, and the rows regularized (always
            zero unless ``regularize_refit``). By default the new factors were applied
            only if the first three are zero; with ``regularize_refit`` they were applied
            unless the topology mismatched.

    Note:
        Construction synchronizes and cannot be captured, but with ``defer_pattern`` it
        needs only the topology of the matrix. :meth:`update` can be
        captured in a CUDA graph (see its documentation), and application is
        CUDA graph-capturable after warming up the kernels, including inside the
        conditional graph loop of the iterative solvers. Applications of one instance
        on different streams are ordered against each other and never overlap. Do not
        overlap an update with application, and do not replay one captured graph that
        applies the instance concurrently with another use of the same instance. Treat ``G`` and ``GT`` as read-only.
        Rebuild or call :meth:`update` after changing the input values. This
        preconditioner does not provide automatic differentiation.

    Raises:
        TypeError: The matrix scalar type is unsupported.
        ValueError: The matrix shape or an option is invalid, a diagonal is missing,
            nonpositive or nonfinite, or the factor cannot be represented by the
            selected storage type with finite entries and a positive diagonal.

    Example:
        .. code-block:: python

            from warp.optim.linear import FSAI, cg

            M = FSAI(A, max_row_size=16, reuse_pattern=True)
            cg(A, b, x, M=M)
            # After updating A.values without changing A's sparsity:
            M.update()
            cg(A, b, x, M=M)
    """

    def __init__(
        self,
        A: sp.BsrMatrix,
        max_row_size: int = 8,
        kap_tolerance: float = 1.0e-3,
        pivot_floor: float | None = None,
        factor_dtype: type | None = None,
        reuse_pattern: bool = False,
        max_step_size: int = 1,
        regularize_refit: bool = False,
        validate_topology: bool = True,
        defer_pattern: bool = False,
    ):
        if not isinstance(A, sp.BsrMatrix) or A.shape[0] != A.shape[1] or A.block_shape[0] != A.block_shape[1]:
            raise ValueError("FSAI requires a square BSR matrix with square blocks")
        if A.scalar_type not in (wp.float16, wp.float32, wp.float64):
            raise TypeError("FSAI supports float16, float32 and float64")
        if not isinstance(max_row_size, int) or not 1 <= max_row_size <= 64:
            raise ValueError("max_row_size must be an integer in [1, 64]")
        if not isinstance(max_step_size, int) or not 1 <= max_step_size <= 64:
            raise ValueError("max_step_size must be an integer in [1, 64]")
        if not 0 <= kap_tolerance < 1:
            raise ValueError("kap_tolerance must be in [0, 1)")
        if pivot_floor is None:
            pivot_floor = {wp.float16: 1.0e-3, wp.float32: 1.0e-6, wp.float64: 1.0e-12}[A.scalar_type]
        if not 0 < pivot_floor < 1:
            raise ValueError("pivot_floor must be in (0, 1)")
        if factor_dtype is None:
            factor_dtype = A.scalar_type
        if factor_dtype != A.scalar_type and not (A.scalar_type == wp.float64 and factor_dtype == wp.float32):
            raise ValueError("factor_dtype must be the matrix scalar type, or wp.float32 for a float64 matrix")
        if reuse_pattern and A.row_counts is not None:
            raise ValueError("reuse_pattern requires compact BSR storage; compact with bsr_compress")
        if defer_pattern and not reuse_pattern:
            raise ValueError("defer_pattern requires reuse_pattern")
        self.max_row_size = max_row_size
        self.max_step_size = max_step_size
        self.factor_dtype = factor_dtype
        self.source = A
        self.pattern_ready = None
        self.pattern_status = None
        if defer_pattern:
            self.truncated_rows = None
            self._refit = _DeferredPlan(
                self, A, max_row_size, max_step_size, kap_tolerance, pivot_floor, regularize_refit, validate_topology
            )
            self.update_status = self._refit.status
            self.pattern_ready = self._refit.ready
            self.pattern_status = self._refit.pattern_status
            self._init_runtime(A, A.shape[0], A.scalar_type, A.device)
            if A.device.is_cuda:
                wp.synchronize_stream(A.device.stream)
            return
        # Selection reads the matrix blocks in place, including padded storage; no
        # scalar copy and no host matrix staging.
        n = A.shape[0]
        dtype, device = A.scalar_type, A.device
        scale = wp.empty(n, dtype=dtype, device=device)
        status = wp.zeros(3, dtype=int, device=device)
        _, _, diagonal = _block_kernels(A.dtype, dtype)
        build = _build_kernels(A.dtype, dtype, max_row_size, max_step_size)
        wp.launch(
            diagonal,
            n,
            [A.offsets, A.row_counts, A.columns, A.values, scale, status, 0],
            device=device,
        )
        cols = wp.empty(n * max_row_size, dtype=int, device=device)
        vals = wp.empty(n * max_row_size, dtype=dtype, device=device)
        wp.launch(
            build,
            n,
            [
                A.offsets,
                A.row_counts,
                A.columns,
                A.values,
                scale,
                dtype(kap_tolerance),
                dtype(pivot_floor),
                cols,
                vals,
                status,
                wp.zeros(1, dtype=int, device=device),
            ],
            device=device,
        )
        diagnostics = status.numpy()
        if diagnostics[0]:
            raise ValueError(f"FSAI requires finite positive diagonals ({diagnostics[0]} invalid rows)")
        self.truncated_rows = int(diagnostics[1])
        if diagnostics[2]:
            raise ValueError("FSAI factor lost finite entries or a positive diagonal")

        self.G, growth_rank = _pack_factor(n, max_row_size, cols, vals)
        if factor_dtype != dtype:
            self.G = sp.bsr_copy(self.G, scalar_type=factor_dtype)
            status.zero_()
            wp.launch(
                _validate_float32_factor,
                n,
                [self.G.offsets, self.G.columns, self.G.values, status],
                device=device,
            )
            if status.numpy()[0]:
                raise ValueError("factor_dtype conversion lost finite entries or a positive diagonal")
        self.GT = sp.bsr_transposed(self.G)
        self._init_runtime(A, n, dtype, device)
        self._refit = None
        self.update_status = None
        if reuse_pattern:
            self._refit = _RefitPlan(
                self, A, growth_rank, max_row_size, pivot_floor, regularize_refit, validate_topology
            )
            self.update_status = self._refit.status
        if device.is_cuda:
            # Factor work is asynchronous; finish it so first use on any other stream,
            # or inside a captured graph, never reads unfinished factors.
            wp.synchronize_stream(device.stream)

    def _init_runtime(self, A, n, dtype, device):
        # One persistent scratch vector, so application allocates nothing and can run
        # inside a conditional graph body. Applications on different CUDA streams are
        # ordered against each other through an event.
        self._tmp = wp.empty(n, dtype=dtype, device=device)
        self._tmp_event = wp.Event(device) if device.is_cuda else None
        self._last_stream = None
        self._views = {}
        super().__init__(A.shape, A.dtype, device, self._apply)

    def _flat(self, vector):
        """Return a flat scalar view of ``vector``, cached so repeated applications skip the view setup."""
        if vector.dtype == self.scalar_type and vector.ndim == 1:
            return vector
        key = (vector.ptr, vector.shape, vector.dtype)
        entry = self._views.get(key)
        if entry is None:
            if len(self._views) >= 8:
                self._views.clear()
            # The entry keeps ``vector`` alive, so its address cannot be reused while cached.
            entry = (vector, vector.view(self.scalar_type).flatten())
            self._views[key] = entry
        return entry[1]

    def update(self, A: sp.BsrMatrix | None = None) -> "FSAI":
        """Refit on the original factor supports, preserving captured apply buffers.

        Requires ``reuse_pattern=True`` and identical compact BSR topology/storage.
        This is not a new adaptive pattern search, except that with ``defer_pattern``
        the first successful update selects the supports. Rebuild when convergence worsens.
        Failed validation leaves the previous factors intact.

        Outside graph capture, the update synchronizes and raises on failure, except that
        with both ``regularize_refit`` and ``validate_topology=False`` nothing can raise, so
        it does not synchronize. During
        CUDA graph capture, the update performs no host reads and cannot raise
        for numerical failures; after each replay, inspect :attr:`update_status`.
        Run one eager update first so the kernels are loaded before capture, and
        have the matrix keep the same arrays so a replay reads its new values.

        Args:
            A: Matrix with new numerical values. ``None`` reuses :attr:`source`.

        Returns:
            This preconditioner with updated numerical factors.

        Raises:
            ValueError: No reuse plan exists, matrix metadata or array sizes changed,
                or, when not capturing, the topology changed or a local factorization
                or storage conversion failed. Unlike
                initial construction, refits reject a bad pivot without truncating
                the fixed support. Rebuild to select new supports.
        """
        if self._refit is None:
            raise ValueError("Construct FSAI with reuse_pattern=True before updating")
        A = self.source if A is None else A
        if not isinstance(A, sp.BsrMatrix):
            raise ValueError("update requires a Warp BSR matrix")
        self._refit.update(self, A)
        self.source = A
        return self

    def _apply(self, x, y, z, alpha, beta):
        x, y, z = (self._flat(v) for v in (x, y, z))
        # Outside graph capture, order this application after the previous one on any
        # other stream, since both share the scratch vector. Inside capture on this
        # stream the graph's own dependencies apply and events must not be recorded or
        # waited on; another stream capturing does not exempt this one.
        stream = x.device.stream if x.device.is_cuda and not x.device.stream.is_capturing else None
        if stream is not None and stream is not self._last_stream:
            # Work on one stream is already ordered; only a change of stream needs an
            # event, recorded on the stream that last used the scratch vector.
            last = self._last_stream
            if last is not None and not last.is_capturing:
                last.record_event(self._tmp_event)
                stream.wait_event(self._tmp_event)
            self._last_stream = stream
        if alpha != 0.0:
            _matvec(self.G, x, self._tmp, self._tmp, 1.0, 0.0)
        # x is fully consumed before writing z, including when x, y, z alias.
        _matvec(self.GT, self._tmp, y, z, alpha, beta)


@lru_cache(None)
def _pack_kernels(dtype, width):
    ivec = wp.types.vector(length=width, dtype=int)
    vec = wp.types.vector(length=width, dtype=dtype)

    @wp.kernel(enable_backward=False, module="unique")
    def sort_rows(
        columns: wp.array[int],
        values: wp.array[dtype],
        ranks: wp.array[int],
        counts: wp.array[int],
        ready: wp.array[int],
    ):
        i = wp.tid()
        if ready[0] != 0:
            return
        cols = ivec(-1)
        vals = vec(dtype(0))
        order = ivec(-1)
        size = int(0)
        for p in range(wp.static(width)):
            c = columns[i * wp.static(width) + p]
            if c >= 0:
                v = values[i * wp.static(width) + p]
                q = size
                while q > 0:
                    if cols[q - 1] <= c:
                        break
                    cols[q] = cols[q - 1]
                    vals[q] = vals[q - 1]
                    order[q] = order[q - 1]
                    q -= 1
                cols[q] = c
                vals[q] = v
                order[q] = p
                size += 1
        counts[i + 1] = size
        for p in range(size):
            columns[i * wp.static(width) + p] = cols[p]
            values[i * wp.static(width) + p] = vals[p]
            ranks[i * wp.static(width) + p] = order[p]

    @wp.kernel(enable_backward=False, module="unique")
    def pack(
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        ranks: wp.array[int],
        outcols: wp.array[int],
        outvals: wp.array[dtype],
        outranks: wp.array[int],
    ):
        i = wp.tid()
        for e in range(offsets[i], offsets[i + 1]):
            p = i * wp.static(width) + e - offsets[i]
            outcols[e] = columns[p]
            outvals[e] = values[p]
            outranks[e] = ranks[p]

    return sort_rows, pack


def _pack_factor(n, width, columns, values):
    """Pack rows into sorted BSR storage, also returning each entry's growth rank."""
    result = sp.bsr_zeros(n, n, values.dtype, device=values.device)
    counts = wp.zeros(n + 1, dtype=int, device=values.device)
    sort, pack = _pack_kernels(values.dtype, width)
    ranks = wp.empty(n * width, dtype=int, device=values.device)
    wp.launch(
        sort, n, [columns, values, ranks, counts, wp.zeros(1, dtype=int, device=values.device)], device=values.device
    )
    array_scan(counts, result.offsets)
    result.notify_nnz_changed()
    packed_ranks = wp.empty(result.nnz_sync(), dtype=int, device=values.device)
    wp.launch(
        pack,
        n,
        [result.offsets, columns, values, ranks, result.columns, result.values, packed_ranks],
        device=values.device,
    )
    return result, packed_ranks


@wp.kernel(enable_backward=False, module="unique")
def _compare_topology(a: wp.array[int], b: wp.array[int], bad: wp.array[int]):
    i = wp.tid()
    if a[i] != b[i]:
        wp.atomic_add(bad, 0, 1)


@wp.kernel(enable_backward=False, module="unique")
def _transpose_map(
    offsets: wp.array[int],
    columns: wp.array[int],
    toff: wp.array[int],
    tcol: wp.array[int],
    mapping: wp.array[int],
):
    i = wp.tid()
    for e in range(offsets[i], offsets[i + 1]):
        mapping[e] = sp.bsr_block_index(columns[e], i, toff, tcol)


@lru_cache(None)
def _deferred_kernels(dtype, storage, width, regularize):
    """Create the kernels that select supports and build the transposed index on the device."""

    @wp.func
    def pending(ready: wp.array[int], status: wp.array[int]):
        # The pattern is built only once, and only if the update was accepted.
        ok = ready[0] == 0 and status[0] == 0
        if not wp.static(regularize):
            ok = ok and status[1] == 0 and status[2] == 0
        return ok

    @wp.func
    def accepted(status: wp.array[int]):
        ok = status[0] == 0
        if not wp.static(regularize):
            ok = ok and status[1] == 0 and status[2] == 0
        return ok

    @wp.kernel(enable_backward=False, module="unique")
    def zero(values: wp.array[int], ready: wp.array[int]):
        e = wp.tid()
        if ready[0] == 0:
            values[e] = 0

    @wp.kernel(enable_backward=False, module="unique")
    def publish(
        support_columns: wp.array[int],
        support_counts: wp.array[int],
        stride: int,
        counts: wp.array[int],
        ready: wp.array[int],
        status: wp.array[int],
        columns: wp.array[int],
        row_counts: wp.array[int],
    ):
        # Copy the selected supports into the factor and count the entries of every column.
        row = wp.tid()
        if pending(ready, status):
            size = support_counts[row + 1]
            row_counts[row] = size
            for p in range(size):
                column = support_columns[row * stride + p]
                columns[row * stride + p] = column
                wp.atomic_add(counts, column + 1, 1)

    @wp.kernel(enable_backward=False, module="unique")
    def scan(counts: wp.array[int], offsets: wp.array[int], ready: wp.array[int], status: wp.array[int]):
        # One thread, once: the pattern is built a single time.
        if pending(ready, status):
            total = int(0)
            for c in range(offsets.shape[0]):
                total += counts[c]
                offsets[c] = total

    @wp.kernel(enable_backward=False, module="unique")
    def fill(
        columns: wp.array[int],
        row_counts: wp.array[int],
        stride: int,
        offsets: wp.array[int],
        cursor: wp.array[int],
        entries: wp.array[int],
        ready: wp.array[int],
        status: wp.array[int],
    ):
        row = wp.tid()
        if pending(ready, status):
            for p in range(row_counts[row]):
                column = columns[row * stride + p]
                entries[offsets[column] + wp.atomic_add(cursor, column, 1)] = row * stride + p

    @wp.kernel(enable_backward=False, module="unique")
    def sort_columns(
        offsets: wp.array[int],
        entries: wp.array[int],
        stride: int,
        columns: wp.array[int],
        ready: wp.array[int],
        status: wp.array[int],
    ):
        # Sorting each column by factor entry makes the transposed sums independent of
        # the order in which the scatter ran. The sorted entries give the transposed rows.
        column = wp.tid()
        if pending(ready, status):
            begin = offsets[column]
            end = offsets[column + 1]
            for e in range(begin + 1, end):
                key = entries[e]
                q = e - 1
                while q >= begin and entries[q] > key:
                    entries[q + 1] = entries[q]
                    q -= 1
                entries[q + 1] = key
            for e in range(begin, end):
                columns[e] = entries[e] // stride

    @wp.kernel(enable_backward=False, module="unique")
    def mark_ready(ready: wp.array[int], status: wp.array[int]):
        if pending(ready, status):
            ready[0] = 1

    @wp.kernel(enable_backward=False, module="unique")
    def commit(
        candidate: wp.array[storage],
        status: wp.array[int],
        entries: wp.array[int],
        offsets: wp.array[int],
        values: wp.array[storage],
        transposed: wp.array[storage],
        n: int,
    ):
        e = wp.tid()
        if accepted(status):
            values[e] = candidate[e]
            if e < offsets[n]:
                transposed[e] = candidate[entries[e]]

    return types.SimpleNamespace(
        zero=zero,
        publish=publish,
        scan=scan,
        fill=fill,
        sort_columns=sort_columns,
        mark_ready=mark_ready,
        commit=commit,
    )


@lru_cache(None)
def _refit_kernels(block_type, dtype, storage, width, regularize):
    _, entry, diagonal = _block_kernels(block_type, dtype)
    vec = wp.types.vector(length=width, dtype=dtype)
    mat = wp.types.matrix(shape=(width, width), dtype=dtype)
    ivec = wp.types.vector(length=width, dtype=int)

    @wp.kernel(enable_backward=False, module="unique")
    def refit(
        offsets: wp.array[int],
        row_counts: wp.array[int],
        columns: wp.array[int],
        values: wp.array[block_type],
        goff: wp.array[int],
        gcol: wp.array[int],
        gcount: wp.array[int],
        growth_rank: wp.array[int],
        scale: wp.array[dtype],
        floor: dtype,
        out: wp.array[storage],
        status: wp.array[int],
    ):
        i = wp.tid()
        start = goff[i]
        size = goff[i + 1] - start
        if gcount:
            size = gcount[i]
        # Factor in the order the support was grown, so pivots (and the floor test)
        # match construction; sorted storage order can give very different pivots.
        # The local system is padded with identity rows to the static width, which leaves
        # the solution on the support unchanged and keeps every loop bound constant.
        member = ivec(0)
        for p in range(size):
            member[growth_rank[start + p]] = p
        local = ivec(-1)
        for k in range(size):
            local[k] = gcol[start + member[k]]
        chol = mat(dtype(0))
        valid = bool(True)
        regularized = bool(False)
        for k in range(wp.static(width)):
            r = local[k]
            for m in range(k + 1):
                c = local[m]
                a = dtype(0)
                if k < size:
                    if r == c:
                        # Equilibrated diagonals are exactly one, as in construction.
                        a = dtype(1)
                    else:
                        a = entry(offsets, row_counts, columns, values, r, c) * scale[r] * scale[c]
                elif k == m:
                    a = dtype(1)
                for t in range(m):
                    a -= chol[k, t] * chol[m, t]
                if k == m:
                    if not wp.isfinite(a):
                        valid = False
                        a = dtype(1)
                    elif a <= floor:
                        if wp.static(regularize):
                            # Raise the pivot to the floor and report the row.
                            regularized = True
                            a = floor
                        else:
                            valid = False
                            a = dtype(1)
                    chol[k, m] = wp.sqrt(a)
                else:
                    chol[k, m] = a / chol[m, m]
        # Growth position 0 is the row's own index.
        z = vec(dtype(0))
        for k in range(wp.static(width)):
            a = dtype(0)
            if k == 0:
                a = dtype(1)
            for t in range(k):
                a -= chol[k, t] * z[t]
            z[k] = a / chol[k, k]
        for rev in range(wp.static(width)):
            k = wp.static(width) - 1 - rev
            a = z[k]
            for t in range(k + 1, wp.static(width)):
                a -= chol[t, k] * z[t]
            z[k] = a / chol[k, k]
        norm = wp.sqrt(z[0])
        for p in range(size):
            value = storage(z[growth_rank[start + p]] / norm * scale[gcol[start + p]])
            out[start + p] = value
            if not wp.isfinite(value):
                valid = False
            # Canonical lower-triangular support: the diagonal is last.
            if p == size - 1 and value <= storage(0):
                valid = False
        if wp.static(regularize):
            # A failed row falls back to its own scaled diagonal (Jacobi), which stays
            # finite and positive, so one bad row never blocks the others.
            if not valid:
                fallback = storage(scale[i])
                if not (scale[i] > dtype(0) and wp.isfinite(fallback) and fallback > storage(0)):
                    fallback = storage(1)
                for p in range(size):
                    out[start + p] = storage(0)
                out[start + size - 1] = fallback
        if not valid:
            wp.atomic_add(status, 2, 1)
        elif regularized:
            wp.atomic_add(status, 3, 1)

    @wp.kernel(enable_backward=False, module="unique")
    def commit(
        candidate: wp.array[storage],
        status: wp.array[int],
        mapping: wp.array[int],
        values: wp.array[storage],
        transposed: wp.array[storage],
    ):
        # Publish the refit factor only if validation passed; otherwise the previous
        # factor stays in place. With regularization only a topology mismatch blocks.
        e = wp.tid()
        accepted = status[0] == 0
        if not wp.static(regularize):
            accepted = accepted and status[1] == 0 and status[2] == 0
        if accepted:
            values[e] = candidate[e]
            transposed[mapping[e]] = candidate[e]

    return diagonal, refit, commit


class _PlanBase:
    """State and validation shared by the refit plans."""

    def __init__(self, A, n, factor_dtype, width, floor, regularize, validate_topology):
        self.signature = (A.shape, A.dtype, A.device, A.row_counts is not None)
        # Every size used by an update is fixed here, so updating never reads the
        # block count back from the device and can be captured in a CUDA graph.
        self.n = n
        self.nrow = A.nrow
        self.nnz = A.nnz_sync()
        self.topology = None
        if validate_topology:
            self.topology = [wp.clone(A.offsets[: A.nrow + 1])]
            if self.nnz:
                self.topology.append(wp.clone(A.columns[: self.nnz]))
        self.floor = floor
        self.regularize = regularize
        # Only these configurations can report a failure to the host; otherwise an eager
        # update never needs to synchronize.
        self.raises = validate_topology or not regularize
        self.diagonal, self.refit, self.commit = _refit_kernels(A.dtype, A.scalar_type, factor_dtype, width, regularize)
        self.scale = wp.empty(n, dtype=A.scalar_type, device=A.device)
        # Counts of [topology mismatches, invalid diagonals, failed local factorizations,
        # regularized local factorizations].
        self.status = wp.zeros(4, dtype=int, device=A.device)
        self.topology_status = self.status[0:1]

    def prepare(self, A):
        """Validate ``A``, reset the status, and compute the diagonal scaling."""
        if (A.shape, A.dtype, A.device, A.row_counts is not None) != self.signature:
            raise ValueError("FSAI update requires the original shape, dtype, device and topology")
        if A.offsets.size < self.nrow + 1 or A.columns.size < self.nnz or A.values.size < self.nnz:
            raise ValueError("FSAI update requires storage covering every block of the original matrix")
        self.status.zero_()
        if self.topology is not None:
            wp.launch(
                _compare_topology,
                self.nrow + 1,
                [A.offsets[: self.nrow + 1], self.topology[0], self.topology_status],
                device=A.device,
            )
            if self.nnz:
                wp.launch(
                    _compare_topology,
                    self.nnz,
                    [A.columns[: self.nnz], self.topology[1], self.topology_status],
                    device=A.device,
                )
        wp.launch(
            self.diagonal,
            self.n,
            [A.offsets, None, A.columns, A.values, self.scale, self.status, 1],
            device=A.device,
        )

    def check(self):
        """Raise if the most recent update was rejected, synchronizing the device."""
        topology, diagonal, refit, _ = self.status.numpy()
        if topology:
            raise ValueError("FSAI update requires identical sparsity; rebuild to change it")
        if (diagonal or refit) and not self.regularize:
            raise ValueError("FSAI refit requires finite positive diagonals and SPD local supports")


class _RefitPlan(_PlanBase):
    """Refit on supports selected at construction, with compact factors."""

    def __init__(self, owner, A, growth_rank, width, floor, regularize, validate_topology):
        super().__init__(A, owner.G.nrow, owner.factor_dtype, width, floor, regularize, validate_topology)
        self.growth_rank = growth_rank
        self.transpose = wp.empty(owner.G.nnz_sync(), dtype=int, device=A.device)
        wp.launch(
            _transpose_map,
            owner.G.nrow,
            [owner.G.offsets, owner.G.columns, owner.GT.offsets, owner.GT.columns, self.transpose],
            device=A.device,
        )
        self.candidate = wp.empty_like(owner.G.values)

    def update(self, owner, A):
        self.prepare(A)
        wp.launch(
            self.refit,
            self.n,
            [
                A.offsets,
                None,
                A.columns,
                A.values,
                owner.G.offsets,
                owner.G.columns,
                None,
                self.growth_rank,
                self.scale,
                A.scalar_type(self.floor),
                self.candidate,
                self.status,
            ],
            device=A.device,
        )
        wp.launch(
            self.commit,
            self.transpose.size,
            [self.candidate, self.status, self.transpose, owner.G.values, owner.GT.values],
            device=A.device,
        )
        if not A.device.is_capturing and self.raises:
            self.check()


class _DeferredPlan(_PlanBase):
    """Select the supports on the device at the first successful update.

    Every kernel that builds the pattern or the transposed index first checks the
    ``ready`` flag and returns if it is set, so a graph that captured the first update
    selects the pattern on its first replay and only refits on later replays.
    """

    def __init__(self, owner, A, width, step_size, tolerance, floor, regularize, validate_topology):
        n = A.shape[0]
        device, dtype, storage = A.device, A.scalar_type, owner.factor_dtype
        super().__init__(A, n, storage, width, floor, regularize, validate_topology)
        self.width = width
        self.tolerance = tolerance
        self.select = _build_kernels(A.dtype, dtype, width, step_size)
        self.sort, _ = _pack_kernels(dtype, width)
        self.kernels = _deferred_kernels(dtype, storage, width, regularize)
        self.ready = wp.zeros(1, dtype=int, device=device)
        self.pattern_status = wp.zeros(3, dtype=int, device=device)
        self.published = False
        self.support_columns = wp.full(n * width, -1, dtype=int, device=device)
        self.support_values = wp.zeros(n * width, dtype=dtype, device=device)
        self.support_ranks = wp.zeros(n * width, dtype=int, device=device)
        self.support_counts = wp.zeros(n + 1, dtype=int, device=device)
        self.candidate = wp.zeros(n * width, dtype=storage, device=device)
        self.transpose_counts = wp.zeros(n + 1, dtype=int, device=device)
        self.transpose_cursor = wp.zeros(n, dtype=int, device=device)
        self.transpose_entries = wp.zeros(n * width, dtype=int, device=device)

        # Until the first update the factor is the identity.
        rows = np.arange(n)
        columns = np.full(n * width, -1, dtype=np.int32)
        columns[rows * width] = rows
        values = np.zeros(n * width, dtype=wp.dtype_to_numpy(storage))
        values[rows * width] = 1
        owner.G = sp.bsr_zeros(n, n, storage, device=device, row_capacity=width)
        owner.G.columns = wp.array(columns, dtype=int, device=device)
        owner.G.values = wp.array(values, dtype=storage, device=device)
        owner.G.row_counts = wp.full(n, 1, dtype=int, device=device)
        owner.G.notify_nnz_changed(nnz=n * width)
        owner.GT = sp.bsr_zeros(n, n, storage, device=device)
        owner.GT.offsets = wp.array(np.arange(n + 1), dtype=int, device=device)
        transposed_columns = np.zeros(n * width, dtype=np.int32)
        transposed_columns[:n] = rows
        owner.GT.columns = wp.array(transposed_columns, dtype=int, device=device)
        transposed_values = np.zeros(n * width, dtype=values.dtype)
        transposed_values[:n] = 1
        owner.GT.values = wp.array(transposed_values, dtype=storage, device=device)
        owner.GT.notify_nnz_changed(nnz=n * width)
        self.pattern_ready = self.ready

    def update(self, owner, A):
        device, n, width = A.device, self.n, self.width
        k = self.kernels
        self.prepare(A)
        capturing = device.is_capturing
        # Eagerly, a published pattern needs none of the pattern kernels at all.
        pending = capturing or not self.published
        if pending:
            wp.launch(k.zero, self.pattern_status.size, [self.pattern_status, self.ready], device=device)
            wp.launch(
                self.select,
                n,
                [
                    A.offsets,
                    None,
                    A.columns,
                    A.values,
                    self.scale,
                    A.scalar_type(self.tolerance),
                    A.scalar_type(self.floor),
                    self.support_columns,
                    self.support_values,
                    self.pattern_status,
                    self.ready,
                ],
                device=device,
            )
            wp.launch(
                self.sort,
                n,
                [self.support_columns, self.support_values, self.support_ranks, self.support_counts, self.ready],
                device=device,
            )
        wp.launch(
            self.refit,
            n,
            [
                A.offsets,
                None,
                A.columns,
                A.values,
                owner.G.offsets,
                self.support_columns,
                self.support_counts[1:],
                self.support_ranks,
                self.scale,
                A.scalar_type(self.floor),
                self.candidate,
                self.status,
            ],
            device=device,
        )
        if pending:
            self._publish(owner, device)
        wp.launch(
            k.commit,
            n * width,
            [
                self.candidate,
                self.status,
                self.transpose_entries,
                owner.GT.offsets,
                owner.G.values,
                owner.GT.values,
                n,
            ],
            device=device,
        )
        if not capturing:
            if self.raises:
                self.check()
            if not self.published:
                self.published = bool(self.ready.numpy()[0])

    def _publish(self, owner, device):
        """Publish the selected supports and build the transposed index, once.

        The counting arrays start at zero and are only touched by an update that completes
        this build, which also sets the ready flag, so they never need to be cleared.
        """
        n, width, k = self.n, self.width, self.kernels
        wp.launch(
            k.publish,
            n,
            [
                self.support_columns,
                self.support_counts,
                width,
                self.transpose_counts,
                self.ready,
                self.status,
                owner.G.columns,
                owner.G.row_counts,
            ],
            device=device,
        )
        wp.launch(k.scan, 1, [self.transpose_counts, owner.GT.offsets, self.ready, self.status], device=device)
        wp.launch(
            k.fill,
            n,
            [
                owner.G.columns,
                owner.G.row_counts,
                width,
                owner.GT.offsets,
                self.transpose_cursor,
                self.transpose_entries,
                self.ready,
                self.status,
            ],
            device=device,
        )
        wp.launch(
            k.sort_columns,
            n,
            [owner.GT.offsets, self.transpose_entries, width, owner.GT.columns, self.ready, self.status],
            device=device,
        )
        wp.launch(k.mark_ready, 1, [self.ready, self.status], device=device)


@lru_cache(None)
def _matvec_kernel(storage_type, dtype):
    @wp.kernel(enable_backward=False, module="unique")
    def mv(
        offsets: wp.array[int],
        row_counts: wp.array[int],
        columns: wp.array[int],
        values: wp.array[storage_type],
        x: wp.array[dtype],
        y: wp.array[dtype],
        z: wp.array[dtype],
        alpha: dtype,
        beta: dtype,
    ):
        row = wp.tid()
        result = dtype(0)
        if alpha != dtype(0):
            end = offsets[row + 1]
            if row_counts:
                end = offsets[row] + row_counts[row]
            for e in range(offsets[row], end):
                result += dtype(values[e]) * x[columns[e]]
            result *= alpha
        if beta != dtype(0):
            result += beta * y[row]
        z[row] = result

    return mv


def _matvec(a, x, y, z, alpha, beta):
    """Compute ``z = alpha * a @ x + beta * y`` for a scalar CSR factor, one thread per row."""
    wp.launch(
        _matvec_kernel(a.scalar_type, x.dtype),
        a.nrow,
        [a.offsets, a.row_counts, a.columns, a.values, x, y, z, x.dtype(alpha), x.dtype(beta)],
        device=a.device,
    )
