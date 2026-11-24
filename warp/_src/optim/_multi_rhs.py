# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeat one linear operator over independent right-hand sides."""

import copy
from operator import index
from typing import Any, Literal

import warp as wp
from warp._src import sparse as sp
from warp._src import sparse_multi_rhs as batch
from warp._src.optim.linear import LinearOperator, aslinearoperator
from warp._src.types import type_length


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


def _scalar_shape(operator):
    """Return an operator's shape in scalar entries; vector-valued diagonals count every component."""
    if len(operator.shape) == 1:
        return (operator.shape[0] * type_length(operator.dtype),)
    return operator.shape


@wp.kernel(enable_backward=False)
def _offsets(offsets: wp.array[int], n: int):
    i = wp.tid()
    offsets[i] = i * n


class MultiRHSOperator(LinearOperator):
    """Repeat one sparse matrix or linear operator over independent RHS rows.

    This is a logical block-diagonal operator, with no duplicated matrix storage.
    Inputs to ``matvec`` are contiguous flat scalar arrays representing ``(nrhs, n)``.
    Sparse matrices use the batched :func:`warp.sparse.bsr_mv` strategies; other
    operators are applied to each RHS row in turn. Exact aliases are supported;
    shifted output overlap and strided storage are rejected before launching.

    Each object owns its scratch. ``with_nrhs`` creates smaller capacities sharing
    it. Eager applications issued from different CUDA streams are ordered with an
    event; simultaneously executing graphs that share scratch require caller
    ordering. Value updates in existing buffers are observed on replay;
    topology/address changes require recapture. No adjoint is supplied.

    Args:
        A: Square sparse matrix or a single-system :class:`LinearOperator`.
        nrhs: Positive RHS count.
        algorithm: Batched product strategy for a sparse ``A``, fixed at
            construction and kept by ``with_nrhs``: ``"auto"``, ``"loop"``,
            ``"rows"`` or ``"columns"``, as in :func:`warp.sparse.bsr_mv`. ``"auto"``
            reads back the mean row length once, so unlike per-call
            :func:`warp.sparse.bsr_mv` it can also choose ``"columns"`` for CSR rows
            averaging more than 16 entries. Explicit strategies pin a kernel and
            raise :class:`ValueError` where unsupported. Other operators accept only
            ``"auto"`` or ``"loop"``. The chosen strategy is available as
            ``algorithm``.
    """

    def __init__(self, A, nrhs: int, algorithm: Literal["auto", "loop", "rows", "columns"] = "auto"):
        if isinstance(nrhs, bool):
            raise TypeError("nrhs must be an integer, not bool")
        if index(nrhs) < 1:
            raise ValueError("nrhs must be a positive integer")
        batch.validate(algorithm)
        self.source = A
        self.single = aslinearoperator(A)
        self.n, self.nrhs = _scalar_shape(self.single)[0], index(nrhs)
        if (
            self.n < 1
            or _scalar_shape(self.single) not in ((self.n,), (self.n, self.n))
            or self.single.batch_offsets is not None
        ):
            raise ValueError("Expected one nonempty square system")
        dtype = self.single.scalar_type
        if dtype not in (wp.float32, wp.float64):
            raise ValueError("A must have float32 or float64 scalar type")
        device = self.single.device
        self.sparse = isinstance(A, sp.BsrMatrix)
        if self.sparse:
            mean_row_count = None
            if algorithm == "auto" and device.is_cuda and not device.is_capturing:
                # Preparation may read back the row lengths once, which per-call AUTO cannot.
                mean_row_count = A.mean_row_count_sync()
            self.algorithm = batch.resolve(algorithm, A, self.nrhs, device, False, mean_row_count=mean_row_count)
        elif algorithm in ("auto", "loop"):
            self.algorithm = "loop"
        else:
            raise ValueError(f"algorithm={algorithm!r} requires a sparse matrix")
        size = self.n * self.nrhs
        if self.algorithm != "loop":
            size = max(size, batch.scratch_size(self.algorithm, A, self.nrhs))
        self._work = wp.empty(size, dtype=dtype, device=device)
        offsets = wp.empty(self.nrhs + 1, dtype=int, device=device)
        wp.launch(_offsets, self.nrhs + 1, [offsets, self.n], device=device)
        stream = device.stream if device.is_cuda else None
        if stream is not None and not stream.is_capturing:
            # Solver setup reads batch offsets before any matvec joins the stream ordering,
            # so finish writing them before another stream can use this operator.
            wp.synchronize_stream(stream)
        self._stream_order = _StreamOrder(device)
        count = self.n * self.nrhs
        super().__init__(
            (count, count),
            dtype,
            device,
            self._apply,
            batch_offsets=offsets,
            max_batch_length=self.n,
        )

    def with_nrhs(self, nrhs: int):
        """Prepare a smaller capacity sharing scratch, strategy and stream ordering.

        Args:
            nrhs: Positive capacity no greater than this operator's ``nrhs``.

        Returns:
            A :class:`MultiRHSOperator` acting on ``nrhs`` RHS rows.
        """
        if isinstance(nrhs, bool):
            raise TypeError("nrhs must be an integer, not bool")
        if not 1 <= index(nrhs) <= self.nrhs:
            raise ValueError("nrhs must be positive and at most the prepared capacity")
        result = copy.copy(self)
        result.nrhs = index(nrhs)
        count = result.nrhs * self.n
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
        shape = (self.nrhs, self.n)
        if alpha == 0:
            wp.launch(_combine, z.size, [x, y, z, wp.float64(0), wp.float64(beta)], device=self.device)
            return
        if self.algorithm != "loop":
            batch.product(
                self.algorithm,
                self.source,
                x.reshape(shape),
                y.reshape(shape),
                z.reshape(shape),
                float(alpha),
                float(beta),
                self._work,
            )
            return
        first = self._work[: self.n * self.nrhs]
        for j in range(self.nrhs):
            sl = slice(j * self.n, (j + 1) * self.n)
            self.single.matvec(x[sl], first[sl], first[sl], 1.0, 0.0)
        wp.launch(_combine, z.size, [first, y, z, wp.float64(alpha), wp.float64(beta)], device=self.device)


@wp.kernel(enable_backward=False)
def _combine(x: wp.array[Any], y: wp.array[Any], z: wp.array[Any], alpha: wp.float64, beta: wp.float64):
    i = wp.tid()
    v = z.dtype(0)
    if alpha != wp.float64(0):
        v = z.dtype(alpha) * x[i]
    if beta != wp.float64(0):
        v += z.dtype(beta) * y[i]
    z[i] = v
