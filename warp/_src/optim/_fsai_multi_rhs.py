# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared multiple-RHS application of FSAI factors.

No factor construction or iterative solver lives here.
"""

import copy
from operator import index
from typing import Any

import warp as wp
from warp._src.optim._multi_rhs_kernels import (
    csr_block_product,
    csr_eight_rhs,
    csr_eight_rhs_padded,
    transpose_axpby,
    transpose_rows,
)
from warp._src.optim.linear import LinearOperator


@wp.kernel(enable_backward=False)
def _offsets(offsets: wp.array[int], n: int):
    i = wp.tid()
    offsets[i] = i * n


@wp.kernel(enable_backward=False)
def _row_product(
    offsets: wp.array[int],
    counts: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    x: wp.array2d[Any],
    z: wp.array2d[Any],
):
    rhs, row = wp.tid()
    end = offsets[row + 1]
    if counts:
        end = offsets[row] + counts[row]
    value = x.dtype(0)
    for e in range(offsets[row], end):
        value += x.dtype(values[e]) * x[rhs, columns[e]]
    z[rhs, row] = value


@wp.kernel(enable_backward=False)
def _combine(x: wp.array[Any], y: wp.array[Any], z: wp.array[Any], alpha: wp.float64, beta: wp.float64):
    i = wp.tid()
    value = z.dtype(0)
    if alpha != wp.float64(0):
        value = z.dtype(alpha) * x[i]
    if beta != wp.float64(0):
        value += z.dtype(beta) * y[i]
    z[i] = value


class FSAIMultiRHS(LinearOperator):
    """Prepared multiple-RHS view of an existing FSAI preconditioner.

    Prefer :meth:`FSAI.as_multi_rhs` for construction and its complete application
    and lifetime contract. Each instance owns scratch; :meth:`with_nrhs` returns
    a smaller view sharing that scratch for serial execution. No new factors are
    built. This operator uses contiguous flat arrays representing RHS rows and
    exposes independent batch partitions to the existing linear solvers.

    Args:
        factor: Existing :class:`FSAI` whose buffers remain valid during use.
        nrhs: Positive RHS count.
        dtype: Working scalar dtype, float32 or float64. Defaults to the factor's
            scalar type, or float32 for float16 factors.
        layout: ``"columns"`` for coalesced CUDA products or ``"rows"`` for
            row-oriented products. CPU always uses the latter.
    """

    def __init__(self, factor, nrhs: int, dtype: type | None = None, layout: str = "columns"):
        from warp._src.optim.fsai import FSAI  # noqa: PLC0415

        if not isinstance(factor, FSAI):
            raise TypeError("factor must be an FSAI preconditioner")
        if isinstance(nrhs, bool):
            raise TypeError("nrhs must be an integer, not bool")
        if index(nrhs) < 1:
            raise ValueError("nrhs must be a positive integer")
        if dtype is None:
            dtype = wp.float32 if factor.scalar_type == wp.float16 else factor.scalar_type
        if dtype not in (wp.float32, wp.float64):
            raise ValueError("Working dtype must be float32 or float64")
        if layout not in ("rows", "columns"):
            raise ValueError("layout must be 'rows' or 'columns'")
        self.factor, self.nrhs, self.n = factor, index(nrhs), factor.G.nrow
        self.layout = layout if factor.device.is_cuda else "rows"
        shape = (self.n, self.nrhs) if self.layout == "columns" else (self.nrhs, self.n)
        self._first = wp.empty(shape, dtype=dtype, device=factor.device)
        self._second = wp.empty_like(self._first)
        offsets = wp.empty(self.nrhs + 1, dtype=int, device=factor.device)
        wp.launch(_offsets, self.nrhs + 1, [offsets, self.n], device=factor.device)
        self._owner = self
        # Seed with the constructing stream so first use elsewhere waits for setup.
        self._last_stream = factor.device.stream if factor.device.is_cuda else None
        self._event = wp.Event(device=factor.device) if factor.device.is_cuda else None
        super().__init__(
            (self.nrhs * self.n, self.nrhs * self.n),
            dtype,
            factor.device,
            self._apply,
            batch_offsets=offsets,
            max_batch_length=self.n,
        )

    def with_nrhs(self, nrhs: int) -> "FSAIMultiRHS":
        """Return a smaller prepared capacity sharing this operator's scratch.

        No factor or work array is copied. The operators must execute serially,
        including concurrently executing graphs; eager calls from different
        streams are ordered automatically. Construct views outside graph capture.
        """
        if isinstance(nrhs, bool):
            raise TypeError("nrhs must be an integer, not bool")
        if index(nrhs) < 1 or index(nrhs) > self.nrhs:
            raise ValueError("nrhs must be positive and no greater than the prepared capacity")
        result = copy.copy(self)
        result.nrhs = index(nrhs)
        shape = (self.n, result.nrhs) if self.layout == "columns" else (result.nrhs, self.n)
        count = result.nrhs * self.n
        result._first = self._first.flatten()[:count].reshape(shape)
        result._second = self._second.flatten()[:count].reshape(shape)
        result._shape = (count, count)
        result._batch_offsets = self.batch_offsets[: result.nrhs + 1]
        result._matvec = result._apply
        return result

    def _apply(self, x, y, z, alpha, beta):
        for a in (x, y, z):
            if a.device != self.device or a.dtype != self.dtype or a.shape != (self.nrhs * self.n,):
                raise ValueError("Expected flat arrays of nrhs*n elements with the prepared dtype and device")
            if not a.is_contiguous:
                raise ValueError("Multiple-RHS application requires contiguous arrays")
        # Exact aliases are safe; shifted overlapping output views are not.
        for a in (x, y):
            if a.ptr != z.ptr and max(a.ptr, z.ptr) < min(
                a.ptr + a.size * wp.types.type_size_in_bytes(a.dtype),
                z.ptr + z.size * wp.types.type_size_in_bytes(z.dtype),
            ):
                raise ValueError("Partially overlapping output views are unsupported")
        if self.device.is_cuda:
            stream, last = self.device.stream, self._owner._last_stream
            # Same-stream calls, the common case, skip the native capture query.
            if stream is not last and not stream.is_capturing:
                if last is not None and not last.is_capturing:
                    last.record_event(self._event)
                    stream.wait_event(self._event)
                self._owner._last_stream = stream
        if alpha == 0:
            wp.launch(_combine, z.size, [x, y, z, wp.float64(0), wp.float64(beta)], device=self.device)
            return
        if self.layout == "rows":
            source = x.reshape((self.nrhs, self.n))
            for factor, src, dst in ((self.factor.G, source, self._first), (self.factor.GT, self._first, self._second)):
                if self.device.is_cuda:
                    csr_block_product(factor, src, dst)
                else:
                    wp.launch(
                        _row_product,
                        (self.nrhs, self.n),
                        [factor.offsets, factor.row_counts, factor.columns, factor.values, src, dst],
                        device=self.device,
                    )
            wp.launch(
                _combine,
                z.size,
                [self._second.flatten(), y, z, wp.float64(alpha), wp.float64(beta)],
                device=self.device,
            )
            return
        wp.launch_tiled(
            transpose_rows,
            dim=((self.nrhs + 31) // 32, (self.n + 31) // 32),
            inputs=[x.reshape((self.nrhs, self.n)), self._first],
            block_dim=128,
            device=self.device,
        )
        for factor, source, result in (
            (self.factor.G, self._first, self._second),
            (self.factor.GT, self._second, self._first),
        ):
            inputs = [factor.offsets]
            if factor.row_counts is not None:
                inputs.append(factor.row_counts)
            inputs.extend([factor.columns, factor.values, source, result])
            wp.launch_tiled(
                csr_eight_rhs if factor.row_counts is None else csr_eight_rhs_padded,
                dim=((self.nrhs + 7) // 8, self.n),
                inputs=inputs,
                block_dim=32,
                device=self.device,
            )
        wp.launch_tiled(
            transpose_axpby,
            dim=((self.n + 31) // 32, (self.nrhs + 31) // 32),
            inputs=[
                self._first,
                y.reshape((self.nrhs, self.n)),
                z.reshape((self.nrhs, self.n)),
                wp.float64(alpha),
                wp.float64(beta),
            ],
            block_dim=128,
            device=self.device,
        )
