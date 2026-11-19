# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Products of one BSR matrix with a batch of vectors stored as rows of a 2D array."""

import functools
from typing import Any

import warp as wp
from warp._src.types import type_scalar_type

ALGORITHMS = ("auto", "loop", "rows", "columns")
"""Kernel strategies for :func:`warp.sparse.bsr_mv` with a batch of vectors."""

# Block shapes whose batched CUDA kernels were benchmarked; AUTO keeps other shapes on LOOP.
_TUNED_CUDA_BLOCKS = ((1, 1), (3, 3))


@wp.func
def copy_value(value: Any):
    return value


@wp.kernel(enable_backward=False, module="unique")
def transpose_rows(source: wp.array2d[Any], target: wp.array2d[Any]):
    i, j = wp.tid()
    # A padded shared stride avoids the worst transpose bank conflicts.
    tile = wp.tile_load(source, shape=(32, 33), offset=(i * 32, j * 32), storage="shared")
    square = wp.tile_view(tile, shape=(32, 32), offset=(0, 0))
    result = wp.tile_map(copy_value, wp.tile_transpose(square))
    wp.tile_store(target, result, offset=(j * 32, i * 32))


@wp.func
def multiply(a: Any, b: Any):
    return a * b


@wp.kernel(enable_backward=False, module="unique")
def transpose_axpby(
    source: wp.array2d[Any],
    y: wp.array2d[Any],
    z: wp.array2d[Any],
    alpha: wp.float64,
    beta: wp.float64,
):
    i, j = wp.tid()
    tile = wp.tile_load(source, shape=(32, 33), offset=(i * 32, j * 32), storage="shared")
    square = wp.tile_view(tile, shape=(32, 32), offset=(0, 0))
    result = wp.tile_map(copy_value, wp.tile_transpose(square))
    result = wp.tile_map(
        multiply,
        result,
        wp.tile_full(shape=(32, 32), value=source.dtype(alpha), dtype=source.dtype),
    )
    if beta != wp.float64(0.0):
        previous = wp.tile_load(y, shape=(32, 32), offset=(j * 32, i * 32))
        result += wp.tile_map(
            multiply,
            previous,
            wp.tile_full(shape=(32, 32), value=source.dtype(beta), dtype=source.dtype),
        )
    wp.tile_store(z, result, offset=(j * 32, i * 32))


def _block_size(block_type):
    """Return the scalar type, the side of a square block type (1 for a scalar), and whether it is a scalar."""
    scalar = type_scalar_type(block_type)
    is_scalar = block_type == scalar
    return scalar, (1 if is_scalar else block_type._shape_[0]), is_scalar


@functools.lru_cache(None)
def rows_kernel(block_type):
    """Return the CUDA kernel computing ``y[r] = A @ x[r]`` for up to four vectors per 32-thread block.

    Eight lanes share each block row for one vector; compact and padded storage differ only in
    where a row ends. Used for square blocks larger than one entry.
    """
    from warp._src.sparse import _bsr_row_end  # noqa: PLC0415 (sparse imports this module)

    scalar, block, _ = _block_size(block_type)
    vector = wp.types.vector(length=block, dtype=scalar)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        row_counts: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[block_type],
        x: wp.array2d[scalar],
        y: wp.array2d[scalar],
    ):
        group, row, lane = wp.tid()
        rhs, slot = group * 4 + lane // 8, lane % 8
        value = vector(scalar(0.0))
        if rhs < x.shape[0]:
            for e in range(offsets[row] + slot, _bsr_row_end(offsets, row_counts, row), 8):
                col = columns[e] * block
                xb = vector()
                for j in range(wp.static(block)):
                    xb[j] = x[rhs, col + j]
                value += values[e] * xb
        for i in range(wp.static(block)):
            total = wp.tile_sum(wp.tile_reshape(wp.tile(value[i]), shape=(4, 8)), axis=1)
            if rhs < x.shape[0] and slot == 0:
                y[rhs, row * block + i] = wp.tile_extract(total, lane // 8)

    return compute


@functools.lru_cache(None)
def columns_kernel(block_type, vector_scalar=None):
    """Return the CUDA kernel computing ``y[:, r] = A @ x[:, r]`` for up to eight transposed vectors per block.

    The vectors are columns of ``x`` and ``y``, so the eight lanes reading one matrix entry access
    adjacent memory; four lanes share each matrix row. A scalar ``block_type`` may differ from
    ``vector_scalar``, the scalar type of ``x`` and ``y``; entries are converted as they are read.
    """
    from warp._src.sparse import _bsr_row_end  # noqa: PLC0415 (sparse imports this module)

    scalar, block, is_scalar = _block_size(block_type)
    if vector_scalar is not None and vector_scalar != scalar:
        if not is_scalar:
            raise TypeError("Mixed-precision column products support scalar blocks only")
        scalar = vector_scalar
    vector = wp.types.vector(length=block, dtype=scalar)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        row_counts: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[block_type],
        x: wp.array2d[scalar],
        y: wp.array2d[scalar],
    ):
        group, row, lane = wp.tid()
        rhs, slot = group * 8 + lane % 8, lane // 8
        if wp.static(is_scalar):
            value = scalar(0.0)
            if rhs < x.shape[1]:
                for e in range(offsets[row] + slot, _bsr_row_end(offsets, row_counts, row), 4):
                    value += scalar(values[e]) * x[columns[e], rhs]
            total = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=0)
            if rhs < x.shape[1] and slot == 0:
                y[row, rhs] = wp.tile_extract(total, lane % 8)
        else:
            value = vector(scalar(0.0))
            if rhs < x.shape[1]:
                for e in range(offsets[row] + slot, _bsr_row_end(offsets, row_counts, row), 4):
                    col = columns[e] * block
                    xb = vector()
                    for j in range(wp.static(block)):
                        xb[j] = x[col + j, rhs]
                    value += values[e] * xb
            for i in range(wp.static(block)):
                total = wp.tile_sum(wp.tile_reshape(wp.tile(value[i]), shape=(4, 8)), axis=0)
                if rhs < x.shape[1] and slot == 0:
                    y[row * block + i, rhs] = wp.tile_extract(total, lane % 8)

    return compute


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
def thread_rows_kernel(scalar, block, value_scalar=None):
    """Return the kernel computing one entry of ``y[r] = A @ x[r]`` per thread, for any square block side.

    Threads are indexed by ``(vector, scalar row)`` and read only their row of each block from the
    matrix's scalar ``(slots, block, block)`` view. ``scalar`` is the type of ``x`` and ``y``; matrix
    entries of a different ``value_scalar`` are converted as they are read.
    """
    value_scalar = scalar if value_scalar is None else value_scalar
    from warp._src.sparse import _bsr_row_end  # noqa: PLC0415 (sparse imports this module)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        row_counts: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array3d[value_scalar],
        x: wp.array2d[scalar],
        y: wp.array2d[scalar],
    ):
        rhs, row = wp.tid()
        block_row = row // wp.static(block)
        sub = row - block_row * wp.static(block)
        value = scalar(0)
        for e in range(offsets[block_row], _bsr_row_end(offsets, row_counts, block_row)):
            col = columns[e] * wp.static(block)
            for j in range(wp.static(block)):
                value += scalar(values[e, sub, j]) * x[rhs, col + j]
        y[rhs, row] = value

    return compute


def supports(algorithm, A, device):
    """Return whether a row or column strategy can run ``A`` on ``device``."""
    if A.block_shape[0] != A.block_shape[1]:
        return False
    return device.is_cuda or algorithm != "columns"


def auto_strategy(block_shape, count, is_cuda):
    """Return the strategy ``"auto"`` selects for a supported matrix and at least two vectors.

    Args:
        block_shape: The matrix block shape.
        count: Batch size.
        is_cuda: Whether the product runs on CUDA.
    """
    # From a sweep over FEM and irregular matrices (asv/benchmarks/sparse/matrix_vector_batch.py).
    if not is_cuda or block_shape == (1, 1) or count <= 4:
        return "rows"
    return "columns"


def validate(algorithm):
    """Raise :class:`ValueError` unless ``algorithm`` names a batched product strategy."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported batched product algorithm: {algorithm!r}; expected one of {ALGORITHMS}")


def resolve(algorithm, A, count, device, transpose, requires_grad=False):
    """Return the concrete strategy for a batch of ``count`` vectors, validating explicit choices.

    The row and column kernels are forward-only, so ``"auto"`` selects ``"loop"`` when
    ``requires_grad`` is set, and an explicit ``"rows"`` or ``"columns"`` raises.
    """
    validate(algorithm)
    if algorithm == "auto":
        if transpose or requires_grad or count < 2 or not supports("rows", A, device):
            return "loop"
        if device.is_cuda and A.block_shape not in _TUNED_CUDA_BLOCKS:
            return "loop"
        return auto_strategy(A.block_shape, count, device.is_cuda)
    if algorithm == "loop":
        return algorithm
    if transpose:
        raise ValueError(f"algorithm={algorithm!r} does not support transpose; use 'loop'")
    if requires_grad:
        raise ValueError(
            f"algorithm={algorithm!r} is forward-only and cannot propagate gradients; use 'auto' or 'loop' "
            "for products recorded with gradients"
        )
    if not supports(algorithm, A, device):
        supported = "square blocks on CUDA" if algorithm == "columns" else "square blocks"
        raise ValueError(f"algorithm={algorithm!r} supports {supported}; got block shape {A.block_shape} on {device}")
    return algorithm


def scratch_size(algorithm, A, count):
    """Return the scalar scratch entries a strategy needs for ``count`` vectors."""
    if algorithm == "columns":
        return count * (A.shape[0] + A.shape[1])
    if algorithm == "rows":
        return count * A.shape[0]
    return 0


def product(algorithm, A, x, y, z, alpha, beta, scratch):
    """Compute ``z[r] = alpha * A @ x[r] + beta * y[r]`` for each row with a row or column strategy.

    ``x``, ``y`` and ``z`` are contiguous 2D scalar arrays that may alias exactly; ``scratch`` is
    a flat scalar array of at least :func:`scratch_size` entries.
    """
    device = A.values.device
    count, rows, cols = x.shape[0], A.shape[0], A.shape[1]
    if alpha == 0:
        # Leave x unread, so uninitialized or nonfinite inputs cannot reach z.
        wp.launch(
            _combine, z.size, [y.flatten(), y.flatten(), z.flatten(), wp.float64(0), wp.float64(beta)], device=device
        )
        return
    # A None row_counts reaches the kernels as a null array: compact storage.
    matrix = [A.offsets, A.row_counts, A.columns, A.values]
    if algorithm == "columns":
        first = scratch[: cols * count].reshape((cols, count))
        second = scratch[cols * count : (cols + rows) * count].reshape((rows, count))
        wp.launch_tiled(
            transpose_rows,
            dim=((count + 31) // 32, (cols + 31) // 32),
            inputs=[x, first],
            block_dim=128,
            device=device,
        )
        wp.launch_tiled(
            columns_kernel(A.values.dtype),
            dim=((count + 7) // 8, A.nrow),
            inputs=[*matrix, first, second],
            block_dim=32,
            device=device,
        )
        wp.launch_tiled(
            transpose_axpby,
            dim=((rows + 31) // 32, (count + 31) // 32),
            inputs=[second, y, z, wp.float64(alpha), wp.float64(beta)],
            block_dim=128,
            device=device,
        )
        return
    direct = alpha == 1 and beta == 0 and x.ptr != z.ptr
    output = z if direct else scratch[: count * rows].reshape((count, rows))
    if device.is_cuda and A.block_shape != (1, 1) and count > 2:
        # Groups of lanes sharing a block row beat one thread per scalar row beyond two vectors.
        wp.launch_tiled(
            rows_kernel(A.values.dtype),
            dim=((count + 3) // 4, A.nrow),
            inputs=[*matrix, x, output],
            block_dim=32,
            device=device,
        )
    else:
        wp.launch(
            thread_rows_kernel(A.scalar_type, A.block_shape[0]),
            (count, rows),
            [A.offsets, A.row_counts, A.columns, A.scalar_values, x, output],
            device=device,
        )
    if not direct:
        wp.launch(
            _combine,
            y.size,
            [output.flatten(), y.flatten(), z.flatten(), wp.float64(alpha), wp.float64(beta)],
            device=device,
        )
