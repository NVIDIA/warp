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

"""Tiled scalar CSR block products and layout conversions."""

from typing import Any

import warp as wp


@wp.func
def copy_value(value: Any):
    return value


@wp.kernel(module="unique")
def transpose_rows(source: wp.array2d[Any], target: wp.array2d[Any]):
    i, j = wp.tid()
    # A padded shared stride avoids the worst transpose bank conflicts.
    tile = wp.tile_load(source, shape=(32, 33), offset=(i * 32, j * 32), storage="shared")
    square = wp.tile_view(tile, shape=(32, 32), offset=(0, 0))
    result = wp.tile_map(copy_value, wp.tile_transpose(square))
    wp.tile_store(target, result, offset=(j * 32, i * 32))


@wp.kernel(module="unique")
def csr_eight_rhs(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[Any],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
):
    group, row, lane = wp.tid()
    rhs, slot = group * 8 + lane % 8, lane // 8
    value = x.dtype(0.0)
    if rhs < x.shape[1]:
        for e in range(offsets[row] + slot, offsets[row + 1], 4):
            value += x.dtype(values[e]) * x[columns[e], rhs]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=0)
    if rhs < x.shape[1] and slot == 0:
        y[row, rhs] = wp.tile_extract(totals, lane % 8)


@wp.func
def multiply(a: Any, b: Any):
    return a * b


@wp.kernel(module="unique")
def csr_eight_rhs_padded(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[Any],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
):
    group, row, lane = wp.tid()
    rhs, slot = group * 8 + lane % 8, lane // 8
    value = x.dtype(0.0)
    if rhs < x.shape[1]:
        for e in range(offsets[row] + slot, offsets[row] + counts[row], 4):
            value += x.dtype(values[e]) * x[columns[e], rhs]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=0)
    if rhs < x.shape[1] and slot == 0:
        y[row, rhs] = wp.tile_extract(totals, lane % 8)


@wp.kernel(module="unique")
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


@wp.kernel(module="unique")
def csr_four_rhs(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[Any],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
):
    group, row, lane = wp.tid()
    rhs, slot = group * 4 + lane // 8, lane % 8
    value = x.dtype(0.0)
    if rhs < x.shape[0]:
        for e in range(offsets[row] + slot, offsets[row + 1], 8):
            value += x.dtype(values[e]) * x[rhs, columns[e]]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=1)
    if rhs < x.shape[0] and slot == 0:
        y[rhs, row] = wp.tile_extract(totals, lane // 8)


@wp.kernel(module="unique")
def csr_four_rhs_padded(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[Any],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
):
    group, row, lane = wp.tid()
    rhs, slot = group * 4 + lane // 8, lane % 8
    value = x.dtype(0.0)
    if rhs < x.shape[0]:
        for e in range(offsets[row] + slot, offsets[row] + counts[row], 8):
            value += x.dtype(values[e]) * x[rhs, columns[e]]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=1)
    if rhs < x.shape[0] and slot == 0:
        y[rhs, row] = wp.tile_extract(totals, lane // 8)


def csr_block_product(matrix, x, y):
    if matrix.row_counts is not None:
        wp.launch_tiled(
            csr_four_rhs_padded,
            dim=((x.shape[0] + 3) // 4, matrix.nrow),
            inputs=[matrix.offsets, matrix.row_counts, matrix.columns, matrix.values, x, y],
            block_dim=32,
            device=matrix.device,
        )
        return
    wp.launch_tiled(
        csr_four_rhs,
        dim=((x.shape[0] + 3) // 4, matrix.nrow),
        inputs=[matrix.offsets, matrix.columns, matrix.values, x, y],
        block_dim=32,
        device=matrix.device,
    )
