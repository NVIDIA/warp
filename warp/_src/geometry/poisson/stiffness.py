# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact degree-one stiffness stencils, verified against assembled Warp FEM.

Each row has at most 27 overlapping functions at its own or the previous level.
Construct sorted CSR rows directly, without cell triplets or a global sort.
"""

import warp as wp
import warp.sparse as sparse

from .field import basis_key, find_key
from .keyed import coordinates

wp.set_module_options({"fuse_fp": False})


@wp.func
def hat_integrals(depth: int, index: int, trial_depth: int, trial_index: int):
    resolution = 1 << depth
    ratio = 1 << (depth - trial_depth)
    first = wp.max(0, wp.max(index - 1, (trial_index - 1) * ratio))
    end = wp.min(resolution, wp.min(index + 1, (trial_index + 1) * ratio))
    result = wp.vec2d(wp.float64(0.0))
    for cell in range(first, end):
        t0 = wp.float64(cell - index)
        c0 = wp.float64(cell) / wp.float64(ratio) - wp.float64(trial_index)
        ts = wp.float64(1.0)
        if cell >= index:
            ts = wp.float64(-1.0)
        cs = wp.float64(1.0) / wp.float64(ratio)
        if cell >= trial_index * ratio:
            cs = -cs
        tv = wp.float64(1.0) - wp.abs(t0)
        cv = wp.float64(1.0) - wp.abs(c0)
        result[0] += (tv * cv + (tv * cs + ts * cv) * wp.float64(0.5) + ts * cs / wp.float64(3.0)) / wp.float64(
            resolution
        )
        result[1] += ts * cs * wp.float64(resolution)
    return result


@wp.kernel
def _neighbors_kernel(
    row_keys: wp.array[wp.int64],
    col_keys: wp.array[wp.int64],
    depth: int,
    trial_depth: int,
    columns: wp.array2d[int],
    positions: wp.array2d[wp.uint8],
    counts: wp.array[int],
):
    i = wp.tid()
    c = coordinates(row_keys[i])
    shift = depth - trial_depth
    center = wp.vec3i(c[0] >> shift, c[1] >> shift, c[2] >> shift)
    count = int(0)
    for k in range(27):
        q = center + wp.vec3i(k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
        index = int(-1)
        if wp.min(q) >= 0 and wp.max(q) <= 1 << trial_depth:
            index = find_key(col_keys, basis_key(trial_depth, q[0], q[1], q[2]))
        columns[i, k] = index
        positions[i, k] = wp.uint8(count)
        if index >= 0:
            count += 1
    counts[i] = count


@wp.kernel
def _integrals_kernel(depth: int, trial_depth: int, table: wp.array2d[wp.vec2d]):
    i, k = wp.tid()
    trial = (i >> (depth - trial_depth)) + k - 1
    table[i, k] = hat_integrals(depth, i, trial_depth, trial)


@wp.kernel
def _fill_kernel(
    row_keys: wp.array[wp.int64],
    col_keys: wp.array[wp.int64],
    depth: int,
    trial_depth: int,
    candidates: wp.array2d[int],
    positions: wp.array2d[wp.uint8],
    table: wp.array2d[wp.vec2d],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.float64],
):
    i, k = wp.tid()
    c = coordinates(row_keys[i])
    j = candidates[i, k]
    if j >= 0:
        out = offsets[i] + int(positions[i, k])
        x = table[c[0], k // 9]
        y = table[c[1], (k // 3) % 3]
        z = table[c[2], k % 3]
        values[out] = x[1] * y[0] * z[0] + x[0] * y[1] * z[0] + x[0] * y[0] * z[1]
        columns[out] = j


def assemble_level_stiffness(row_keys, col_keys, depth, trial_depth=None):
    """Same-level or fine-test/coarse-trial Neumann hat stiffness in float64."""
    if trial_depth is None:
        trial_depth = depth
    if depth < 0 or trial_depth < 0 or depth - trial_depth not in (0, 1):
        raise ValueError("Require the same or immediately coarser trial level")
    n, device = len(row_keys), row_keys.device
    matrix = sparse.bsr_zeros(n, len(col_keys), block_type=wp.float64, device=device)
    candidates = wp.empty((n, 27), dtype=int, device=device)
    positions = wp.empty((n, 27), dtype=wp.uint8, device=device)
    counts = wp.zeros(n + 1, dtype=int, device=device)
    wp.launch(
        _neighbors_kernel,
        n,
        inputs=[row_keys, col_keys, depth, trial_depth, candidates, positions, counts],
        device=device,
    )
    wp.utils.array_scan(counts, matrix.offsets, inclusive=False)
    # Warp's nnz is an upper bound; row offsets carry exact device-side counts.
    matrix.nnz = 27 * n
    matrix.columns = wp.empty(matrix.nnz, dtype=int, device=device)
    matrix.values = wp.empty(matrix.nnz, dtype=wp.float64, device=device)
    table = wp.empty(((1 << depth) + 1, 3), dtype=wp.vec2d, device=device)
    wp.launch(_integrals_kernel, table.shape, inputs=[depth, trial_depth, table], device=device)
    # Neighboring lanes write neighboring CSR entries rather than striding
    # through 27-entry rows. The row-local rank is known during neighbor search.
    wp.launch(
        _fill_kernel,
        (n, 27),
        inputs=[
            row_keys,
            col_keys,
            depth,
            trial_depth,
            candidates,
            positions,
            table,
            matrix.offsets,
            matrix.columns,
            matrix.values,
        ],
        device=device,
    )
    return matrix
