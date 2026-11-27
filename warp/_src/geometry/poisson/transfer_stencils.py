# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sorted trilinear prolongation rows without triplet sorting or host counts."""

from functools import lru_cache

import warp as wp
import warp.sparse as sparse

from .field import basis_key, find_key
from .keyed import coordinates

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def transfer_kernels(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)

    @wp.func
    def entry(f: wp.vec3i, coarse: wp.array[wp.int64], depth: int, k: int):
        # Lexicographic coordinates give increasing canonical column indices.
        c = wp.vec3i(f[0] // 2, f[1] // 2, f[2] // 2) + wp.vec3i(k // 4, (k // 2) % 2, k % 2)
        h = vec3(dtype(1.0)) - wp.abs(vec3(f) * dtype(0.5) - vec3(c))
        weight = h[0] * h[1] * h[2]
        column = int(-1)
        if weight > dtype(0.0):
            column = find_key(coarse, basis_key(depth - 1, c[0], c[1], c[2]))
        return column, weight

    @wp.kernel
    def counts_kernel(fine: wp.array[wp.int64], coarse: wp.array[wp.int64], depth: int, out: wp.array[int]):
        i = wp.tid()
        f = coordinates(fine[i])
        count = int(0)
        for k in range(8):
            column, _weight = entry(f, coarse, depth, k)
            if column >= 0:
                count += 1
        out[i] = count

    @wp.kernel
    def fill_kernel(
        fine: wp.array[wp.int64],
        coarse: wp.array[wp.int64],
        depth: int,
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
    ):
        i = wp.tid()
        f = coordinates(fine[i])
        out = offsets[i]
        for k in range(8):
            column, weight = entry(f, coarse, depth, k)
            if column >= 0:
                columns[out] = column
                values[out] = weight
                out += 1

    return counts_kernel, fill_kernel


def assemble_prolongation(coarse_keys, fine_keys, depth, dtype):
    device = fine_keys.device
    n = len(fine_keys)
    matrix = sparse.bsr_zeros(n, len(coarse_keys), block_type=dtype, device=device)
    counts_kernel = wp.zeros(n + 1, dtype=int, device=device)
    count, fill_kernel = transfer_kernels(dtype)
    wp.launch(count, n, inputs=[fine_keys, coarse_keys, depth, counts_kernel], device=device)
    wp.utils.array_scan(counts_kernel, matrix.offsets, inclusive=False)
    # Warp permits nnz to be an upper bound; offsets retain the exact count.
    matrix.nnz = 8 * n
    matrix.columns = wp.empty(matrix.nnz, dtype=int, device=device)
    matrix.values = wp.empty(matrix.nnz, dtype=dtype, device=device)
    wp.launch(
        fill_kernel,
        n,
        inputs=[fine_keys, coarse_keys, depth, matrix.offsets, matrix.columns, matrix.values],
        device=device,
    )
    return matrix
