# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add clustered point forms directly to the verified 27-neighbor CSR stencil."""

from dataclasses import dataclass
from functools import lru_cache

import warp as wp

from .field import basis_key, find_key
from .keyed import coordinates

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def screening_kernel(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)
    row_values = wp.types.vector(length=27, dtype=wp.float64)

    @wp.kernel
    def bin_points_kernel(positions: wp.array[vec3], depth: int, keys: wp.array[wp.int64], ids: wp.array[int]):
        i = wp.tid()
        n = 1 << depth
        p = positions[i] * dtype(n)
        c = wp.vec3i(
            wp.clamp(int(wp.floor(p[0])), 0, n - 1),
            wp.clamp(int(wp.floor(p[1])), 0, n - 1),
            wp.clamp(int(wp.floor(p[2])), 0, n - 1),
        )
        keys[i] = basis_key(depth, c[0], c[1], c[2])
        ids[i] = i

    @wp.func
    def hat(p: vec3, c: wp.vec3i, depth: int):
        t = p * dtype(1 << depth) - vec3(c)
        h = wp.max(vec3(dtype(1.0)) - wp.abs(t), vec3(dtype(0.0)))
        return h[0] * h[1] * h[2]

    @wp.kernel
    def add_kernel(
        row_keys: wp.array[wp.int64],
        col_keys: wp.array[wp.int64],
        point_keys: wp.array[wp.int64],
        point_ids: wp.array[int],
        positions: wp.array[vec3],
        weights: wp.array[dtype],
        depth: int,
        trial_depth: int,
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[wp.float64],
        load: wp.array[wp.float64],
    ):
        i = wp.tid()
        c = coordinates(row_keys[i])
        accumulated = row_values(wp.float64(0.0))
        rhs = wp.float64(0.0)
        # Bin actual aggregate positions: rounding can move a centroid across
        # its original tree-cell boundary, so a bin may contain several points.
        # A hat sees only the eight cells touching its vertex.
        for k in range(8):
            cell = c - wp.vec3i(k // 4, (k // 2) % 2, k % 2)
            if wp.min(cell) >= 0 and wp.max(cell) < 1 << depth:
                key = basis_key(depth, cell[0], cell[1], cell[2])
                first = find_key(point_keys, key)
                if first >= 0:
                    j = first
                    while j < point_keys.shape[0]:
                        if point_keys[j] != key:
                            break
                        index = point_ids[j]
                        p = vec3(positions[index])
                        weight = dtype(weights[index])
                        if weight != dtype(0.0):
                            phi = hat(p, c, depth)
                            rhs += wp.float64(phi * weight)
                            for entry in range(offsets[i], offsets[i + 1]):
                                q = coordinates(col_keys[columns[entry]])
                                trial = hat(p, q, trial_depth)
                                accumulated[entry - offsets[i]] += wp.float64(phi * trial * weight)
                        j += 1
        load[i] = rhs
        for entry in range(offsets[i], offsets[i + 1]):
            values[entry] += accumulated[entry - offsets[i]]

    return bin_points_kernel, add_kernel


@dataclass
class PointBins:
    keys: wp.array
    ids: wp.array


def bin_constraints(points, depth):
    n = len(points.positions)
    device = points.positions.device
    keys = wp.empty(2 * n, dtype=wp.int64, device=device)
    ids = wp.empty(2 * n, dtype=int, device=device)
    wp.launch(screening_kernel(points.weights.dtype)[0], n, inputs=[points.positions, depth, keys, ids], device=device)
    wp.utils.radix_sort_pairs(keys, ids, n)
    return PointBins(keys[:n], ids[:n])


def add_screening_stencil(matrix, row_keys, col_keys, points, depth, trial_depth=None, bins=None):
    if trial_depth is None:
        trial_depth = depth
    device = row_keys.device
    if bins is None:
        bins = bin_constraints(points, depth)
    load = wp.empty(len(row_keys), dtype=wp.float64, device=device)
    wp.launch(
        screening_kernel(points.weights.dtype)[1],
        len(row_keys),
        inputs=[
            row_keys,
            col_keys,
            bins.keys,
            bins.ids,
            points.positions,
            points.weights,
            depth,
            trial_depth,
            matrix.offsets,
            matrix.columns,
            matrix.values,
            load,
        ],
        device=device,
    )
    return load
