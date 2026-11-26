# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic polynomial normal loads for cancellation-sensitive float32 roots.

The general FEM load path remains the baseline. This path evaluates the same
piecewise-polynomial integrals and preserves the pinned serial accumulation order.
"""

from functools import lru_cache

import warp as wp

from .field import basis_key, find_key
from .keyed import coordinates

wp.set_module_options({"fuse_fp": False})


@wp.func
def quadratic_piece(t0: wp.float64, step: wp.float64, offset: int):
    b = wp.vec3d(wp.float64(0.0))
    if offset == -1:
        b = wp.vec3d(
            wp.float64(0.5) * (wp.float64(1.0) - t0) * (wp.float64(1.0) - t0),
            -(wp.float64(1.0) - t0) * step,
            wp.float64(0.5) * step * step,
        )
    elif offset == 0:
        b = wp.vec3d(
            wp.float64(0.5) + t0 - t0 * t0,
            (wp.float64(1.0) - wp.float64(2.0) * t0) * step,
            -step * step,
        )
    elif offset == 1:
        b = wp.vec3d(wp.float64(0.5) * t0 * t0, t0 * step, wp.float64(0.5) * step * step)
    return b


@wp.func
def normal_integral(test_depth: int, test_index: int, normal_depth: int, normal_index: int):
    """Integrals of a Neumann hat (and its derivative) against a Dirichlet B2."""
    nt = 1 << test_depth
    nn = 1 << normal_depth
    resolution = wp.max(nt, nn)
    test_step = wp.float64(nt) / wp.float64(resolution)
    normal_step = wp.float64(nn) / wp.float64(resolution)
    first = wp.max(
        0,
        wp.max(
            (test_index - 1) * (resolution // nt),
            (normal_index - 1) * (resolution // nn),
        ),
    )
    end = wp.min(
        resolution,
        wp.min(
            (test_index + 1) * (resolution // nt),
            (normal_index + 2) * (resolution // nn),
        ),
    )
    out = wp.vec2d(wp.float64(0.0))
    for cell in range(first, end):
        tx = wp.float64(cell) * test_step - wp.float64(test_index)
        sign = wp.float64(-wp.float64(1.0))
        if tx + wp.float64(0.5) * test_step < wp.float64(0.0):
            sign = wp.float64(1.0)
        h0 = wp.float64(1.0) + sign * tx
        h1 = sign * test_step
        normal_cell = int(wp.floor((wp.float64(cell) + wp.float64(0.5)) * normal_step))
        t0 = wp.float64(cell) * normal_step - wp.float64(normal_cell)
        b = quadratic_piece(t0, normal_step, normal_index - normal_cell)
        # Odd reflection enforces the reference quadratic Dirichlet boundary.
        b -= quadratic_piece(t0, normal_step, -1 - normal_index - normal_cell)
        b -= quadratic_piece(t0, normal_step, 2 * nn - 1 - normal_index - normal_cell)
        out[0] += (
            h0 * b[0]
            + (h0 * b[1] + h1 * b[0]) * wp.float64(0.5)
            + (h0 * b[2] + h1 * b[1]) / wp.float64(3.0)
            + h1 * b[2] * wp.float64(0.25)
        ) / wp.float64(resolution)
        out[1] += (b[0] + b[1] * wp.float64(0.5) + b[2] / wp.float64(3.0)) * sign * test_step
    return out


@wp.func
def normal_stencil(test_depth: int, test: wp.vec3i, normal_depth: int, normal: wp.vec3i):
    x = normal_integral(test_depth, test[0], normal_depth, normal[0])
    y = normal_integral(test_depth, test[1], normal_depth, normal[1])
    z = normal_integral(test_depth, test[2], normal_depth, normal[2])
    return wp.vec3d(x[1] * y[0] * z[0], x[0] * y[1] * z[0], x[0] * y[0] * z[1])


@wp.func
def morton_xy(c: wp.vec3i):
    key = int(0)
    for bit in range(15):
        key |= ((c[0] >> bit) & 1) << (2 * bit)
        key |= ((c[1] >> bit) & 1) << (2 * bit + 1)
    return key


@wp.kernel
def parent_order_kernel(
    keys: wp.array[wp.int64],
    orders: wp.array[int],
    indices: wp.array[int],
):
    i, k = wp.tid()
    c = 2 * coordinates(keys[i]) + wp.vec3i(k % 6 - 3, (k // 6) % 6 - 3, k // 36 - 3)
    j = i * 216 + k
    orders[j] = 2147483647
    indices[j] = k
    if wp.min(c) >= 0:
        orders[j] = morton_xy(c)


@wp.kernel
def segment_offsets_kernel(offsets: wp.array[int]):
    i = wp.tid()
    offsets[i] = i * 36


@lru_cache(None)
def kernels(dtype):
    vec4 = wp.types.vector(length=4, dtype=dtype)

    @wp.func
    def contribution(
        test_depth: int,
        test: wp.vec3i,
        normal_depth: int,
        normal: wp.vec3i,
        keys: wp.array[wp.int64],
        values: wp.array[vec4],
    ):
        dot = dtype(0.0)
        if wp.min(normal) >= 0 and wp.max(normal) < 1 << normal_depth:
            j = find_key(keys, basis_key(normal_depth, normal[0], normal[1], normal[2]))
            if j >= 0:
                stencil = normal_stencil(test_depth, test, normal_depth, normal)
                for axis in range(3):
                    dot += values[j][axis] * dtype(stencil[axis])
        return dot

    @wp.kernel
    def same_kernel(
        keys: wp.array[wp.int64],
        normal_keys: wp.array[wp.int64],
        normals: wp.array[vec4],
        finer: wp.array[dtype],
        depth: int,
        out: wp.array[dtype],
    ):
        i = wp.tid()
        c = coordinates(keys[i])
        value = finer[i]
        for k in range(64):
            n = c + wp.vec3i(k // 16 - 2, (k // 4) % 4 - 2, k % 4 - 2)
            value += contribution(depth, c, depth, n, normal_keys, normals)
        out[i] = value

    @wp.kernel
    def parent_kernel(
        keys: wp.array[wp.int64],
        normal_keys: wp.array[wp.int64],
        normals: wp.array[vec4],
        order: wp.array[int],
        depth: int,
        out: wp.array[dtype],
    ):
        i = wp.tid()
        c = coordinates(keys[i])
        value = dtype(0.0)
        for j in range(216):
            k = order[i * 216 + j]
            n = 2 * c + wp.vec3i(k % 6 - 3, (k // 6) % 6 - 3, k // 36 - 3)
            value += contribution(depth, c, depth + 1, n, normal_keys, normals)
        out[i] = value

    @wp.kernel
    def restrict_kernel(
        fine_keys: wp.array[wp.int64],
        fine: wp.array[dtype],
        coarse_keys: wp.array[wp.int64],
        depth: int,
        out: wp.array[dtype],
    ):
        i = wp.tid()
        c = coordinates(coarse_keys[i])
        value = out[i]
        for k in range(27):
            off = wp.vec3i(k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
            n = 2 * c + off
            if wp.min(n) >= 0 and wp.max(n) <= 1 << depth:
                j = find_key(fine_keys, basis_key(depth, n[0], n[1], n[2]))
                if j >= 0:
                    weight = dtype(1.0)
                    for axis in range(3):
                        if off[axis] != 0:
                            weight *= dtype(0.5)
                    value += fine[j] * weight
        out[i] = value

    @wp.kernel
    def prolong_kernel(
        keys: wp.array[wp.int64],
        own_keys: wp.array[wp.int64],
        own: wp.array[vec4],
        previous_keys: wp.array[wp.int64],
        previous: wp.array[vec4],
        depth: int,
        out: wp.array[vec4],
    ):
        i = wp.tid()
        c = coordinates(keys[i])
        value = vec4(dtype(0.0))
        j = find_key(own_keys, keys[i])
        if j >= 0:
            value = own[j]
        if depth > 0:
            for k in range(8):
                p = wp.vec3i(
                    c[0] // 2 + (k >> 2),
                    c[1] // 2 + ((k >> 1) & 1),
                    c[2] // 2 + (k & 1),
                )
                weight = dtype(1.0)
                for axis in range(3):
                    weight *= dtype(1.0) - wp.abs(dtype(c[axis]) * dtype(0.5) - dtype(p[axis]))
                if weight > dtype(0.0):
                    j = find_key(previous_keys, basis_key(depth - 1, p[0], p[1], p[2]))
                    if j >= 0:
                        value += previous[j] * weight
        out[i] = value

    @wp.kernel
    def from_coarse_kernel(
        keys: wp.array[wp.int64],
        normal_keys: wp.array[wp.int64],
        normals: wp.array[vec4],
        depth: int,
        out: wp.array[dtype],
    ):
        i = wp.tid()
        c = coordinates(keys[i])
        value = dtype(0.0)
        for k in range(64):
            n = wp.vec3i(c[0] // 2, c[1] // 2, c[2] // 2) + wp.vec3i(k // 16 - 2, (k // 4) % 4 - 2, k % 4 - 2)
            value += contribution(depth, c, depth - 1, n, normal_keys, normals)
        out[i] += value

    return same_kernel, parent_kernel, restrict_kernel, prolong_kernel, from_coarse_kernel


def ordered_normal_loads(tree, normals, max_depth, base_depth=0):
    dtype = wp.float64 if wp.types.types_equal(normals.average.dtype, wp.vec3d) else wp.float32
    device = normals.records.keys.device
    same_kernel, parent_kernel, restrict_kernel, prolong_kernel, from_coarse_kernel = kernels(dtype)
    loads = [None] * (max_depth + 1)
    finer = wp.zeros(len(tree.basis[max_depth].keys), dtype=dtype, device=device)
    for d in range(max_depth, base_depth - 1, -1):
        keys = tree.basis[d].keys
        own = normals.levels[d] if normals.levels is not None else normals.records
        loads[d] = wp.empty_like(finer)
        wp.launch(
            same_kernel,
            len(keys),
            inputs=[keys, own.keys, own.values, finer, d, loads[d]],
            device=device,
        )
        if d > base_depth:
            coarse = tree.basis[d - 1].keys
            next_finer = wp.zeros(len(coarse), dtype=dtype, device=device)
            if len(own.keys):
                count = len(coarse) * 216
                orders = wp.empty(count * 2, dtype=int, device=device)
                indices = wp.empty(count * 2, dtype=int, device=device)
                segments = wp.empty(len(coarse) * 6 + 1, dtype=int, device=device)
                wp.launch(
                    parent_order_kernel,
                    (len(coarse), 216),
                    inputs=[coarse, orders, indices],
                    device=device,
                )
                wp.launch(segment_offsets_kernel, len(segments), inputs=[segments], device=device)
                wp.utils.segmented_sort_pairs(orders, indices, count, segments)
                wp.launch(
                    parent_kernel,
                    len(coarse),
                    inputs=[coarse, own.keys, own.values, indices, d - 1, next_finer],
                    device=device,
                )
            if d < max_depth:
                wp.launch(
                    restrict_kernel,
                    len(coarse),
                    inputs=[keys, finer, coarse, d, next_finer],
                    device=device,
                )
            finer = next_finer
    previous_keys = wp.empty(0, dtype=wp.int64, device=device)
    previous = wp.empty(0, dtype=normals.records.values.dtype, device=device)
    for d in range(max_depth):
        own = normals.levels[d] if normals.levels is not None else normals.records
        keys = tree.basis[d].keys
        cumulative = wp.empty(len(keys), dtype=normals.records.values.dtype, device=device)
        wp.launch(
            prolong_kernel,
            len(keys),
            inputs=[keys, own.keys, own.values, previous_keys, previous, d, cumulative],
            device=device,
        )
        if d + 1 >= base_depth:
            wp.launch(
                from_coarse_kernel,
                len(tree.basis[d + 1].keys),
                inputs=[tree.basis[d + 1].keys, keys, cumulative, d + 1, loads[d + 1]],
                device=device,
            )
        previous_keys, previous = keys, cumulative
    return loads
