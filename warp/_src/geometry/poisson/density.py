# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference degree-2 sample density and splat-depth selection on sparse GPU levels."""

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from .field import basis_key, find_key
from .keyed import Records, coordinates, morton_order, reduce_records
from .samples import point_div

wp.set_module_options({"fuse_fp": False})


@wp.func
def quadratic(x: wp.float64):
    one = wp.float64(1.0)
    half = wp.float64(0.5)
    return wp.vec3d(
        (one - x) * (one - x) * half,
        ((one - x) * (x + one) + x * (wp.float64(2.0) - x)) * half,
        x * x * half,
    )


@lru_cache(None)
def precision_kernels(dtype):
    vec2 = wp.types.vector(length=2, dtype=dtype)
    vec3 = wp.types.vector(length=3, dtype=dtype)
    vec4 = wp.types.vector(length=4, dtype=dtype)

    @wp.func
    def real_log(value: dtype):
        # The reference log(Real) returns Real, before subsequent arithmetic.
        # Evaluate internally in double to avoid CUDA logf's larger ulp error.
        return dtype(wp.log(wp.float64(value)))

    @wp.kernel
    def _scale_kernel(out: wp.array[dtype]):
        # Reference averages 1000 values in the profile's scalar precision.
        # The hyperplane response is the sum of squared basis values in x.
        total = dtype(0.0)
        dx = dtype(0.1)
        for x in range(10):
            p = dx / dtype(2.0) + dx * dtype(x)
            b = quadratic(wp.float64(p))
            scale = dtype(wp.float64(1.0) / wp.dot(b, b))
            for _y in range(10):
                for _z in range(10):
                    total += scale
        out[0] = total / dtype(1000.0)

    @wp.kernel
    def _leaf_values_kernel(p: wp.array[vec3], w: wp.array[dtype], out: wp.array[vec4]):
        i = wp.tid()
        out[i] = vec4(p[i][0], p[i][1], p[i][2], w[i])

    @wp.kernel
    def _parents_kernel(
        keys: wp.array[wp.int64],
        values: wp.array[vec4],
        order: wp.array[int],
        depth: int,
        out_keys: wp.array[wp.int64],
        out_values: wp.array[vec4],
    ):
        i = wp.tid()
        j = order[i]
        c = coordinates(keys[j])
        out_keys[i] = basis_key(depth - 1, c[0] // 2, c[1] // 2, c[2] // 2)
        out_values[i] = values[j]

    @wp.kernel
    def _splat_kernel(
        keys: wp.array[wp.int64],
        values: wp.array[vec4],
        order: wp.array[int],
        depth: int,
        scale: wp.array[dtype],
        out_keys: wp.array[wp.int64],
        out_values: wp.array[vec4],
    ):
        i, k = wp.tid()
        j = order[i]
        c = coordinates(keys[j])
        v = values[j]
        n = 1 << depth
        p = point_div(vec3(v[0], v[1], v[2]), v[3])
        h = dtype(1.0) / dtype(n)
        t = (p - vec3(dtype(c[0]) * h, dtype(c[1]) * h, dtype(c[2]) * h)) / h
        bx, by, bz = (
            quadratic(wp.float64(t[0])),
            quadratic(wp.float64(t[1])),
            quadratic(wp.float64(t[2])),
        )
        x, y, z = (k // 9, k // 3 % 3, k % 3)
        target = c + wp.vec3i(x - 1, y - 1, z - 1)
        idx = i * 27 + k
        out_keys[idx] = wp.int64(9223372036854775807)
        out_values[idx] = vec4(dtype(0.0))
        if wp.min(target) >= 0 and wp.max(target) < n:
            weight = v[3] * scale[0]
            val = dtype(wp.float64(weight) * bx[x] * by[y] * bz[z])
            out_keys[idx] = basis_key(depth, target[0], target[1], target[2])
            out_values[idx] = vec4(val, dtype(0.0), dtype(0.0), dtype(0.0))

    @wp.func
    def density_value(
        keys: wp.array[wp.int64],
        values: wp.array[vec4],
        p: vec3,
        depth: int,
    ):
        n = 1 << depth
        c = wp.vec3i(
            wp.max(0, int(wp.ceil(p[0] * dtype(n))) - 1),
            wp.max(0, int(wp.ceil(p[1] * dtype(n))) - 1),
            wp.max(0, int(wp.ceil(p[2] * dtype(n))) - 1),
        )
        h = dtype(1.0) / dtype(n)
        t = (p - vec3(dtype(c[0]) * h, dtype(c[1]) * h, dtype(c[2]) * h)) / h
        bx, by, bz = (
            quadratic(wp.float64(t[0])),
            quadratic(wp.float64(t[1])),
            quadratic(wp.float64(t[2])),
        )
        value = dtype(0.0)
        for k in range(27):
            x, y, z = (k // 9, k // 3 % 3, k % 3)
            target = c + wp.vec3i(x - 1, y - 1, z - 1)
            if wp.min(target) >= 0 and wp.max(target) < n:
                idx = find_key(keys, basis_key(depth, target[0], target[1], target[2]))
                if idx >= 0:
                    value += dtype(bx[x] * by[y] * bz[z] * wp.float64(values[idx][0]))
        return value

    @wp.kernel
    def _evaluate_kernel(
        keys: wp.array[wp.int64],
        values: wp.array[vec4],
        sums: wp.array[vec3],
        weights: wp.array[dtype],
        depth: int,
        out: wp.array2d[dtype],
    ):
        i = wp.tid()
        out[i, depth] = density_value(keys, values, point_div(sums[i], weights[i]), depth)

    @wp.kernel
    def _depths_kernel(
        densities: wp.array2d[dtype],
        kernel_depth: int,
        target: dtype,
        out: wp.array[vec2],
    ):
        i = wp.tid()
        d = kernel_depth
        density = densities[i, d]
        raw = dtype(0.0)
        if density >= target:
            raw = dtype(
                wp.float64(d) + wp.float64(real_log(density / target)) / (wp.log(wp.float64(2.0)) * wp.float64(2.0))
            )
        else:
            fine = density
            while density < target and d > 0:
                fine = density
                d -= 1
                density = densities[i, d]
            raw = dtype(dtype(d) + real_log(density / target) / real_log(density / fine))
        h = dtype(1.0) / dtype(1 << d)
        weight = h * h / density
        out[i] = vec2(raw, weight)

    return SimpleNamespace(
        _scale=_scale_kernel,
        _leaf_values=_leaf_values_kernel,
        _parents=_parents_kernel,
        _splat=_splat_kernel,
        density_value=density_value,
        _evaluate=_evaluate_kernel,
        _depths=_depths_kernel,
    )


@dataclass
class Density:
    levels: list
    aggregates: list
    depths_and_weights: wp.array
    scale: wp.array
    sync_seconds: float
    host_status_reads: int


def build_density(samples, depth, kernel_depth=None, samples_per_node=1.5):
    if kernel_depth is None:
        kernel_depth = max(depth - 2, 0)
    if not 0 <= kernel_depth <= depth <= 15 or samples_per_node <= 0:
        raise ValueError("Require 0 <= kernel_depth <= depth <= 15 and positive samples_per_node")
    device = samples.keys.device
    dtype = samples.weights.dtype
    vec2 = wp.types.vector(length=2, dtype=dtype)
    vec4 = wp.types.vector(length=4, dtype=dtype)
    kernels = precision_kernels(dtype)
    count = len(samples.keys)
    values = wp.empty(count, dtype=vec4, device=device)
    wp.launch(
        kernels._leaf_values,
        count,
        inputs=[samples.position_sums, samples.weights, values],
        device=device,
    )
    aggregates = [None] * (depth + 1)
    aggregates[depth] = Records(samples.keys, values)
    sync = 0.0
    for d in range(depth, 0, -1):
        src = aggregates[d]
        order = morton_order(src.keys, d)
        keys = wp.empty(2 * len(src.keys), dtype=wp.int64, device=device)
        vals = wp.empty_like(src.values)
        wp.launch(
            kernels._parents,
            len(src.keys),
            inputs=[src.keys, src.values, order, d, keys, vals],
            device=device,
        )
        aggregates[d - 1] = reduce_records(keys, vals)
        sync += aggregates[d - 1].sync_seconds
    scale = wp.empty(1, dtype=dtype, device=device)
    wp.launch(kernels._scale, 1, inputs=[scale], device=device)
    levels = []
    densities = wp.empty((count, kernel_depth + 1), dtype=dtype, device=device)
    for d in range(kernel_depth + 1):
        src = aggregates[d]
        order = morton_order(src.keys, d)
        keys = wp.empty(2 * 27 * len(src.keys), dtype=wp.int64, device=device)
        vals = wp.empty(27 * len(src.keys), dtype=vec4, device=device)
        wp.launch(
            kernels._splat,
            (len(src.keys), 27),
            inputs=[src.keys, src.values, order, d, scale, keys, vals],
            device=device,
        )
        level = reduce_records(keys, vals)
        levels.append(level)
        sync += level.sync_seconds
        wp.launch(
            kernels._evaluate,
            count,
            inputs=[
                level.keys,
                level.values,
                samples.position_sums,
                samples.weights,
                d,
                densities,
            ],
            device=device,
        )
    dw = wp.empty(count, dtype=vec2, device=device)
    wp.launch(
        kernels._depths,
        count,
        inputs=[densities, kernel_depth, samples_per_node, dw],
        device=device,
    )
    return Density(levels, aggregates, dw, scale, sync, depth + kernel_depth + 1)
