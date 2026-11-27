# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracked density/normal splat replay with the primal sparse layout fixed."""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from . import keyed
from .density import Density
from .density import precision_kernels as density_kernels
from .normals import NormalField
from .normals import precision_kernels as normal_kernels

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _negative_kernel(src: wp.array[wp.vec4], dst: wp.array[wp.vec4]):
    i = wp.tid()
    dst[i] = -src[i]


@wp.kernel
def _negative_double_kernel(src: wp.array[wp.vec4d], dst: wp.array[wp.vec4d]):
    i = wp.tid()
    dst[i] = -src[i]


@lru_cache(None)
def precision_kernels(dtype):
    vec2 = wp.vec2d if dtype == wp.float64 else wp.vec2
    vec3 = wp.vec3d if dtype == wp.float64 else wp.vec3

    @wp.func
    def real_log(value: dtype):
        return dtype(wp.log(wp.float64(value)))

    @wp.kernel
    def select_depth_kernel(densities: wp.array2d[dtype], kernel_depth: int, target: dtype, selected: wp.array[int]):
        i = wp.tid()
        d = kernel_depth
        while densities[i, d] < target and d > 0:
            d -= 1
        selected[i] = d

    @wp.kernel
    def depth_values_kernel(
        densities: wp.array2d[dtype], selected: wp.array[int], kernel_depth: int, target: dtype, out: wp.array[vec2]
    ):
        i = wp.tid()
        d = selected[i]
        density = densities[i, d]
        raw = dtype(0)
        if densities[i, kernel_depth] >= target:
            raw = dtype(
                wp.float64(d) + wp.float64(real_log(density / target)) / (wp.log(wp.float64(2)) * wp.float64(2))
            )
        else:
            fine = densities[i, wp.min(d + 1, kernel_depth)]
            raw = dtype(dtype(d) + real_log(density / target) / real_log(density / fine))
        h = dtype(1) / dtype(1 << d)
        out[i] = vec2(raw, h * h / density)

    @wp.kernel
    def average_backward_kernel(
        contributions: wp.array[vec3],
        average: wp.array[vec3],
        grad_average: wp.array[vec3],
        grad_contributions: wp.array[vec3],
    ):
        i = wp.tid()
        mean = average[0]
        if contributions[i][2] > dtype(0) and mean[2] > dtype(0):
            g = grad_average[0]
            dx = g[0] / mean[2]
            dy = g[1] / mean[2]
            dw = g[2] - (g[0] * mean[0] + g[1] * mean[1]) / mean[2]
            wp.atomic_add(grad_contributions, i, vec3(dx, dy, dw))

    return SimpleNamespace(
        select_depth=select_depth_kernel, depth_values=depth_values_kernel, average_backward=average_backward_kernel
    )


def _morton_order(keys, depth):
    n, device = len(keys), keys.device
    codes = wp.empty(2 * n, dtype=wp.int64, device=device)
    ids = wp.empty(2 * n, dtype=int, device=device)
    wp.launch(keyed._order_kernel, n, inputs=[keys, depth, codes, ids], device=device, record_tape=False)
    wp.utils.radix_sort_pairs(codes, ids, n)
    return ids


def _reduce(keys, values, count):
    """Retain original stable serial sums; known primal count avoids a readback.

    Only use inside a fixed-layout replay. A changed key set is outside this
    low-level function's contract, and must be checked by its caller/tests.
    """
    n, device = len(values), values.device
    dtype = wp.float64 if values.dtype == wp.vec4d else wp.float32
    ids = wp.empty(2 * n, dtype=int, device=device)
    wp.launch(keyed._ids_kernel, n, inputs=[ids], device=device, record_tape=False)
    wp.utils.radix_sort_pairs(keys, ids, n)
    marks = wp.zeros(n, dtype=int, device=device)
    prefix = wp.empty_like(marks)
    wp.launch(keyed._mark_kernel, n, inputs=[keys, marks], device=device, record_tape=False)
    wp.utils.array_scan(marks, prefix, inclusive=True)
    out_keys = wp.empty(count, dtype=wp.int64, device=device)
    out_values = wp.empty(count, dtype=values.dtype, device=device, requires_grad=True)
    wp.launch(
        keyed.precision_kernels(dtype)._reduce,
        n,
        inputs=[keys, ids, values, marks, prefix, out_keys, out_values],
        device=device,
    )
    return keyed.Records(out_keys, out_values)


def replay_density(samples, frozen, *, samples_per_node=1.5):
    """Replay the primal density hierarchy with tracked continuous values."""
    depth = len(frozen.aggregates) - 1
    kernel_depth = len(frozen.levels) - 1
    dtype, device = samples.weights.dtype, samples.keys.device
    vec2 = wp.vec2d if dtype == wp.float64 else wp.vec2
    vec4 = wp.vec4d if dtype == wp.float64 else wp.vec4
    kernels = density_kernels(dtype)
    count = len(samples.keys)
    values = wp.empty(count, dtype=vec4, device=device, requires_grad=True)
    wp.launch(kernels._leaf_values, count, inputs=[samples.position_sums, samples.weights, values], device=device)
    aggregates = [None] * (depth + 1)
    aggregates[depth] = keyed.Records(samples.keys, values)
    for d in range(depth, 0, -1):
        src = aggregates[d]
        order = _morton_order(src.keys, d)
        keys = wp.empty(2 * len(src.keys), dtype=wp.int64, device=device)
        vals = wp.empty_like(src.values, requires_grad=True)
        wp.launch(kernels._parents, len(src.keys), inputs=[src.keys, src.values, order, d, keys, vals], device=device)
        aggregates[d - 1] = _reduce(keys, vals, len(frozen.aggregates[d - 1].keys))
    scale = frozen.scale  # Quadratic response normalization is a fixed constant.
    levels = []
    densities = wp.empty((count, kernel_depth + 1), dtype=dtype, device=device, requires_grad=True)
    for d in range(kernel_depth + 1):
        src = aggregates[d]
        order = _morton_order(src.keys, d)
        keys = wp.empty(2 * 27 * len(src.keys), dtype=wp.int64, device=device)
        vals = wp.empty(27 * len(src.keys), dtype=vec4, device=device, requires_grad=True)
        wp.launch(
            kernels._splat,
            (len(src.keys), 27),
            inputs=[src.keys, src.values, order, d, scale, keys, vals],
            device=device,
        )
        level = _reduce(keys, vals, len(frozen.levels[d].keys))
        levels.append(level)
        wp.launch(
            kernels._evaluate,
            count,
            inputs=[level.keys, level.values, samples.position_sums, samples.weights, d, densities],
            device=device,
        )
    dw = wp.empty(count, dtype=vec2, device=device, requires_grad=True)
    selected = wp.empty(count, dtype=int, device=device)
    replay = precision_kernels(dtype)
    # The search mutates its loop-carried density/fine values. Freeze only its
    # discrete stopping index, then differentiate immutable density reads.
    wp.launch(
        replay.select_depth,
        count,
        inputs=[densities, kernel_depth, dtype(samples_per_node), selected],
        device=device,
        record_tape=False,
    )
    wp.launch(
        replay.depth_values,
        count,
        inputs=[densities, selected, kernel_depth, dtype(samples_per_node), dw],
        device=device,
    )
    return Density(levels, aggregates, dw, scale, 0.0, 0)


def replay_normals(samples, density, frozen, *, tape, minimum, maximum):
    """Replay normal splats and their average with fixed sparse level sizes."""
    dtype, device = samples.weights.dtype, samples.keys.device
    vec4 = wp.vec4d if dtype == wp.float64 else wp.vec4
    kernels = normal_kernels(dtype)
    count = len(samples.keys)
    sortkeys = wp.empty(2 * count, dtype=int, device=device)
    order = wp.empty_like(sortkeys)
    wp.launch(
        kernels._input_order, count, inputs=[samples.first_indices, sortkeys, order], device=device, record_tape=False
    )
    wp.utils.radix_sort_pairs(sortkeys, order, count)
    keys = wp.empty(2 * 54 * count, dtype=wp.int64, device=device)
    vals = wp.empty(54 * count, dtype=vec4, device=device, requires_grad=True)
    wp.launch(
        kernels._splats,
        (count, 54),
        inputs=[
            samples.position_sums,
            samples.normal_sums,
            samples.weights,
            density.depths_and_weights,
            order,
            minimum,
            maximum,
            keys,
            vals,
        ],
        device=device,
    )
    summed = _reduce(keys, vals, len(frozen.records.keys))
    negative = wp.empty_like(summed.values, requires_grad=True)
    wp.launch(
        _negative_double_kernel if dtype == wp.float64 else _negative_kernel,
        len(negative),
        inputs=[summed.values, negative],
        device=device,
    )
    records = keyed.Records(summed.keys, negative)
    contributions = wp.empty(count, dtype=samples.positions.dtype, device=device, requires_grad=True)
    average = wp.empty(1, dtype=samples.positions.dtype, device=device, requires_grad=True)
    wp.launch(
        kernels._average_contributions,
        count,
        inputs=[samples.normal_sums, samples.weights, density.depths_and_weights, order, contributions],
        device=device,
    )
    wp.launch_tiled(
        kernels._average_tiled, dim=1, inputs=[contributions, average], block_dim=32, device=device, record_tape=False
    )

    # The serial tiled accumulator is intentionally a custom VJP: automatic
    # reversal of its mutable loop accumulator loses contributions.
    def backward_average():
        wp.launch(
            precision_kernels(dtype).average_backward,
            count,
            inputs=[contributions, average, average.grad, contributions.grad],
            device=device,
            record_tape=False,
        )

    tape.record_func(backward_average, [contributions, average])
    levels = []
    start = 0
    for level in frozen.levels:
        size = len(level.keys)
        if size:
            levels.append(keyed.Records(records.keys[start : start + size], records.values[start : start + size]))
        else:
            levels.append(
                keyed.Records(
                    wp.empty(0, dtype=wp.int64, device=device),
                    wp.empty(0, dtype=vec4, device=device, requires_grad=True),
                )
            )
        start += size
    return NormalField(records, average, levels)
