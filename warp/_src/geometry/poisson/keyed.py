# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic sparse records and Morton traversal, using Warp sort/scan."""

import time
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import warp as wp
import warp.utils

wp.set_module_options({"fuse_fp": False})


@wp.func
def coordinates(key: wp.int64):
    return wp.vec3i(
        int((key >> wp.int64(32)) & wp.int64(65535)),
        int((key >> wp.int64(16)) & wp.int64(65535)),
        int(key & wp.int64(65535)),
    )


@wp.func
def morton(c: wp.vec3i, depth: int):
    m = wp.int64(0)
    for bit in range(depth):
        for a in range(3):
            m |= wp.int64((c[a] >> bit) & 1) << wp.int64(3 * bit + a)
    return m


@wp.kernel
def _order_kernel(
    keys: wp.array[wp.int64],
    depth: int,
    codes: wp.array[wp.int64],
    ids: wp.array[int],
):
    i = wp.tid()
    codes[i] = morton(coordinates(keys[i]), depth)
    ids[i] = i


@wp.kernel
def _ids_kernel(ids: wp.array[int]):
    i = wp.tid()
    ids[i] = i


@wp.kernel
def _mark_kernel(keys: wp.array[wp.int64], marks: wp.array[int]):
    i = wp.tid()
    if keys[i] != wp.int64(0x7FFFFFFFFFFFFFFF):
        if i == 0:
            marks[i] = 1
        elif keys[i] != keys[i - 1]:
            marks[i] = 1


@wp.kernel
def _count_kernel(prefix: wp.array[int], out: wp.array[int]):
    out[0] = prefix[prefix.shape[0] - 1]


@lru_cache(None)
def precision_kernels(dtype):
    vec4 = wp.types.vector(length=4, dtype=dtype)

    @wp.kernel
    def _reduce_kernel(
        keys: wp.array[wp.int64],
        ids: wp.array[int],
        values: wp.array[vec4],
        marks: wp.array[int],
        prefix: wp.array[int],
        out_keys: wp.array[wp.int64],
        out_values: wp.array[vec4],
    ):
        i = wp.tid()
        if marks[i] != 0:
            total = vec4(dtype(0.0))
            j = i
            while j < marks.shape[0]:
                if keys[j] != keys[i]:
                    break
                total += values[ids[j]]
                j += 1
            g = prefix[i] - 1
            out_keys[g] = keys[i]
            out_values[g] = total

    return SimpleNamespace(_reduce=_reduce_kernel)


@dataclass
class Records:
    keys: wp.array
    values: wp.array
    sync_seconds: float = 0.0


def morton_order(keys, depth):
    n = len(keys)
    codes = wp.empty(2 * n, dtype=wp.int64, device=keys.device)
    ids = wp.empty(2 * n, dtype=int, device=keys.device)
    wp.launch(_order_kernel, n, inputs=[keys, depth, codes, ids], device=keys.device)
    wp.utils.radix_sort_pairs(codes, ids, n)
    return ids


def reduce_records(keys, values):
    """Keys have 2*N capacity; input order defines floating-point sum order."""
    n = len(values)
    device = keys.device
    dtype = wp.float64 if wp.types.types_equal(values.dtype, wp.vec4d) else wp.float32
    kernel = precision_kernels(dtype)._reduce
    if not n:
        return Records(
            wp.empty(0, dtype=wp.int64, device=device),
            wp.empty(0, dtype=values.dtype, device=device),
        )
    ids = wp.empty(2 * n, dtype=int, device=device)
    wp.launch(_ids_kernel, n, inputs=[ids], device=device)
    wp.utils.radix_sort_pairs(keys, ids, n)
    marks = wp.zeros(n, dtype=int, device=device)
    prefix = wp.empty_like(marks)
    count = wp.empty(1, dtype=int, device=device)
    wp.launch(_mark_kernel, n, inputs=[keys, marks], device=device)
    wp.utils.array_scan(marks, prefix, inclusive=True)
    wp.launch(_count_kernel, 1, inputs=[prefix, count], device=device)
    start = time.perf_counter()
    size = int(count.numpy()[0])
    sync = time.perf_counter() - start
    out_keys = wp.empty(size, dtype=wp.int64, device=device)
    out_values = wp.empty(size, dtype=values.dtype, device=device)
    wp.launch(
        kernel,
        n,
        inputs=[keys, ids, values, marks, prefix, out_keys, out_values],
        device=device,
    )
    return Records(out_keys, out_values, sync)


@wp.kernel
def _compact_unique_kernel(
    keys: wp.array[wp.int64],
    marks: wp.array[int],
    prefix: wp.array[int],
    out: wp.array[wp.int64],
):
    i = wp.tid()
    if marks[i] != 0:
        out[prefix[i] - 1] = keys[i]


def unique_keys(keys, count):
    """Set union without allocating/reducing a dummy vec4 for every candidate."""
    device = keys.device
    if not count:
        return Records(
            wp.empty(0, dtype=wp.int64, device=device),
            wp.empty(0, dtype=wp.vec4, device=device),
        )
    scratch = wp.zeros(2 * count, dtype=int, device=device)
    wp.utils.radix_sort_pairs(keys, scratch, count)
    marks = wp.zeros(count, dtype=int, device=device)
    prefix = wp.empty_like(marks)
    scalar = wp.empty(1, dtype=int, device=device)
    wp.launch(_mark_kernel, count, inputs=[keys, marks], device=device)
    wp.utils.array_scan(marks, prefix, inclusive=True)
    wp.launch(_count_kernel, 1, inputs=[prefix, scalar], device=device)
    start = time.perf_counter()
    size = int(scalar.numpy()[0])
    sync = time.perf_counter() - start
    out = wp.empty(size, dtype=wp.int64, device=device)
    wp.launch(_compact_unique_kernel, count, inputs=[keys, marks, prefix, out], device=device)
    return Records(out, wp.zeros(size, dtype=wp.vec4, device=device), sync)
