# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse GPU tree completion for the pinned degree-1/degree-2-normal profile.

Signed coordinates belong to per-level arrays. Padding is never packed into an
unsigned Morton code. The level keys below are for logical nodes, not field keys.
"""

import time
from dataclasses import dataclass
from typing import Any

import warp as wp
import warp.utils

from .field import basis_key
from .keyed import Records, coordinates, unique_keys

wp.set_module_options({"fuse_fp": False})


@wp.func
def tree_key(c: wp.vec3i):
    return (
        (wp.int64(c[0] + 524288) << wp.int64(40)) | (wp.int64(c[1] + 524288) << wp.int64(20)) | wp.int64(c[2] + 524288)
    )


@wp.func
def tree_coordinates(key: wp.int64):
    return wp.vec3i(
        int((key >> wp.int64(40)) & wp.int64(1048575)) - 524288,
        int((key >> wp.int64(20)) & wp.int64(1048575)) - 524288,
        int(key & wp.int64(1048575)) - 524288,
    )


@wp.func
def parent(c: wp.vec3i):
    return wp.vec3i(
        int(wp.floor(float(c[0]) / 2.0)), int(wp.floor(float(c[1]) / 2.0)), int(wp.floor(float(c[2]) / 2.0))
    )


@wp.kernel
def _coarse_kernel(d: int, count: int, lo: int, offset: int, keys: wp.array[wp.int64]):
    i = wp.tid()
    c = wp.vec3i(i // (count * count) + lo, (i // count) % count + lo, i % count + lo)
    keys[offset + i] = tree_key(c)


@wp.kernel
def _parent_keys_kernel(fine: wp.array[wp.int64], out: wp.array[wp.int64]):
    i = wp.tid()
    out[i] = tree_key(parent(tree_coordinates(fine[i])))


@wp.kernel
def _seed_broods_kernel(normal_keys: wp.array[wp.int64], normals: wp.array[Any], d: int, keys: wp.array[wp.int64]):
    i = wp.tid()
    keys[i] = wp.int64(0x7FFFFFFFFFFFFFFF)
    nd = int(normal_keys[i] >> wp.int64(48))
    if nd >= d and wp.float64(wp.length_sq(normals[i])) > wp.float64(0.0):
        c = coordinates(normal_keys[i])
        c = wp.vec3i(c[0] >> (nd - d), c[1] >> (nd - d), c[2] >> (nd - d))
        keys[i] = tree_key(parent(c))


@wp.kernel
def _support_broods_kernel(fine: wp.array[wp.int64], radius: int, offset: int, depth: int, keys: wp.array[wp.int64]):
    i, j = wp.tid()
    width = radius * 2 + 1
    c = parent(tree_coordinates(fine[i]))
    c += wp.vec3i(j // (width * width) - radius, (j // width) % width - radius, j % width - radius)
    idx = offset + i * width * width * width + j
    keys[idx] = wp.int64(0x7FFFFFFFFFFFFFFF)
    # The original child bounds [-2*n, 2*n) fall on brood boundaries.
    n = 1 << depth
    if wp.min(c) >= -n and wp.max(c) < n:
        keys[idx] = tree_key(c)


@wp.func
def _lower_bound(keys: wp.array[wp.int64], key: wp.int64):
    lo = int(0)
    hi = keys.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel
def _expand_broods_kernel(broods: wp.array[wp.int64], children: wp.array[wp.int64]):
    i = wp.tid()
    key = broods[i]
    c = 2 * tree_coordinates(key)
    x = (key >> wp.int64(40)) << wp.int64(40)
    xy = (key >> wp.int64(20)) << wp.int64(20)
    x0 = _lower_bound(broods, x)
    x1 = _lower_bound(broods, x + (wp.int64(1) << wp.int64(40)))
    xy0 = _lower_bound(broods, xy)
    xy1 = _lower_bound(broods, xy + (wp.int64(1) << wp.int64(20)))
    for k in range(8):
        ox = (k >> 2) & 1
        oy = (k >> 1) & 1
        oz = k & 1
        # Lexicographic children: two x halves, then two y halves, then z.
        # Prefix sizes come from the already sorted unique parent keys.
        dest = 8 * x0 + 4 * ox * (x1 - x0) + 4 * (xy0 - x0) + 2 * oy * (xy1 - xy0) + 2 * (i - xy0) + oz
        children[dest] = tree_key(c + wp.vec3i(ox, oy, oz))


def expand_broods(broods):
    """Expand unique sorted parents into unique sorted children without sorting."""
    children = wp.empty(8 * len(broods.keys), dtype=wp.int64, device=broods.keys.device)
    wp.launch(_expand_broods_kernel, len(broods.keys), inputs=[broods.keys, children], device=children.device)
    return Records(children, wp.zeros(len(children), dtype=wp.vec4, device=children.device))


@wp.kernel
def _basis_marks_kernel(keys: wp.array[wp.int64], d: int, marks: wp.array[int]):
    i = wp.tid()
    c = tree_coordinates(keys[i])
    marks[i] = 0
    if wp.min(c) >= 0 and wp.max(c) <= 1 << d:
        marks[i] = 1


@wp.kernel
def _basis_count_kernel(prefix: wp.array[int], out: wp.array[int]):
    out[0] = prefix[prefix.shape[0] - 1]


@wp.kernel
def _basis_fill_kernel(
    keys: wp.array[wp.int64], d: int, marks: wp.array[int], prefix: wp.array[int], out: wp.array[wp.int64]
):
    i = wp.tid()
    if marks[i]:
        c = tree_coordinates(keys[i])
        out[prefix[i] - 1] = basis_key(d, c[0], c[1], c[2])


def compact_basis(keys, depth):
    """The signed-to-field key map preserves lexicographic order in the cube."""
    device = keys.device
    n = len(keys)
    if not n:
        return Records(wp.empty(0, dtype=wp.int64, device=device), wp.empty(0, dtype=wp.vec4, device=device))
    marks = wp.empty(n, dtype=int, device=device)
    prefix = wp.empty_like(marks)
    count = wp.empty(1, dtype=int, device=device)
    wp.launch(_basis_marks_kernel, n, inputs=[keys, depth, marks], device=device)
    wp.utils.array_scan(marks, prefix, inclusive=True)
    wp.launch(_basis_count_kernel, 1, inputs=[prefix, count], device=device)
    start = time.perf_counter()
    size = int(count.numpy()[0])
    sync = time.perf_counter() - start
    out = wp.empty(size, dtype=wp.int64, device=device)
    wp.launch(_basis_fill_kernel, n, inputs=[keys, depth, marks, prefix, out], device=device)
    return Records(out, wp.zeros(size, dtype=wp.vec4, device=device), sync)


@dataclass
class LogicalTree:
    levels: list
    basis: list
    host_sync_seconds: float


def build_tree(normals, depth, full_depth, base_depth=None):
    if base_depth is None:
        base_depth = full_depth
    if not 0 <= base_depth <= full_depth <= depth <= 15:
        raise ValueError("Require 0 <= base_depth <= full_depth <= depth <= 15")
    device = normals.records.keys.device
    levels = [None] * (depth + 1)
    basis = [None] * (depth + 1)
    sync = 0.0
    requests = None
    broods = None
    for d in range(depth, -1, -1):
        # Finer normal broods are already represented by `parents` below.
        # Only seed this level's records instead of projecting every finer
        # normal and sorting the same ancestor broods again at every depth.
        records = normals.levels[d] if normals.levels is not None else normals.records
        count = len(records.keys)
        # Every level consists of complete eight-child broods. The unique
        # parents are therefore already available from the finer-level union.
        parents = (
            broods
            if broods is not None
            else Records(wp.empty(0, dtype=wp.int64, device=device), wp.empty(0, dtype=wp.vec4, device=device))
        )

        if d > base_depth:
            requests = parents
        elif requests is not None:
            rk = wp.empty(2 * len(requests.keys), dtype=wp.int64, device=device)
            wp.launch(_parent_keys_kernel, len(requests.keys), inputs=[requests.keys, rk], device=device)
            requests = unique_keys(rk, len(requests.keys))
            sync += requests.sync_seconds
        else:
            requests = parents
        coarse_count = (2 if d == 0 else (1 << (d - 1)) + 2) if d <= full_depth else 0
        nc = coarse_count**3
        ns = len(requests.keys) * 27
        np = len(parents.keys)
        total = count + nc + ns + np
        keys = wp.empty(2 * total, dtype=wp.int64, device=device)
        wp.launch(_seed_broods_kernel, count, inputs=[records.keys, records.values, d, keys], device=device)
        if nc:
            wp.launch(_coarse_kernel, nc, inputs=[d, coarse_count, -1, count, keys], device=device)
        if ns:
            wp.launch(
                _support_broods_kernel,
                (len(requests.keys), 27),
                inputs=[requests.keys, 1, count + nc, d, keys],
                device=device,
            )
        if np:
            wp.launch(
                _support_broods_kernel,
                (len(parents.keys), 1),
                inputs=[parents.keys, 0, count + nc + ns, d, keys],
                device=device,
            )
        broods = unique_keys(keys, total)
        sync += broods.sync_seconds
        levels[d] = expand_broods(broods)
        basis[d] = compact_basis(levels[d].keys, d)
        sync += basis[d].sync_seconds
    return LogicalTree(levels, basis, sync)
