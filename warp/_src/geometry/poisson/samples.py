# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU input normalization and deterministic leaf aggregation.

This is the first hierarchy stage, not a finished adaptive reconstruction tree.
All bulk computation uses Warp; two tiny host status reads are measured explicitly.
"""

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import warp as wp
import warp.utils

from .field import basis_key

wp.set_module_options({"fuse_fp": False})


@wp.func
def point_div(v: wp.vec3, s: float):
    # Preserve the reference component-wise division order explicitly.
    return wp.vec3(v[0] / s, v[1] / s, v[2] / s)


@wp.func
def point_div(v: wp.vec3d, s: wp.float64):
    return wp.vec3d(v[0] / s, v[1] / s, v[2] / s)


@lru_cache(None)
def precision_kernels(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)

    @wp.kernel
    def _init_bounds_kernel(bounds: wp.array[dtype]):
        i = wp.tid()
        bounds[i] = dtype(wp.inf)
        bounds[i + 3] = -dtype(wp.inf)

    @wp.kernel
    def _bounds_kernel(
        points: wp.array[vec3],
        bounds: wp.array[dtype],
        status: wp.array[int],
    ):
        p = points[wp.tid()]
        if wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]):
            for a in range(3):
                wp.atomic_min(bounds, a, p[a])
                wp.atomic_max(bounds, a + 3, p[a])
        else:
            wp.atomic_add(status, 0, 1)

    @wp.kernel
    def _transform_kernel(
        bounds: wp.array[dtype],
        scale: dtype,
        transform: wp.array[dtype],
        status: wp.array[int],
    ):
        width = (
            wp.max(
                bounds[3] - bounds[0],
                wp.max(bounds[4] - bounds[1], bounds[5] - bounds[2]),
            )
            * scale
        )
        if not wp.isfinite(width) or width <= dtype(0.0):
            status[0] += 1
        else:
            inv = dtype(wp.float64(1.0) / wp.float64(width))
            transform[0] = inv
            determinant = inv * (inv * inv)
            normal_scale = (
                inv * inv / determinant * dtype(wp.pow(wp.float64(determinant), wp.float64(1.0) / wp.float64(3.0)))
            )
            transform[4] = normal_scale
            for a in range(3):
                axis = (a + 1) % 3
                origin = (bounds[axis + 3] + bounds[axis]) / dtype(2.0) - width / dtype(2.0)
                transform[a + 1] = -origin * inv

    @wp.kernel
    def _classify_kernel(
        points: wp.array[vec3],
        normals: wp.array[vec3],
        transform: wp.array[dtype],
        depth: int,
        q: wp.array[vec3],
        unit_normals: wp.array[vec3],
        keys: wp.array[wp.int64],
        order: wp.array[int],
        input_status: wp.array[int],
    ):
        i = wp.tid()
        p = points[i]
        n = normals[i]
        inv = transform[0]
        p = vec3(
            p[1] * inv + transform[1],
            p[2] * inv + transform[2],
            p[0] * inv + transform[3],
        )
        # Reference similarity-transform normals preserve the cofactor order.
        n = vec3(n[1], n[2], n[0]) * transform[4]
        length2 = wp.dot(n, n)
        q[i] = p
        order[i] = i
        keys[i] = wp.int64(9223372036854775807)
        unit_normals[i] = vec3(dtype(0.0))
        input_status[i] = 1  # Invalid normal; checked before unit-cube bounds.
        if length2 > dtype(0.0) and wp.isfinite(length2):
            input_status[i] = 2
        if length2 > dtype(0.0) and wp.isfinite(length2) and (wp.min(p) >= dtype(0.0)) and (wp.max(p) <= dtype(1.0)):
            input_status[i] = 0
            unit_normals[i] = point_div(n, wp.sqrt(length2))
            res = 1 << depth
            # ChildIndex uses > center: exact split planes belong below.
            x = wp.max(0, int(wp.ceil(p[0] * dtype(res))) - 1)
            y = wp.max(0, int(wp.ceil(p[1] * dtype(res))) - 1)
            z = wp.max(0, int(wp.ceil(p[2] * dtype(res))) - 1)
            keys[i] = basis_key(depth, x, y, z)

    @wp.kernel
    def _mark_kernel(keys: wp.array[wp.int64], n: int, marks: wp.array[int]):
        i = wp.tid()
        if keys[i] != wp.int64(9223372036854775807):
            if i == 0:
                marks[i] = 1
            elif keys[i] != keys[i - 1]:
                marks[i] = 1

    @wp.kernel
    def _count_kernel(prefix: wp.array[int], count: wp.array[int]):
        count[0] = prefix[prefix.shape[0] - 1]

    @wp.kernel
    def _groups_kernel(
        keys: wp.array[wp.int64],
        order: wp.array[int],
        marks: wp.array[int],
        prefix: wp.array[int],
        positions: wp.array[vec3],
        normals: wp.array[vec3],
        unique: wp.array[wp.int64],
        sums: wp.array[vec3],
        normal_sums: wp.array[vec3],
        weights: wp.array[dtype],
        first_indices: wp.array[int],
        input_group_indices: wp.array[int],
        group_offsets: wp.array[int],
    ):
        i = wp.tid()
        if marks[i] != 0:
            g = prefix[i] - 1
            p = vec3(dtype(0.0))
            n = vec3(dtype(0.0))
            weight = dtype(0.0)
            j = i
            # Stable radix sorting retains input order within each group.
            while j < marks.shape[0]:
                if keys[j] != keys[i]:
                    break
                p += positions[order[j]]
                n += normals[order[j]]
                weight += dtype(1.0)
                input_group_indices[order[j]] = g
                j += 1
            unique[g] = keys[i]
            sums[g] = p
            normal_sums[g] = n
            weights[g] = weight
            first_indices[g] = order[i]
            group_offsets[g] = i
            if g == unique.shape[0] - 1:
                group_offsets[g + 1] = j

    return SimpleNamespace(
        _init_bounds=_init_bounds_kernel,
        _bounds=_bounds_kernel,
        _transform=_transform_kernel,
        _classify=_classify_kernel,
        _mark=_mark_kernel,
        _count=_count_kernel,
        _groups=_groups_kernel,
    )


@dataclass
class SampleGroups:
    keys: wp.array
    position_sums: wp.array
    normal_sums: wp.array
    weights: wp.array
    transform: wp.array
    positions: wp.array
    normals: wp.array
    first_indices: wp.array
    host_sync_seconds: float
    host_status_reads: int = 2
    input_status: wp.array = None
    input_group_indices: wp.array = None
    input_order: wp.array = None
    group_offsets: wp.array = None


def prepare_samples(points, normals, depth=8, scale=1.1):
    """Finite float32/float64 positions, confidence off, reference alignment.

    Reject nonfinite positions and zero-extent input explicitly. Zero/nonfinite
    normals are omitted from grouping but their positions affect the bounding box,
    as in the reference. Coincident samples retain their multiplicity and order.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 15:
        raise ValueError("depth must be an integer in 0..15")
    if not math.isfinite(scale) or scale <= 1.0:
        raise ValueError("initial sample profile requires finite scale > 1")
    if not any(wp.types.types_equal(points.dtype, v) for v in (wp.vec3, wp.vec3d)) or not wp.types.types_equal(
        points.dtype, normals.dtype
    ):
        raise ValueError("sample processing requires matching float32/float64 3D vectors")
    if points.ndim != 1 or points.shape != normals.shape or not points.shape[0]:
        raise ValueError("points and normals must have equal nonempty 1D shapes")
    if points.device != normals.device or not points.device.is_cuda:
        raise ValueError("sample processing requires both arrays on the same CUDA device")
    device, count = points.device, points.shape[0]
    dtype = wp.float64 if wp.types.types_equal(points.dtype, wp.vec3d) else wp.float32
    kernels = precision_kernels(dtype)
    bounds = wp.empty(6, dtype=dtype, device=device)
    transform = wp.zeros(5, dtype=dtype, device=device)
    status = wp.zeros(1, dtype=int, device=device)
    wp.launch(kernels._init_bounds, 3, inputs=[bounds], device=device)
    wp.launch(kernels._bounds, count, inputs=[points, bounds, status], device=device)
    wp.launch(kernels._transform, 1, inputs=[bounds, scale, transform, status], device=device)
    t = time.perf_counter()
    invalid = int(status.numpy()[0])
    sync = time.perf_counter() - t
    if invalid:
        raise ValueError("Nonfinite positions or degenerate/overflowing bounding cube")
    q = wp.empty_like(points)
    ns = wp.empty_like(normals)
    keys = wp.empty(2 * count, dtype=wp.int64, device=device)
    order = wp.empty(2 * count, dtype=int, device=device)
    input_status = wp.empty(count, dtype=int, device=device)
    wp.launch(
        kernels._classify,
        count,
        inputs=[points, normals, transform, depth, q, ns, keys, order, input_status],
        device=device,
    )
    wp.utils.radix_sort_pairs(keys, order, count)
    marks = wp.zeros(count, dtype=int, device=device)
    prefix = wp.empty_like(marks)
    wp.launch(kernels._mark, count, inputs=[keys, count, marks], device=device)
    wp.utils.array_scan(marks, prefix, inclusive=True)
    wp.launch(kernels._count, 1, inputs=[prefix, status], device=device)
    t = time.perf_counter()
    groups = int(status.numpy()[0])
    sync += time.perf_counter() - t
    if not groups:
        raise ValueError("No accepted samples with finite nonzero normals")
    unique = wp.empty(groups, dtype=wp.int64, device=device)
    sums = wp.empty(groups, dtype=points.dtype, device=device)
    normal_sums = wp.empty_like(sums)
    weights = wp.empty(groups, dtype=dtype, device=device)
    first_indices = wp.empty(groups, dtype=int, device=device)
    input_group_indices = wp.full(count, -1, dtype=int, device=device)
    group_offsets = wp.empty(groups + 1, dtype=int, device=device)
    wp.launch(
        kernels._groups,
        count,
        inputs=[
            keys,
            order,
            marks,
            prefix,
            q,
            ns,
            unique,
            sums,
            normal_sums,
            weights,
            first_indices,
            input_group_indices,
            group_offsets,
        ],
        device=device,
    )
    return SampleGroups(
        unique,
        sums,
        normal_sums,
        weights,
        transform,
        q,
        ns,
        first_indices,
        sync,
        input_status=input_status,
        input_group_indices=input_group_indices,
        input_order=order[:count],
        group_offsets=group_offsets,
    )
