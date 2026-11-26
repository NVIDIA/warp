# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay continuous sample preparation with fixed accepted/group assignments.

Differentiate sample normalization and grouped oriented inputs.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from .samples import SampleGroups, point_div
from .samples import precision_kernels as sample_kernels

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def precision_kernels(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)

    @wp.kernel
    def extrema_kernel(points: wp.array[vec3], bounds: wp.array[dtype], owners: wp.array[int]):
        i = wp.tid()
        for a in range(3):
            if points[i][a] == bounds[a]:
                wp.atomic_min(owners, a, i)
            if points[i][a] == bounds[a + 3]:
                wp.atomic_min(owners, a + 3, i)

    @wp.kernel
    def gather_bounds_kernel(points: wp.array[vec3], owners: wp.array[int], bounds: wp.array[dtype]):
        i = wp.tid()
        bounds[i] = points[owners[i]][i % 3]

    @wp.kernel
    def normalize_kernel(
        points: wp.array[vec3],
        normals: wp.array[vec3],
        transform: wp.array[dtype],
        accepted: wp.array[int],
        positions: wp.array[vec3],
        unit_normals: wp.array[vec3],
    ):
        i = wp.tid()
        p = points[i]
        inv = transform[0]
        positions[i] = vec3(p[1] * inv + transform[1], p[2] * inv + transform[2], p[0] * inv + transform[3])
        normal = vec3(dtype(0))
        # Do not evaluate invalid-normal arithmetic: 0*NaN in an otherwise
        # unused automatic adjoint could poison transform gradients.
        if accepted[i] == 0:
            n = normals[i]
            n = vec3(n[1], n[2], n[0]) * transform[4]
            normal = point_div(n, wp.sqrt(wp.dot(n, n)))
        unit_normals[i] = normal

    @wp.kernel
    def sums_kernel(
        order: wp.array[int],
        offsets: wp.array[int],
        points: wp.array[vec3],
        normals: wp.array[vec3],
        out_points: wp.array[vec3],
        out_normals: wp.array[vec3],
    ):
        i = wp.tid()
        p = vec3(dtype(0))
        n = vec3(dtype(0))
        for j in range(offsets[i], offsets[i + 1]):
            p += points[order[j]]
            n += normals[order[j]]
        out_points[i] = p
        out_normals[i] = n

    return SimpleNamespace(
        extrema=extrema_kernel, gather_bounds=gather_bounds_kernel, normalize=normalize_kernel, sums=sums_kernel
    )


def replay_samples(points, normals, frozen, *, depth, scale=1.1):
    """Return tracked continuous values for a fixed ``SampleGroups`` layout.

    The original forward must already have accepted these finite inputs. Reuse
    its leaf grouping, input order, and accepted-normal decisions. Perturbations
    used to validate this map must stay away from assignment/normal-validity and
    longest-axis changes. At equal extrema, select the lowest input index as the
    piecewise bound derivative; it is not a derivative of changing tie choices.
    Inputs and replay outputs must remain immutable until backward completes.
    """
    if (
        points.shape != normals.shape
        or points.dtype != normals.dtype
        or points.dtype not in (wp.vec3, wp.vec3d)
        or points.ndim != 1
        or points.device != normals.device
        or frozen.keys.device != points.device
        or len(points) != len(frozen.input_status)
    ):
        raise ValueError("sample replay requires matching vectors and a frozen layout on the same device")
    dtype = wp.float64 if points.dtype == wp.vec3d else wp.float32
    device = points.device
    original = sample_kernels(dtype)
    kernels = precision_kernels(dtype)
    bounds = wp.empty(6, dtype=dtype, device=device)
    status = wp.zeros(1, dtype=int, device=device)
    owners = wp.full(6, 2**31 - 1, dtype=int, device=device)
    wp.launch(original._init_bounds, 3, inputs=[bounds], device=device, record_tape=False)
    wp.launch(original._bounds, len(points), inputs=[points, bounds, status], device=device, record_tape=False)
    wp.launch(kernels.extrema, len(points), inputs=[points, bounds, owners], device=device, record_tape=False)
    tracked_bounds = wp.empty_like(bounds, requires_grad=True)
    wp.launch(kernels.gather_bounds, 6, inputs=[points, owners, tracked_bounds], device=device)
    transform = wp.zeros(5, dtype=dtype, device=device, requires_grad=True)
    wp.launch(original._transform, 1, inputs=[tracked_bounds, dtype(scale), transform, status], device=device)
    positions = wp.empty_like(points, requires_grad=True)
    unit_normals = wp.empty_like(normals, requires_grad=True)
    wp.launch(
        kernels.normalize,
        len(points),
        inputs=[points, normals, transform, frozen.input_status, positions, unit_normals],
        device=device,
    )
    position_sums = wp.empty(len(frozen.keys), dtype=points.dtype, device=device, requires_grad=True)
    normal_sums = wp.empty_like(position_sums, requires_grad=True)
    wp.launch(
        kernels.sums,
        len(frozen.keys),
        inputs=[frozen.input_order, frozen.group_offsets, positions, unit_normals, position_sums, normal_sums],
        device=device,
    )
    return SampleGroups(
        frozen.keys,
        position_sums,
        normal_sums,
        frozen.weights,
        transform,
        positions,
        unit_normals,
        frozen.first_indices,
        0.0,
        host_status_reads=0,
        input_status=frozen.input_status,
        input_group_indices=frozen.input_group_indices,
        input_order=frozen.input_order,
        group_offsets=frozen.group_offsets,
    )
