# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference exact and level-clustered point constraints, evaluated by PicQuadrature."""

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from .field import find_key
from .keyed import coordinates
from .samples import point_div
from .tree import tree_key

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def precision_kernels(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)
    vec4 = wp.types.vector(length=4, dtype=dtype)

    @wp.kernel
    def _points_kernel(
        keys: wp.array[wp.int64],
        data: wp.array[vec4],
        active: wp.array[wp.int64],
        average: wp.array[vec3],
        point_weight: dtype,
        depth: int,
        source_depth: int,
        positions: wp.array[vec3],
        weights: wp.array[dtype],
    ):
        i = wp.tid()
        v = data[i]
        positions[i] = point_div(vec3(v[0], v[1], v[2]), v[3])
        c = coordinates(keys[i])
        shift = source_depth - depth
        c = wp.vec3i(c[0] >> shift, c[1] >> shift, c[2] >> shift)
        w = v[3] * dtype(1 << source_depth)
        w *= point_weight * average[0][1]
        # Root padding moves the original root's record outside the valid
        # depth-zero space cell; the pinned oracle has no points at d=0.
        if depth == 0 or find_key(active, tree_key(c)) < 0:
            w = dtype(0.0)
        weights[i] = w

    return SimpleNamespace(_points=_points_kernel)


@dataclass
class PointConstraints:
    positions: wp.array
    weights: wp.array


def point_constraints(density, normals, tree, depth, point_weight, exact=False):
    source_depth = len(density.aggregates) - 1 if exact else depth
    src = density.aggregates[source_depth]
    device = src.keys.device
    dtype = density.scale.dtype
    kernel = precision_kernels(dtype)._points
    p = wp.empty(len(src.keys), dtype=normals.average.dtype, device=device)
    w = wp.empty(len(src.keys), dtype=dtype, device=device)
    wp.launch(
        kernel,
        len(src.keys),
        inputs=[
            src.keys,
            src.values,
            tree.levels[depth].keys,
            normals.average,
            point_weight,
            depth,
            source_depth,
            p,
            w,
        ],
        device=device,
    )
    if exact:
        # PoissonRecon 262b0f5 initializes exact ranges before SPACE_FLAG exists;
        # _isValidSpaceNode rejects every sample. Preserve its observable run
        # behavior, verified against the official --exact CLI.
        w.zero_()
    return PointConstraints(p, w)
