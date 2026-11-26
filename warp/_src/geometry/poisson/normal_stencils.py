# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-product normal loads, checked against the public FEM quadrature path."""

from functools import lru_cache

import warp as wp

from .field import basis_key, find_key
from .keyed import coordinates
from .ordered_normal_loads import normal_integral

wp.set_module_options({"fuse_fp": False})


@wp.func
def normal_center(index: int, test_depth: int, normal_depth: int):
    center = index
    if test_depth > normal_depth:
        center = index >> 1
    elif test_depth < normal_depth:
        center = index << 1
    return center


@wp.kernel
def _integrals_kernel(test_depth: int, normal_depth: int, first: int, table: wp.array2d[wp.vec2d]):
    i, k = wp.tid()
    j = normal_center(i, test_depth, normal_depth) + first + k
    table[i, k] = normal_integral(test_depth, i, normal_depth, j)


@lru_cache(None)
def load_kernel(vector):
    @wp.kernel
    def load_kernel(
        keys: wp.array[wp.int64],
        normal_keys: wp.array[wp.int64],
        normals: wp.array[vector],
        test_depth: int,
        normal_depth: int,
        first: int,
        width: int,
        table: wp.array2d[wp.vec2d],
        out: wp.array[wp.float64],
    ):
        i = wp.tid()
        c = coordinates(keys[i])
        center = wp.vec3i(
            normal_center(c[0], test_depth, normal_depth),
            normal_center(c[1], test_depth, normal_depth),
            normal_center(c[2], test_depth, normal_depth),
        )
        value = wp.float64(0.0)
        for k in range(width * width * width):
            x, y, z = k // (width * width), (k // width) % width, k % width
            n = center + wp.vec3i(x + first, y + first, z + first)
            if wp.min(n) >= 0 and wp.max(n) < 1 << normal_depth:
                j = find_key(normal_keys, basis_key(normal_depth, n[0], n[1], n[2]))
                if j >= 0:
                    ix, iy, iz = table[c[0], x], table[c[1], y], table[c[2], z]
                    normal = normals[j]
                    value += wp.float64(normal[0]) * ix[1] * iy[0] * iz[0]
                    value += wp.float64(normal[1]) * ix[0] * iy[1] * iz[0]
                    value += wp.float64(normal[2]) * ix[0] * iy[0] * iz[1]
        out[i] = value

    return load_kernel


def assemble_normal_stencil(keys, records, test_depth, normal_depth):
    if abs(test_depth - normal_depth) > 1:
        raise ValueError("Normal stencils require adjacent levels")
    device = keys.device
    out = wp.zeros(len(keys), dtype=wp.float64, device=device)
    if not len(records.keys):
        return out
    width, first = (6, -3) if test_depth < normal_depth else (4, -2)
    table = wp.empty(((1 << test_depth) + 1, width), dtype=wp.vec2d, device=device)
    wp.launch(_integrals_kernel, table.shape, inputs=[test_depth, normal_depth, first, table], device=device)
    wp.launch(
        load_kernel(records.values.dtype),
        len(keys),
        inputs=[keys, records.keys, records.values, test_depth, normal_depth, first, width, table, out],
        device=device,
    )
    return out
