# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the reference quadratic Dirichlet vector field for FEM quadrature."""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp
import warp.fem as fem

from .density import quadratic
from .field import basis_key, find_key
from .forms import value


@lru_cache(None)
def precision_kernels(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)
    vec4 = wp.types.vector(length=4, dtype=dtype)

    @wp.func
    def normal_value(
        p: vec3,
        keys: wp.array[wp.int64],
        values: wp.array[vec4],
        depth: int,
        first_depth: int,
    ):
        value = vec3(dtype(0.0))
        for d in range(first_depth, depth + 1):
            n = 1 << d
            q = p * dtype(n)
            c = wp.vec3i(
                wp.min(int(wp.floor(q[0])), n - 1),
                wp.min(int(wp.floor(q[1])), n - 1),
                wp.min(int(wp.floor(q[2])), n - 1),
            )
            t = q - vec3(c)
            bx, by, bz = (
                quadratic(wp.float64(t[0])),
                quadratic(wp.float64(t[1])),
                quadratic(wp.float64(t[2])),
            )
            for k in range(27):
                x, y, z = (k // 9, k // 3 % 3, k % 3)
                source = c + wp.vec3i(x - 1, y - 1, z - 1)
                sign = dtype(1.0)
                for axis in range(3):
                    if source[axis] < 0:
                        source[axis] = -1 - source[axis]
                        sign = -sign
                    elif source[axis] >= n:
                        source[axis] = 2 * n - 1 - source[axis]
                        sign = -sign
                idx = find_key(keys, basis_key(d, source[0], source[1], source[2]))
                if idx >= 0:
                    v = values[idx]
                    value += vec3(v[0], v[1], v[2]) * dtype(bx[x] * by[y] * bz[z]) * sign
        return value

    @wp.func
    def _cached_normal(index: int, values: wp.array[vec3]):
        return values[index]

    return SimpleNamespace(normal_value=normal_value, _cached_normal=_cached_normal)


@fem.integrand
def normal_load(s: fem.Sample, v: fem.Field, normal: fem.Field):
    return wp.dot(fem.grad(v, s), normal(s))


def cache_normal_quadrature(geometry, normals, depth, first_depth=0):
    domain = fem.Cells(geometry)
    quadrature = fem.RegularQuadrature(domain, order=3)
    normal = fem.ImplicitField(
        domain,
        precision_kernels(domain.geometry.scalar_type).normal_value,
        values={
            "keys": normals.records.keys,
            "values": normals.records.values,
            "depth": depth,
            "first_depth": first_depth,
        },
        degree=2,
    )
    data = wp.empty(
        quadrature.total_point_count(),
        dtype=normals.average.dtype,
        device=normals.records.keys.device,
    )
    fem.interpolate(value, dest=data, at=quadrature, fields={"u": normal})
    return quadrature, fem.ImplicitField(
        domain,
        precision_kernels(geometry.scalar_type)._cached_normal,
        values={"values": data},
        degree=2,
    )


def assemble_normal_load(space, normals, depth, cache=None, first_depth=0):
    domain = fem.Cells(space.geometry)
    if cache is None:
        quadrature = fem.RegularQuadrature(domain, order=3)
        normal = fem.ImplicitField(
            domain,
            precision_kernels(domain.geometry.scalar_type).normal_value,
            values={
                "keys": normals.records.keys,
                "values": normals.records.values,
                "depth": depth,
                "first_depth": first_depth,
            },
            degree=2,
        )
    else:
        quadrature, normal = cache
    v = fem.make_test(space, domain=domain)
    return fem.integrate(
        normal_load,
        quadrature=quadrature,
        fields={"v": v, "normal": normal},
        output_dtype=wp.float64,
        device=normals.records.keys.device,
    )
