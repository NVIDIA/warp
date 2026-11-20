# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-side code generation cost dominated by builtin overload resolution.

Every builtin call in a kernel is resolved against the builtin's overloads
(``mul`` has well over 100), so the Python half of a module build scales with
the number of builtin calls. These benchmarks time that half without running
a native compiler:

- ``time_codegen_scalar`` / ``time_codegen_vector`` generate the CPU source of
  a module with scalar or vector/matrix/quaternion arithmetic, as a module
  load does before compiling.
- ``time_declare_static_dispatch`` re-declares a ``module="unique"`` kernel
  that calls helpers through ``wp.static()`` in an unrolled loop. Such
  deferred static expressions are evaluated by building the module when its
  hash is computed, so every declaration pays for code generation.

Nothing is launched or compiled, so the benchmarks need no GPU.
"""

import warp as wp

from ..benchmarks_utils import setup_once


def scalar_arithmetic(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    c: wp.array(dtype=wp.float64),
    n: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
    out64: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    x = a[i]
    y = b[i]
    z = c[i]
    k = n[i]
    s = x * y + x / (y + 1.0) - wp.abs(x - y)
    s = wp.min(s, wp.max(x, y)) * wp.clamp(y, -1.0, 1.0)
    s = s + wp.sqrt(wp.abs(x)) * wp.floor(y) - wp.sign(x) * 0.5
    t = z * z + z / (z + wp.float64(1.0)) - wp.abs(z)
    t = wp.min(t, wp.max(z, wp.float64(0.0))) * wp.float64(2.0)
    m = k * 3 + k / 2 - k % 5
    m = wp.min(m, wp.max(k, 0)) + wp.abs(k - 7)
    w = wp.where(x > y, x * 2.0, y * 0.5) + float(m) * 0.25
    for _ in range(4):
        w = w * 0.9 + s * 0.1 - wp.float32(t) * 0.01
        s = wp.min(s * 1.1, w + 1.0)
    out[i] = w + s
    out64[i] = t + wp.float64(w)


def vector_arithmetic(
    p: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    m: wp.array(dtype=wp.mat33),
    tf: wp.array(dtype=wp.transform),
    out_v: wp.array(dtype=wp.vec3),
    out_m: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    v = p[i]
    r = q[i]
    a = m[i]
    t = tf[i]
    u = wp.quat_rotate(r, v) + v * 2.0 - wp.cross(v, wp.vec3(0.0, 0.0, 1.0))
    u = u / (wp.length(u) + 1.0e-6) + wp.normalize(v) * wp.dot(u, v)
    w = a * u + wp.transpose(a) * v - wp.outer(u, v) * v
    b = a * wp.transpose(a) + wp.outer(u, w) * 0.5 - wp.identity(n=3, dtype=float)
    b = b * wp.determinant(a) + wp.skew(u) * wp.skew(w)
    x = wp.transform_point(t, w) + wp.transform_vector(t, u)
    r2 = r * wp.quat_from_axis_angle(wp.normalize(x + wp.vec3(1.0)), 0.1)
    for _ in range(3):
        x = wp.quat_rotate_inv(r2, x) * 0.5 + w * 0.5
        b = b * 0.5 + wp.outer(x, x) * 0.1
    out_v[i] = x + b[0] - wp.cw_mul(u, w)
    out_m[i] = b


@wp.func
def pair_sphere_sphere(a: wp.vec3, b: wp.vec3, ra: float, rb: float) -> float:
    d = b - a
    return wp.length(d) - ra - rb


@wp.func
def pair_sphere_box(a: wp.vec3, b: wp.vec3, ra: float, rb: float) -> float:
    d = wp.vec3(wp.abs(b[0] - a[0]), wp.abs(b[1] - a[1]), wp.abs(b[2] - a[2])) - wp.vec3(rb)
    outside = wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0))
    return wp.length(outside) + wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0) - ra


@wp.func
def pair_capsule_capsule(a: wp.vec3, b: wp.vec3, ra: float, rb: float) -> float:
    axis = wp.vec3(0.0, 0.0, 1.0)
    d = b - a
    t = wp.clamp(wp.dot(d, axis), -1.0, 1.0)
    closest = a + axis * t
    return wp.length(b - closest) - ra - rb


@wp.func
def pair_plane_sphere(a: wp.vec3, b: wp.vec3, ra: float, rb: float) -> float:
    n = wp.normalize(a + wp.vec3(0.0, 0.0, 1.0e-6))
    return wp.dot(b - a, n) - rb + 0.0 * ra


PAIR_FUNCTIONS = (pair_sphere_sphere, pair_sphere_box, pair_capsule_capsule, pair_plane_sphere)


def static_dispatch(
    kinds: wp.array(dtype=wp.int32),
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.vec2),
    dist: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    r = radii[i]
    for k in range(wp.static(len(PAIR_FUNCTIONS))):
        if kinds[i] == k:
            dist[i] = wp.static(PAIR_FUNCTIONS[k])(a[i], b[i], r[0], r[1])


class BuiltinOverloadResolution:
    """Time the Python-side code generation of builtin-heavy kernels."""

    repeat = 15
    number = 5
    warmup_time = 0.2

    @setup_once
    def setup(self):
        wp.init()
        self._modules = {}
        for name, fn in (("scalar", scalar_arithmetic), ("vector", vector_arithmetic)):
            module = wp.kernel(fn, module="unique").module
            module.get_module_hash()
            self._modules[name] = (module, module.resolve_options(wp.config) | {"output_arch": None})
        wp.kernel(static_dispatch, module="unique")

    def time_codegen_scalar(self):
        module, options = self._modules["scalar"]
        module._run_codegen(options, is_cpu=True)

    def time_codegen_vector(self):
        module, options = self._modules["vector"]
        module._run_codegen(options, is_cpu=True)

    def time_declare_static_dispatch(self):
        wp.kernel(static_dispatch, module="unique")
