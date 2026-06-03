# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-kernel declaration cost of unique-module (factory) kernels.

Workloads that build kernels with ``@wp.kernel(module="unique")`` at import
time — e.g. mjwarp, which instantiates ~180 factory kernels — pay a full
per-kernel declaration once for every kernel. Declaring a unique-module kernel
runs the whole Python-side front end: source extraction, ``ast.parse``,
argument-type resolution, and the module-hash pass (which traverses the
adjoint to fingerprint it). Native compilation is *not* part of this — it is
cached and amortised separately — so this declaration cost is what dominates
import time for factory-heavy programs.

This benchmark reproduces that pattern directly: it re-declares the same
function with ``module="unique"`` in a loop. Each call runs the full
declaration even though the resulting module is content-identical and gets
discarded by hash-based reuse — i.e. the front end runs end-to-end on every
iteration, exactly as it does for each distinct factory kernel at import.

Because it spans the entire declaration front end rather than one helper, it
tracks the cumulative effect of the declaration-speedup work: the
``co_lines()`` source-extraction fast path lands first, and the follow-up
codegen-walk / AST-traversal optimisations show up on the same benchmark as
they land. Three representative kernel sizes are declared, mirroring the spread
of real factory kernels:

- ``declare_small``  — a ~20-line sparse mat-vec kernel
- ``declare_medium`` — a ~35-line bound-projection kernel
- ``declare_large``  — a ~55-line equality-constraint kernel

The functions are valid Warp kernels but are only declared, never launched, so
the benchmark is device-independent and needs no GPU.
"""

import warp as wp


def small_sparse_mv(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    alpha: wp.float32,
    beta: wp.float32,
):
    i = wp.tid()
    row_start = rows[i]
    row_end = rows[i + 1]
    acc = float(0.0)
    count = int(0)
    for k in range(row_start, row_end):
        col = cols[k]
        acc += vals[k] * x[col]
        count += 1
    if count == 0:
        y[i] = beta * y[i]
    else:
        y[i] = alpha * acc + beta * y[i]


def medium_bound_project(
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    diag: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    active: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    lo = lower[i]
    hi = upper[i]
    d = diag[i]
    r = rhs[i]
    xi = x[i]
    grad = d * xi - r
    step = grad / wp.max(d, 1.0e-6)
    proposed = xi - step
    if proposed < lo:
        proposed = lo
    if proposed > hi:
        proposed = hi
    residual = d * proposed - r
    if residual < 0.0:
        residual = -residual
    is_active = float(1.0)
    if proposed <= lo + 1.0e-7:
        is_active = float(0.0)
    elif proposed >= hi - 1.0e-7:
        is_active = float(0.0)
    blended = proposed * is_active + xi * (1.0 - is_active)
    out[i] = blended
    active[i] = is_active


def large_equality_flex(
    body_a: wp.array(dtype=wp.int32),
    body_b: wp.array(dtype=wp.int32),
    anchor_a: wp.array(dtype=wp.vec3),
    anchor_b: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    rot: wp.array(dtype=wp.quatf),
    stiffness: wp.array(dtype=wp.float32),
    damping: wp.array(dtype=wp.float32),
    vel: wp.array(dtype=wp.vec3),
    out_force: wp.array(dtype=wp.vec3),
    out_torque: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    a = body_a[i]
    b = body_b[i]
    pa = pos[a]
    pb = pos[b]
    world_a = pa + wp.quat_rotate(rot[a], anchor_a[i])
    world_b = pb + wp.quat_rotate(rot[b], anchor_b[i])
    delta = world_b - world_a
    dist = wp.length(delta)
    if dist < 1.0e-9:
        direction = wp.vec3(0.0, 0.0, 0.0)
    else:
        direction = delta / dist
    rel_vel = vel[b] - vel[a]
    normal_vel = wp.dot(rel_vel, direction)
    magnitude = stiffness[i] * dist + damping[i] * normal_vel
    soft = float(1.0)
    if dist > 1.0:
        soft = 1.0 / dist
    elif dist < 0.01:
        soft = 0.0
    force = direction * (magnitude * soft)
    torque_a = wp.cross(world_a - pa, force) * soft
    torque_b = wp.cross(world_b - pb, -force) * soft
    clamp = float(1.0e4)
    fx = wp.clamp(force[0], -clamp, clamp)
    fy = wp.clamp(force[1], -clamp, clamp)
    fz = wp.clamp(force[2], -clamp, clamp)
    limited = wp.vec3(fx, fy, fz)
    tax = wp.clamp(torque_a[0], -clamp, clamp)
    tay = wp.clamp(torque_a[1], -clamp, clamp)
    taz = wp.clamp(torque_a[2], -clamp, clamp)
    tbx = wp.clamp(torque_b[0], -clamp, clamp)
    tby = wp.clamp(torque_b[1], -clamp, clamp)
    tbz = wp.clamp(torque_b[2], -clamp, clamp)
    limited_torque_a = wp.vec3(tax, tay, taz)
    limited_torque_b = wp.vec3(tbx, tby, tbz)
    wp.atomic_add(out_force, a, -limited)
    wp.atomic_add(out_force, b, limited)
    wp.atomic_add(out_torque, a, limited_torque_a)
    wp.atomic_add(out_torque, b, limited_torque_b)


class DeclareUniqueModuleKernel:
    """Time full per-kernel declaration of a unique-module kernel by size.

    ``module="unique"`` forces the entire declaration front end to run on every
    call (source extraction, parse, arg resolution, and the module-hash
    adjoint traversal), discarding the result via hash-based reuse. Looping the
    declaration lifts each sample well above the millisecond-scale scheduler
    noise floor; many ASV repeats stabilise the median.
    """

    repeat = 15
    number = 40
    warmup_time = 0.2

    def setup(self):
        wp.init()
        self._small = small_sparse_mv
        self._medium = medium_bound_project
        self._large = large_equality_flex
        # Warm linecache and first-module registration so samples reflect the
        # steady-state per-kernel declaration cost rather than one-time setup.
        for fn in (self._small, self._medium, self._large):
            wp.kernel(fn, module="unique")

    def time_declare_small(self):
        wp.kernel(self._small, module="unique")

    def time_declare_medium(self):
        wp.kernel(self._medium, module="unique")

    def time_declare_large(self):
        wp.kernel(self._large, module="unique")
