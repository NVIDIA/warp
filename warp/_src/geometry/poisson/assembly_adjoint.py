# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-constraint replay and fixed-layout screening VJPs.

Forward matrix assembly stays in its original FEM/stencil implementation. Only
its continuous point form is differentiated here; stiffness and CSR topology
are constant on the saved tree.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from .constraints import PointConstraints
from .constraints import precision_kernels as constraint_kernels
from .field import basis_key, find_key
from .keyed import coordinates
from .screening_stencils import bin_constraints

wp.set_module_options({"fuse_fp": False})


def replay_constraints(density, normals, tree, depth, point_weight, exact=False):
    source_depth = len(density.aggregates) - 1 if exact else depth
    source = density.aggregates[source_depth]
    dtype, device = density.scale.dtype, source.keys.device
    p = wp.empty(len(source.keys), dtype=normals.average.dtype, device=device, requires_grad=True)
    w = wp.empty(len(source.keys), dtype=dtype, device=device, requires_grad=True)
    wp.launch(
        constraint_kernels(dtype)._points,
        len(p),
        inputs=[
            source.keys,
            source.values,
            tree.levels[depth].keys,
            normals.average,
            dtype(point_weight),
            depth,
            source_depth,
            p,
            w,
        ],
        device=device,
    )
    if exact:
        # Exact interpolation is disabled in the pinned reference; these zeros
        # must have no dependency on the computed weights.
        w = wp.zeros_like(w, requires_grad=True)
    return PointConstraints(p, w)


@lru_cache(None)
def precision_kernels(dtype, matrix_dtype):
    vec3 = wp.vec3d if dtype == wp.float64 else wp.vec3

    @wp.func
    def hat_and_gradient(p: vec3, c: wp.vec3i, depth: int):
        n = dtype(1 << depth)
        t = p * n - vec3(c)
        h = wp.max(vec3(dtype(1)) - wp.abs(t), vec3(dtype(0)))
        derivative = vec3(dtype(0))
        for a in range(3):
            if wp.abs(t[a]) < dtype(1) and t[a] != dtype(0):
                derivative[a] = -wp.sign(t[a]) * n
        return h[0] * h[1] * h[2], vec3(
            derivative[0] * h[1] * h[2], h[0] * derivative[1] * h[2], h[0] * h[1] * derivative[2]
        )

    @wp.kernel
    def backward_kernel(
        row_keys: wp.array[wp.int64],
        col_keys: wp.array[wp.int64],
        point_keys: wp.array[wp.int64],
        ids: wp.array[int],
        positions: wp.array[vec3],
        weights: wp.array[dtype],
        depth: int,
        trial_depth: int,
        offsets: wp.array[int],
        columns: wp.array[int],
        grad_matrix: wp.array[matrix_dtype],
        grad_load: wp.array[wp.float64],
        grad_positions: wp.array[vec3],
        grad_weights: wp.array[dtype],
    ):
        row = wp.tid()
        c = coordinates(row_keys[row])
        for corner in range(8):
            cell = c - wp.vec3i(corner // 4, (corner // 2) % 2, corner % 2)
            if wp.min(cell) >= 0 and wp.max(cell) < 1 << depth:
                key = basis_key(depth, cell[0], cell[1], cell[2])
                start = find_key(point_keys, key)
                if start >= 0:
                    j = start
                    while j < point_keys.shape[0]:
                        if point_keys[j] != key:
                            break
                        index = ids[j]
                        p = vec3(positions[index])
                        w = dtype(weights[index])
                        # The stencil's zero-weight branch is frozen. Such
                        # constraints are inactive constants in reconstruction.
                        if w != dtype(0):
                            phi, dphi = hat_and_gradient(p, c, depth)
                            gl = dtype(0)
                            if grad_load.shape[0]:
                                gl = dtype(grad_load[row])
                            gp = (gl * w) * dphi
                            gw = gl * phi
                            for entry in range(offsets[row], offsets[row + 1]):
                                trial, dtrial = hat_and_gradient(p, coordinates(col_keys[columns[entry]]), trial_depth)
                                g = dtype(grad_matrix[entry])
                                # Reverse the input-precision phi*trial*weight
                                # product before its double accumulation.
                                gw += g * (phi * trial)
                                gproduct = g * w
                                gp += (gproduct * trial) * dphi + (gproduct * phi) * dtrial
                            wp.atomic_add(grad_positions, index, gp)
                            wp.atomic_add(grad_weights, index, gw)
                        j += 1

    return SimpleNamespace(backward=backward_kernel)


def record_screening_vjp(matrix, row_keys, col_keys, points, depth, *, tape, trial_depth=None, load=None):
    """Attach a VJP to tracked primal matrix/load checkpoints.

    Point positions must lie inside the unit cube, away from basis knots; bins,
    sparse matrix entries and the zero-weight branch remain fixed. The matrix
    may use either precision. A missing load means this is a cross operator.
    """
    trial_depth = depth if trial_depth is None else trial_depth
    if not matrix.values.requires_grad or not points.positions.requires_grad or not points.weights.requires_grad:
        raise ValueError("screening replay requires tracked values, positions and weights")
    if load is not None and (load.dtype != wp.float64 or not load.requires_grad):
        raise ValueError("point load must be a tracked double array")
    bins = bin_constraints(points, depth)
    empty = wp.empty(0, dtype=wp.float64, device=row_keys.device)

    def backward_kernel():
        wp.launch(
            precision_kernels(points.weights.dtype, matrix.values.dtype).backward,
            len(row_keys),
            inputs=[
                row_keys,
                col_keys,
                bins.keys,
                bins.ids,
                points.positions,
                points.weights,
                depth,
                trial_depth,
                matrix.offsets,
                matrix.columns,
                matrix.values.grad,
                load.grad if load is not None else empty,
                points.positions.grad,
                points.weights.grad,
            ],
            device=row_keys.device,
            record_tape=False,
        )

    tape.record_func(
        backward_kernel, [matrix.values, points.positions, points.weights] + ([load] if load is not None else [])
    )
