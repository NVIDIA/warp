# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference cascadic correction solve using Warp FEM and Warp sparse algebra."""

import math
import time
from functools import lru_cache
from types import SimpleNamespace

import warp as wp
import warp.sparse as sparse
from warp.optim.linear import LinearOperator, aslinearoperator, cg

from .constraints import point_constraints
from .field import basis_key, find_key
from .forms import (
    assemble_point_load,
    assemble_screening,
    assemble_stiffness,
    point_quadrature,
)
from .integration import level_integration_grid
from .keyed import coordinates
from .normal_loads import assemble_reference_normal_loads
from .relaxation import relax_scheduled, relax_sliced, relaxation_command, row_schedule
from .screening_stencils import add_screening_stencil, bin_constraints
from .spaces import make_level_space
from .stiffness import assemble_level_stiffness
from .transfer_stencils import assemble_prolongation

wp.set_module_options({"fuse_fp": False})


@wp.func
def root_row(key: wp.int64):
    c = coordinates(key) + wp.vec3i(2)
    return c[2] * 16 + (c[0] & 1) + 2 * (c[1] & 1) + 2 * (c[0] & 2) + 4 * (c[1] & 2)


@wp.kernel
def root_matrix_rows_kernel(
    keys: wp.array[wp.int64],
    offsets: wp.array[int],
    columns: wp.array[int],
    rows: wp.array[int],
    cols: wp.array[int],
):
    i = wp.tid()
    for j in range(offsets[i], offsets[i + 1]):
        rows[j] = root_row(keys[i])
        cols[j] = root_row(keys[columns[j]])
    # Warp's nnz is an upper bound until explicitly synchronized. The final
    # row owns unused capacity, which triplet assembly must ignore.
    if i == keys.shape[0] - 1:
        for j in range(offsets[keys.shape[0]], rows.shape[0]):
            rows[j] = -1
            cols[j] = -1


@lru_cache(None)
def precision_kernels(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)

    @wp.kernel
    def _transfer_kernel(
        fine: wp.array[wp.int64],
        coarse: wp.array[wp.int64],
        depth: int,
        rows: wp.array[int],
        cols: wp.array[int],
        values: wp.array[dtype],
    ):
        i, k = wp.tid()
        f = coordinates(fine[i])
        c = wp.vec3i(f[0] // 2, f[1] // 2, f[2] // 2) + wp.vec3i(k & 1, k >> 1 & 1, k >> 2 & 1)
        h = vec3(dtype(1.0)) - wp.abs(vec3(f) * dtype(0.5) - vec3(c))
        v = h[0] * h[1] * h[2]
        idx = int(-1)
        if v > dtype(0.0):
            idx = find_key(coarse, basis_key(depth - 1, c[0], c[1], c[2]))
        rows[i * 8 + k] = i
        cols[i * 8 + k] = idx
        values[i * 8 + k] = v

    @wp.kernel
    def _gs_kernel(
        keys: wp.array[wp.int64],
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        b: wp.array[dtype],
        x: wp.array[dtype],
        color: int,
        z: int,
    ):
        i = wp.tid()
        c = coordinates(keys[i])
        if (c[0] & 1) + 2 * (c[1] & 1) + 4 * (c[2] & 1) == color and (z < 0 or c[2] == z):
            r = b[i]
            diagonal = dtype(0.0)
            for j in range(offsets[i], offsets[i + 1]):
                col = columns[j]
                r -= x[col] * values[j]
                if col == i:
                    diagonal = values[j]
            if diagonal != dtype(0.0):
                x[i] += r * (dtype(1.0) / diagonal)

    @wp.kernel
    def _combine_kernel(
        a: wp.array[dtype],
        b: wp.array[dtype],
        scale: dtype,
        out: wp.array[dtype],
    ):
        i = wp.tid()
        out[i] = a[i] + scale * b[i]

    @wp.kernel
    def _rhs_kernel(
        a: wp.array[wp.float64],
        b: wp.array[wp.float64],
        out: wp.array[dtype],
    ):
        i = wp.tid()
        out[i] = dtype(a[i]) + dtype(0.5) * dtype(b[i])

    @wp.kernel
    def _add_mean_kernel(x: wp.array[dtype], out: wp.array[dtype], alpha: dtype):
        # Preserve the reference's padded root row order and scalar accumulation.
        mean = dtype(0.0)
        for i in range(x.shape[0]):
            mean += x[i]
        mean /= dtype(x.shape[0])
        for i in range(out.shape[0]):
            out[i] += alpha * mean

    @wp.kernel
    def _root_embed_kernel(
        keys: wp.array[wp.int64],
        src: wp.array[dtype],
        out: wp.array[dtype],
    ):
        i = wp.tid()
        out[root_row(keys[i])] = src[i]

    @wp.kernel
    def _root_extract_kernel(
        keys: wp.array[wp.int64],
        src: wp.array[dtype],
        out: wp.array[dtype],
    ):
        i = wp.tid()
        out[i] = src[root_row(keys[i])]

    return SimpleNamespace(
        _transfer=_transfer_kernel,
        _gs=_gs_kernel,
        _combine=_combine_kernel,
        _rhs=_rhs_kernel,
        _add_mean=_add_mean_kernel,
        _root_embed=_root_embed_kernel,
        _root_extract=_root_extract_kernel,
    )


def constant_mode_operator(matrix):
    """Reference root operator: K x + mean(x), through public Warp CG APIs."""
    operator = aslinearoperator(matrix)

    def matvec(x, y, z, alpha, beta):
        operator.matvec(x, y, z, alpha, beta)
        wp.launch(
            precision_kernels(x.dtype)._add_mean,
            1,
            inputs=[x, z, x.dtype(alpha)],
            device=x.device,
        )

    return LinearOperator(operator.shape, operator.dtype, operator.device, matvec)


def padded_root_system(matrix, b, keys):
    """Retain the 56 inactive rows included in the official root mean term."""
    count = matrix.nnz
    rows = wp.empty(count, dtype=int, device=b.device)
    cols = wp.empty_like(rows)
    wp.launch(
        root_matrix_rows_kernel,
        len(keys),
        inputs=[keys, matrix.offsets, matrix.columns, rows, cols],
        device=b.device,
    )
    padded = sparse.bsr_from_triplets(64, 64, rows, cols, matrix.values[:count])
    rhs = wp.zeros(64, dtype=b.dtype, device=b.device)
    wp.launch(
        precision_kernels(b.dtype)._root_embed,
        len(keys),
        inputs=[keys, b, rhs],
        device=b.device,
    )
    return padded, rhs


def prolongation(coarse_keys, fine_keys, depth, dtype=wp.float32, direct=True):
    if direct:
        return assemble_prolongation(coarse_keys, fine_keys, depth, dtype)
    device = fine_keys.device
    count = len(fine_keys) * 8
    rows, cols = (
        wp.empty(count, dtype=int, device=device),
        wp.empty(count, dtype=int, device=device),
    )
    values = wp.empty(count, dtype=dtype, device=device)
    if count:
        wp.launch(
            precision_kernels(dtype)._transfer,
            (len(fine_keys), 8),
            inputs=[fine_keys, coarse_keys, depth, rows, cols, values],
            device=device,
        )
    return sparse.bsr_from_triplets(len(fine_keys), len(coarse_keys), rows, cols, values)


def relax(A, b, x, keys, iters, forward=True, z=-1):
    colors = range(8) if forward else range(7, -1, -1)
    for _ in range(iters):
        for color in colors:
            if z >= 0 and (color >> 2) != (z & 1):
                continue
            wp.launch(
                precision_kernels(b.dtype)._gs,
                len(keys),
                inputs=[keys, A.offsets, A.columns, A.values, b, x, color, z],
                device=keys.device,
            )


def regular_multigrid(A, b, keys, transfers, iters, cycles, cg_accuracy, add_dc_term=False):
    base = len(keys) - 1
    matrices = [None] * base + [A]
    restrictions = [sparse.bsr_transposed(P) for P in transfers]
    for d in range(base, 0, -1):
        matrices[d - 1] = sparse.bsr_mm(sparse.bsr_mm(restrictions[d - 1], matrices[d]), transfers[d - 1])
    rhs = [wp.zeros(len(k), dtype=b.dtype, device=k.device) for k in keys]
    rhs[base] = b
    x = [wp.zeros_like(v) for v in rhs]
    schedules = [row_schedule(k, d) for d, k in enumerate(keys)]
    commands = [relaxation_command(matrices[d], rhs[d], x[d], schedules[d]) for d in range(base + 1)]
    sync = sum(schedule.sync_seconds for schedule in schedules)
    root_rhs, root_x = rhs[0], x[0]
    if add_dc_term:
        padded, root_rhs = padded_root_system(matrices[0], rhs[0], keys[0])
        root_x = wp.zeros_like(root_rhs)
        root_operator = constant_mode_operator(padded)
    else:
        root_operator = aslinearoperator(matrices[0])
    for cycle in range(cycles):
        for d in range(base, 0, -1):
            relax_scheduled(matrices[d], rhs[d], x[d], schedules[d], iters, command=commands[d])
            residual = wp.clone(rhs[d])
            sparse.bsr_mv(matrices[d], x[d], residual, alpha=-1.0, beta=1.0)
            sparse.bsr_mv(restrictions[d - 1], residual, rhs[d - 1])
            x[d - 1].zero_()
        # The reference skips CG altogether when its initial squared residual
        # is <= eps^2. After this check its stopping tolerance is relative.
        residual = root_rhs
        if base == 0 and cycle > 0:
            # Unlike a V-cycle with a finer base, a repeated root-only cycle
            # retains x[0]. The reference threshold is relative to b - A x.
            residual = wp.empty_like(root_rhs)
            root_operator.matvec(root_x, root_rhs, residual, -1.0, 1.0)
        start = time.perf_counter()
        norm_squared = wp.utils.array_inner(residual, residual)
        sync += time.perf_counter() - start
        if norm_squared > cg_accuracy * cg_accuracy:
            cg(
                root_operator,
                root_rhs,
                root_x,
                tol=0 if base == 0 else cg_accuracy,
                atol=cg_accuracy * math.sqrt(norm_squared) if base == 0 else 0,
                maxiter=len(keys[0]),
                check_every=1,
                use_graph=False,
            )
        for d in range(1, base + 1):
            sparse.bsr_mv(transfers[d - 1], x[d - 1], x[d], beta=1.0)
            x[d - 1].zero_()
            relax_scheduled(
                matrices[d],
                rhs[d],
                x[d],
                schedules[d],
                iters,
                forward=False,
                command=commands[d],
            )
    if add_dc_term:
        wp.launch(
            precision_kernels(b.dtype)._root_extract,
            len(keys[0]),
            inputs=[keys[0], root_x, x[0]],
            device=b.device,
        )
    return x[base], sync


def solve_levels(
    tree,
    geometry,
    normals,
    density,
    *,
    base_depth,
    solve_depth,
    point_weight=2.0,
    exact=False,
    iters=8,
    base_v_cycles=1,
    cg_accuracy=1e-3,
    stencil_stiffness=True,
    stencil_screening=True,
):
    device = normals.records.keys.device
    dtype = density.scale.dtype
    finest = len(tree.basis) - 1
    keys = [level.keys for level in tree.basis]
    transfers = [prolongation(keys[d - 1], keys[d], d, dtype) for d in range(1, finest + 1)]
    corrections = [wp.zeros(len(k), dtype=dtype, device=device) for k in keys]
    diagnostics = []
    solver_sync = 0.0
    cumulative = None
    operator_geometries = {finest: geometry} if geometry is not None else {}
    # The reference assembles normal constraints through reconstruction depth,
    # including fine contributions when the requested solve stops earlier.
    normal_loads = assemble_reference_normal_loads(
        tree,
        geometry,
        normals,
        finest,
        base_depth,
        transfers,
        operator_geometries=operator_geometries,
    )
    screened = point_weight > 0 and not exact
    for d in range(base_depth, solve_depth + 1):
        if not len(keys[d]):
            break
        load = normal_loads[d]
        direct_points = screened and stencil_stiffness and stencil_screening and d > base_depth and d >= 4
        needs_fem = d == base_depth or not stencil_stiffness or (screened and not direct_points)
        operator_geometry = None
        space = None
        if needs_fem:
            if d not in operator_geometries:
                operator_geometries[d] = level_integration_grid(tree, d, dtype=dtype)
            operator_geometry = operator_geometries[d]
            space = make_level_space(operator_geometry, keys[d], d, d)
        operator_finest = d
        # Double accumulation avoids a coarse row losing accuracy when its
        # integral spans many fine integration cells. Final reference matrices
        # and solve arithmetic use the input precision.
        A = (
            assemble_level_stiffness(keys[d], keys[d], d)
            if stencil_stiffness and d > base_depth
            else assemble_stiffness(space, device, accumulation_dtype=wp.float64)
        )
        if screened:
            points = point_constraints(density, normals, tree, d, point_weight, exact)
        if direct_points:
            bins = bin_constraints(points, d)
            point_load = add_screening_stencil(A, keys[d], keys[d], points, d, bins=bins)
        elif screened:
            quadrature = point_quadrature(space, points.positions, points.weights)
            screen = assemble_screening(
                space,
                points.positions,
                points.weights,
                accumulation_dtype=wp.float64,
                quadrature=quadrature,
            )
            sparse.bsr_axpy(screen, A)
            point_load = assemble_point_load(space, points.positions, points.weights, quadrature=quadrature)
        else:
            point_load = wp.zeros_like(load)
        b = wp.empty(len(keys[d]), dtype=dtype, device=device)
        wp.launch(
            precision_kernels(dtype)._rhs,
            len(b),
            inputs=[load, point_load, b],
            device=device,
        )
        # Assembly produces compact rows. Preserve their exact capacities while
        # casting, instead of rebuilding the same topology through triplet sort.
        A = sparse.bsr_copy(A, scalar_type=dtype, topology="padded")
        if d == base_depth:
            corrections[d], sync = regular_multigrid(
                A,
                b,
                keys[: d + 1],
                transfers[:d],
                iters,
                base_v_cycles,
                cg_accuracy,
                # Pinned _solveRegularMG compares the root row count with
                # dimensions at base_depth, so this is enabled only at base 0.
                # Exact interpolation still claims to constrain the constant
                # mode despite its empty spans.
                add_dc_term=base_depth == 0 and point_weight == 0,
            )
            solver_sync += sync
            cumulative = wp.clone(corrections[d])
        else:
            coarse_space = (
                make_level_space(operator_geometry, keys[d - 1], d - 1, operator_finest) if needs_fem else None
            )
            cross = (
                assemble_level_stiffness(keys[d], keys[d - 1], d, d - 1)
                if stencil_stiffness
                else assemble_stiffness(space, device, trial_space=coarse_space, accumulation_dtype=wp.float64)
            )
            if direct_points:
                add_screening_stencil(cross, keys[d], keys[d - 1], points, d, d - 1, bins=bins)
            elif screened:
                screen = assemble_screening(
                    space,
                    points.positions,
                    points.weights,
                    trial_space=coarse_space,
                    accumulation_dtype=wp.float64,
                    quadrature=quadrature,
                )
                sparse.bsr_axpy(screen, cross)
            cross = sparse.bsr_copy(cross, scalar_type=dtype, topology="padded")
            sparse.bsr_mv(cross, cumulative, b, alpha=-1.0, beta=1.0)
            # Pinned sliceBlockSize=1, degree=1, coarse-to-fine traversal.
            # Independent slices share a launch; every coupled slice/color
            # retains the reference order.
            schedule = row_schedule(keys[d], d, sliced=True)
            solver_sync += schedule.sync_seconds
            relax_sliced(A, b, corrections[d], schedule, d, iters)
            next_cumulative = wp.clone(corrections[d])
            sparse.bsr_mv(transfers[d - 1], cumulative, next_cumulative, beta=1.0)
            cumulative = next_cumulative
        diagnostics.append(
            {
                "depth": d,
                "matrix": A,
                "rhs": b,
                "normal_load": load,
                "coefficients": corrections[d],
                "operator_integration_cells": operator_geometry.cell_count() if operator_geometry is not None else 0,
            }
        )
    return corrections, diagnostics, solver_sync
