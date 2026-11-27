# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact finite cascadic solve replay on saved reconstruction topology."""

from types import SimpleNamespace

import warp as wp
import warp.sparse as sparse

from .assembly_adjoint import record_screening_vjp, replay_constraints
from .constraints import point_constraints
from .forms import assemble_point_load, assemble_screening, point_quadrature
from .integration import level_integration_grid
from .multigrid_adjoint import finite_multigrid, prepare_hierarchy
from .normal_load_adjoint import prepare_normal_restrictions, replay_normal_loads
from .relaxation import row_schedule
from .relaxation_adjoint import differentiable_relaxation
from .screening_stencils import add_screening_stencil, bin_constraints
from .solver import precision_kernels, prolongation
from .solver_adjoint import functional_spmv
from .spaces import make_level_space
from .stiffness import assemble_level_stiffness


def prepare_solve(result):
    """Rebuild cross operators and point loads outside a tape, from saved data.

    Finest square matrices retain the exact primal values. Newly owned tracked
    buffers isolate this replay from the caller's diagnostics and other tapes.
    """
    options, data = result.metadata["parameters"], result.diagnostics
    tree, density, normals = data["tree"], data["density"], data["normals"]
    dtype, device = density.scale.dtype, normals.average.device
    keys = [level.keys for level in tree.basis]
    transfers = [prolongation(keys[d - 1], keys[d], d, dtype) for d in range(1, len(keys))]
    base = options["base_depth"]
    screened = options["point_weight"] > 0 and not options["exact_interpolation"]
    levels = []
    for primal in data["levels"]:
        d = primal["depth"]
        matrix = sparse.bsr_copy(primal["matrix"], topology="padded")
        matrix.values.requires_grad = True
        cross, space, geometry, bins = None, None, None, None
        load = wp.zeros(len(keys[d]), dtype=wp.float64, device=device, requires_grad=True)
        direct = screened and d > base and d >= 4
        if screened:
            points = point_constraints(density, normals, tree, d, options["point_weight"])
            if direct:
                # A zero stencil computes precisely the original point load;
                # its temporary values are discarded, not differentiated.
                scratch = assemble_level_stiffness(keys[d], keys[d], d)
                sparse.bsr_scale(scratch, 0)
                bins = bin_constraints(points, d)
                load = add_screening_stencil(scratch, keys[d], keys[d], points, d, bins=bins)
            else:
                geometry = level_integration_grid(tree, d, dtype)
                space = make_level_space(geometry, keys[d], d, d)
                quadrature = point_quadrature(space, points.positions, points.weights)
                load = assemble_point_load(space, points.positions, points.weights, quadrature=quadrature)
            load.requires_grad = True
        if d > base:
            cross = assemble_level_stiffness(keys[d], keys[d - 1], d, d - 1)
            if direct:
                add_screening_stencil(cross, keys[d], keys[d - 1], points, d, d - 1, bins=bins)
            elif screened:
                coarse = make_level_space(geometry, keys[d - 1], d - 1, d)
                screen = assemble_screening(
                    space,
                    points.positions,
                    points.weights,
                    trial_space=coarse,
                    accumulation_dtype=wp.float64,
                    quadrature=quadrature,
                )
                sparse.bsr_axpy(screen, cross)
            cross = sparse.bsr_copy(cross, scalar_type=dtype, topology="padded")
            cross.values.requires_grad = True
        levels.append(
            SimpleNamespace(
                depth=d,
                matrix=matrix,
                point_load=load,
                cross=cross,
                schedule=row_schedule(keys[d], d, sliced=True) if d > base else None,
            )
        )
    hierarchy = prepare_hierarchy(
        levels[0].matrix, keys[: base + 1], transfers[:base], add_dc_term=base == 0 and options["point_weight"] == 0
    )
    return SimpleNamespace(
        options=options,
        tree=tree,
        keys=keys,
        transfers=transfers,
        levels=levels,
        hierarchy=hierarchy,
        screened=screened,
        normal_restrictions=prepare_normal_restrictions(transfers),
    )


def replay_solve(prepared, density, normals, *, tape):
    """Replay finite corrections; differentiate loads, screening and all solves."""
    options, keys = prepared.options, prepared.keys
    dtype, device = density.scale.dtype, normals.average.device
    base = options["base_depth"]
    normal_loads = replay_normal_loads(
        prepared.tree, normals, len(keys) - 1, base, prepared.transfers, restrictions=prepared.normal_restrictions
    )
    corrections = [wp.zeros(len(k), dtype=dtype, device=device, requires_grad=True) for k in keys]
    cumulative, trace = None, None
    for level in prepared.levels:
        d = level.depth
        if prepared.screened:
            points = replay_constraints(density, normals, prepared.tree, d, options["point_weight"])
            record_screening_vjp(level.matrix, keys[d], keys[d], points, d, tape=tape, load=level.point_load)
            if level.cross is not None:
                record_screening_vjp(level.cross, keys[d], keys[d - 1], points, d, trial_depth=d - 1, tape=tape)
        rhs = wp.empty(len(keys[d]), dtype=dtype, device=device, requires_grad=True)
        wp.launch(
            precision_kernels(dtype)._rhs, len(rhs), inputs=[normal_loads[d], level.point_load, rhs], device=device
        )
        if d == base:
            corrections[d], trace = finite_multigrid(
                prepared.hierarchy,
                rhs,
                tape=tape,
                iters=options["iters"],
                cycles=options["base_v_cycles"],
                cg_accuracy=options["cg_accuracy"],
            )
            cumulative = corrections[d]
        else:
            rhs = functional_spmv(level.cross, cumulative, rhs, alpha=-1, beta=1)
            corrections[d] = differentiable_relaxation(
                level.matrix, rhs, corrections[d], level.schedule, tape=tape, iters=options["iters"], sliced_depth=d
            )
            cumulative = functional_spmv(prepared.transfers[d - 1], cumulative, corrections[d], beta=1)
    return corrections, trace
