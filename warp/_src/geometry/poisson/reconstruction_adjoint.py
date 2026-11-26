# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in reconstruction VJP using checkpointed, finite-algorithm replay.

The public entry point is imported lazily. The default reconstruction
implementation is untouched.
"""

from dataclasses import dataclass, replace

import warp as wp

from .cascadic_adjoint import prepare_solve, replay_solve
from .field_adjoint import DifferentiableField, cube_field, evaluate_cube, record_cumulative_vjp, record_isovalue_vjp
from .keyed import Records
from .reconstruction import precision_kernels, reconstruct
from .sample_adjoint import replay_samples
from .splat_adjoint import replay_density, replay_normals
from .transforms import precision_kernels as transform_kernels


def _tracked(array):
    return wp.clone(array, requires_grad=True)


def _normal_checkpoint(normal):
    records = Records(normal.records.keys, _tracked(normal.records.values))
    levels, begin = [], 0
    for level in normal.levels:
        size = len(level.keys)
        values = (
            records.values[begin : begin + size]
            if size
            else wp.empty(0, dtype=records.values.dtype, device=records.values.device, requires_grad=True)
        )
        levels.append(Records(level.keys, values))
        begin += size
    return replace(normal, records=records, average=_tracked(normal.average), levels=levels)


def _sum_grads(*arrays):
    # Gradients are ordinary device data here. This sum is not a higher-order
    # tape operation; wp.utils.array_sum would reduce rather than add arrays.
    from warp.fem.linalg import array_axpy  # noqa: PLC0415

    result = wp.clone(arrays[0], requires_grad=False)
    for value in arrays[1:]:
        array_axpy(value, result)
    return result


@dataclass
class DifferentiableReconstruction:
    field: DifferentiableField
    metadata: dict
    diagnostics: dict

    @property
    def iso_value(self):
        return self.field.iso_value

    def evaluate(self, queries, *, centered=False):
        return self.field.evaluate(queries, centered=centered)

    def device_field(self):
        """Return a field handle connected to the reconstruction tape."""
        return self.field.device_field()


def reconstruct_differentiable(positions, normals, *, tape, **options):
    """Reconstruct outside ``with tape``, then record field losses on that tape.

    Inputs and returned checkpoint arrays must stay immutable until backward.
    Discrete tree/group/bin assignments and solver stopping branches are fixed.
    Only point positions and normals are differentiable inputs initially;
    scalar reconstruction options, including screening strength, are constants.
    First-order VJPs only; external CUDA graph capture is unsupported.
    """
    if not isinstance(tape, wp.Tape):
        raise TypeError("supply the wp.Tape that will record the field loss")
    if not positions.requires_grad and not normals.requires_grad:
        raise ValueError("track positions and/or normals with requires_grad=True")
    if positions.device.is_capturing:
        raise ValueError("differentiable reconstruction cannot run inside CUDA graph capture")
    # Reconstruction replay requires an inactive outer tape context. Check the documented
    # nesting invariant before any primal work, without changing private state.
    try:
        with wp.Tape():
            pass
    except RuntimeError as error:
        raise ValueError("call reconstruct_differentiable outside any with tape block") from error
    for name, value in options.items():
        if isinstance(value, wp.array):
            raise ValueError(f"{name} must be a constant scalar; only positions and normals support gradients")
    primal = reconstruct(positions, normals, **options)
    parameters, data = primal.metadata["parameters"], primal.diagnostics
    sample, density, normal = data["samples"], data["density"], data["normals"]
    dtype, device = primal.field.coefficients.dtype, positions.device
    depth = parameters["depth"]
    coefficients = _tracked(primal.field.coefficients)
    normalization = _tracked(sample.transform)
    cumulative = _tracked(primal.field.cumulative)
    iso = wp.array([primal.iso_value], dtype=dtype, device=device, requires_grad=True)
    mat44 = wp.mat44d if dtype == wp.float64 else wp.mat44
    transforms = wp.empty(2, dtype=mat44, device=device, requires_grad=True)
    ready = wp.Event(device)

    def backward():
        if device.is_capturing:
            raise ValueError("reconstruction backward cannot run inside CUDA graph capture")
        try:
            with wp.Tape():
                pass
        except RuntimeError as error:
            raise ValueError("reconstruction supports first-order backward only, outside any recording tape") from error
        wp.get_stream(device).wait_event(ready)
        # Isovalue and direct field losses meet at the coefficient checkpoint.
        c = _tracked(primal.field.coefficients)
        cum = _tracked(primal.field.cumulative)
        sums = _tracked(sample.position_sums)
        local_iso = _tracked(iso)
        inner = wp.Tape()
        record_cumulative_vjp(primal.field.keys, c, cum, depth, tape=inner)
        with inner:
            centroids = wp.empty_like(sums, requires_grad=True)
            wp.launch(
                precision_kernels(dtype)._centroids, len(sums), inputs=[sums, sample.weights, centroids], device=device
            )
            values, _, _ = evaluate_cube(cube_field(primal.field.keys, c, cum, depth), centroids, tape=inner)
            record_isovalue_vjp(values, sample.weights, local_iso, tape=inner)
        inner.backward(grads={c: coefficients.grad, local_iso: iso.grad})
        coefficient_seed, centroid_seed = c.grad, sums.grad
        del inner, c, cum, sums, local_iso, centroids, values

        # Solve/assembly stage. Sparse topologies and primal matrix arithmetic
        # are rebuilt before recording continuous replay operations.
        prepared = prepare_solve(primal)
        tracked_normal = _normal_checkpoint(normal)
        tracked_density = replace(density, aggregates=[Records(r.keys, _tracked(r.values)) for r in density.aggregates])
        inner = wp.Tape()
        with inner:
            correction, _ = replay_solve(prepared, tracked_density, tracked_normal, tape=inner)
        seeds, begin = {}, 0
        for value in correction:
            end = begin + len(value)
            if len(value):
                seeds[value] = coefficient_seed[begin:end]
            begin = end
        inner.backward(grads=seeds)
        normal_seed, average_seed = tracked_normal.records.values.grad, tracked_normal.average.grad
        aggregate_seeds = [r.values.grad for r in tracked_density.aggregates]
        del inner, prepared, correction, seeds, tracked_density, tracked_normal, coefficient_seed

        # Normal splats depend on grouped positions/normals and density depth.
        tracked_sample = replace(
            sample, position_sums=_tracked(sample.position_sums), normal_sums=_tracked(sample.normal_sums)
        )
        tracked_density = replace(density, depths_and_weights=_tracked(density.depths_and_weights))
        inner = wp.Tape()
        with inner:
            tracked_normal = replay_normals(
                tracked_sample, tracked_density, normal, tape=inner, minimum=parameters["base_depth"], maximum=depth
            )
        inner.backward(grads={tracked_normal.records.values: normal_seed, tracked_normal.average: average_seed})
        position_seed, normal_sum_seed = tracked_sample.position_sums.grad, tracked_sample.normal_sums.grad
        depth_seed = tracked_density.depths_and_weights.grad
        del inner, tracked_normal, tracked_sample, tracked_density, normal_seed, average_seed

        # Density also supplies the level-clustered point constraints.
        tracked_sample = replace(sample, position_sums=_tracked(sample.position_sums))
        inner = wp.Tape()
        with inner:
            tracked_density = replay_density(tracked_sample, density, samples_per_node=parameters["samples_per_node"])
        seeds = {tracked_density.depths_and_weights: depth_seed}
        seeds.update({r.values: g for r, g in zip(tracked_density.aggregates, aggregate_seeds, strict=True)})
        inner.backward(grads=seeds)
        combined_position_seed = _sum_grads(centroid_seed, position_seed, tracked_sample.position_sums.grad)
        del inner, tracked_density, tracked_sample, seeds, depth_seed, aggregate_seeds, centroid_seed, position_seed

        # Final reverse pass includes extrema-owned normalization and unit
        # normals, adding to any gradients already accumulated on caller inputs.
        inner = wp.Tape()
        with inner:
            tracked_sample = replay_samples(positions, normals, sample, depth=depth, scale=parameters["scale"])
        inner.backward(
            grads={
                tracked_sample.position_sums: combined_position_seed,
                tracked_sample.normal_sums: normal_sum_seed,
                tracked_sample.transform: normalization.grad,
            }
        )

    tape.record_func(
        backward, [coefficients, normalization, iso] + [a for a in (positions, normals) if a.requires_grad]
    )
    record_cumulative_vjp(primal.field.keys, coefficients, cumulative, depth, tape=tape)
    with tape:
        wp.launch(transform_kernels(dtype).evaluation_transform, 1, inputs=[normalization, transforms], device=device)
    field = DifferentiableField(
        cube_field(primal.field.keys, coefficients, cumulative, depth), transforms, iso, tape, ready
    )
    wp.get_stream(device).record_event(ready)
    metadata = dict(primal.metadata)
    metadata["differentiability"] = {
        "mode": "finite_algorithm_fixed_assignments",
        "inputs": ["positions", "normals"],
        "constant_options": ["point_weight", "scale", "samples_per_node", "cg_accuracy", "depths", "iterations"],
        "first_order_only": True,
    }
    return DifferentiableReconstruction(field, metadata, data)
