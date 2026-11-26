# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent GPU PSR/SPSR reconstruction for the pinned degree-one profile."""

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from .density import build_density
from .field import LevelField, cumulative_coefficients
from .normals import build_normal_field
from .samples import point_div, prepare_samples
from .solver import solve_levels
from .transforms import precision_kernels as transform_kernels
from .tree import build_tree


@lru_cache(None)
def precision_kernels(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)

    @wp.kernel
    def _centroids_kernel(
        sums: wp.array[vec3],
        weights: wp.array[dtype],
        out: wp.array[vec3],
    ):
        i = wp.tid()
        out[i] = point_div(sums[i], weights[i])

    @wp.kernel
    def _isovalue_kernel(
        values: wp.array[dtype],
        weights: wp.array[dtype],
        out: wp.array[dtype],
    ):
        total = wp.float64(0.0)
        weight = wp.float64(0.0)
        for i in range(values.shape[0]):
            total += wp.float64(values[i] * weights[i])
            weight += wp.float64(weights[i])
        out[0] = dtype(total / weight)

    @wp.kernel(module="unique")
    def _isovalue_tiled_kernel(values: wp.array[dtype], weights: wp.array[dtype], out: wp.array[dtype]):
        _block, lane = wp.tid()
        total = wp.float64(0.0)
        weight = wp.float64(0.0)
        for begin in range(0, values.shape[0], 128):
            value_tile = wp.tile_load(values, shape=128, offset=begin)
            weight_tile = wp.tile_load(weights, shape=128, offset=begin)
            for j in range(wp.min(128, values.shape[0] - begin)):
                v = wp.tile_extract(value_tile, j)
                w = wp.tile_extract(weight_tile, j)
                # Keep multiplication beside accumulation, including the
                # original float64 fused operation when enabled by Warp.
                total += wp.float64(v * w)
                weight += wp.float64(w)
        if lane == 0:
            out[0] = dtype(total / weight)

    @wp.func
    def _weighted_value(value: dtype, weight: dtype):
        # Preserve the input-precision product before double accumulation.
        return wp.float64(value * weight)

    @wp.func
    def _double_weight(weight: dtype):
        return wp.float64(weight)

    @wp.func
    def _weighted_mean(total: wp.float64, weight: wp.float64):
        return dtype(total / weight)

    @wp.kernel(module="unique")
    def _isovalue_partials_kernel(
        values: wp.array[dtype], weights: wp.array[dtype], totals: wp.array[wp.float64], masses: wp.array[wp.float64]
    ):
        block = wp.tid()
        v = wp.tile_load(values, shape=1024, offset=block * 1024)
        w = wp.tile_load(weights, shape=1024, offset=block * 1024)
        wp.tile_store(totals, wp.tile_sum(wp.tile_map(_weighted_value, v, w)), offset=block)
        wp.tile_store(masses, wp.tile_sum(wp.tile_map(_double_weight, w)), offset=block)

    @wp.kernel(module="unique")
    def _isovalue_finish_kernel(totals: wp.array[wp.float64], masses: wp.array[wp.float64], out: wp.array[dtype]):
        total = wp.tile_zeros(shape=1, dtype=wp.float64)
        mass = wp.tile_zeros(shape=1, dtype=wp.float64)
        for begin in range(0, totals.shape[0], 1024):
            total += wp.tile_sum(wp.tile_load(totals, shape=1024, offset=begin))
            mass += wp.tile_sum(wp.tile_load(masses, shape=1024, offset=begin))
        wp.tile_store(out, wp.tile_map(_weighted_mean, total, mass))

    return SimpleNamespace(
        _centroids=_centroids_kernel,
        _isovalue=_isovalue_kernel,
        _isovalue_tiled=_isovalue_tiled_kernel,
        _isovalue_partials=_isovalue_partials_kernel,
        _isovalue_finish=_isovalue_finish_kernel,
    )


def _compute_isovalue(values, weights):
    """Weighted mean with double partial sums and no host-side reduction."""
    kernels = precision_kernels(values.dtype)
    out = wp.empty(1, dtype=values.dtype, device=values.device)
    if len(values) <= 1024:
        wp.launch_tiled(
            kernels._isovalue_tiled, dim=1, inputs=[values, weights, out], block_dim=32, device=values.device
        )
    else:
        count = (len(values) + 1023) // 1024
        totals = wp.empty(count, dtype=wp.float64, device=values.device)
        masses = wp.empty_like(totals)
        wp.launch_tiled(
            kernels._isovalue_partials,
            dim=count,
            inputs=[values, weights, totals, masses],
            block_dim=128,
            device=values.device,
        )
        wp.launch_tiled(
            kernels._isovalue_finish, dim=1, inputs=[totals, masses, out], block_dim=128, device=values.device
        )
    return out


@dataclass
class Reconstruction:
    field: LevelField
    metadata: dict
    diagnostics: dict
    iso_value_array: wp.array = None

    @property
    def iso_value(self):
        return self.field.iso_value

    def evaluate(self, queries):
        return self.field.evaluate(queries)

    def device_field(self):
        """Return a zero-copy handle for ``poisson_evaluate`` in a Warp kernel."""
        return self.field.device_field()


def reconstruct(
    positions,
    normals,
    *,
    depth=6,
    full_depth=3,
    base_depth=None,
    solve_depth=None,
    kernel_depth=None,
    samples_per_node=1.5,
    scale=1.1,
    point_weight=2.0,
    exact_interpolation=False,
    iters=8,
    base_v_cycles=1,
    cg_accuracy=1e-3,
):
    """Reconstruct oriented CUDA float32/float64 vectors; point_weight=0 selects PSR.

    Degree one, Neumann boundaries, confidence disabled, library alignDir=0.
    The return value evaluates the implicit function and its model-space gradient;
    the surface is ``value == result.iso_value``. Mesh extraction is external.
    """
    if base_depth is None:
        base_depth = full_depth
    if solve_depth is None:
        solve_depth = depth
    if kernel_depth is None:
        kernel_depth = max(depth - 2, 0)
    depths = (depth, full_depth, base_depth, solve_depth, kernel_depth)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in depths):
        raise ValueError("Depth parameters must be integers")
    if not 0 <= base_depth <= full_depth <= solve_depth <= depth <= 15 or depth < 1 or not 0 <= kernel_depth <= depth:
        raise ValueError(
            "Require 0 <= base_depth <= full_depth <= solve_depth <= depth <= 15, depth >= 1, and 0 <= kernel_depth <= depth"
        )
    if depth - full_depth > 7:
        raise ValueError("AdaptiveNanogrid supports at most 8 integration levels; require depth - full_depth <= 7")
    if not isinstance(iters, int) or iters < 1 or not isinstance(base_v_cycles, int) or base_v_cycles < 1:
        raise ValueError("iters and base_v_cycles must be positive integers")
    if not math.isfinite(point_weight) or point_weight < 0 or not math.isfinite(cg_accuracy) or cg_accuracy <= 0:
        raise ValueError("Require finite nonnegative point_weight and positive cg_accuracy")
    if not math.isfinite(samples_per_node) or samples_per_node <= 0 or not math.isfinite(scale) or scale <= 1:
        raise ValueError("Require finite positive samples_per_node and scale > 1")
    if (
        not any(wp.types.types_equal(positions.dtype, v) for v in (wp.vec3, wp.vec3d))
        or not wp.types.types_equal(normals.dtype, positions.dtype)
        or not positions.device.is_cuda
        or normals.device != positions.device
    ):
        raise ValueError("Reconstruction requires matching float32/float64 3D vectors on the same CUDA device")
    device = positions.device
    dtype = wp.float64 if wp.types.types_equal(positions.dtype, wp.vec3d) else wp.float32
    kernels = precision_kernels(dtype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=dtype)
    stage_events = {}

    def stage(name, func):
        start = wp.Event(device, enable_timing=True)
        end = wp.Event(device, enable_timing=True)
        wp.record_event(start)
        result = func()
        wp.record_event(end)
        stage_events[name] = (start, end)
        return result

    with wp.ScopedDevice(device):
        samples = stage(
            "samples",
            lambda: prepare_samples(positions, normals, depth=depth, scale=scale),
        )
        density = stage(
            "density",
            lambda: build_density(samples, depth, kernel_depth, samples_per_node),
        )
        normal = stage(
            "normal_field",
            lambda: build_normal_field(samples, density, base_depth, depth),
        )
        tree = stage("tree", lambda: build_tree(normal, depth, full_depth, base_depth))
        correction, levels, solver_sync = stage(
            "assembly_and_solve",
            lambda: solve_levels(
                tree,
                None,
                normal,
                density,
                base_depth=base_depth,
                solve_depth=solve_depth,
                point_weight=point_weight,
                exact=exact_interpolation,
                iters=iters,
                base_v_cycles=base_v_cycles,
                cg_accuracy=cg_accuracy,
            ),
        )
        count = sum(len(level.keys) for level in tree.basis)
        keys = wp.empty(count, dtype=wp.int64, device=device)
        coefficients = wp.empty(count, dtype=dtype, device=device)
        offset = 0
        for d, level in enumerate(tree.basis):
            wp.copy(keys, level.keys, dest_offset=offset)
            wp.copy(coefficients, correction[d], dest_offset=offset)
            offset += len(level.keys)
        identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        cumulative = cumulative_coefficients(keys, coefficients, depth)
        cube_field = LevelField(keys, coefficients, identity, (0.0, 0.0, 0.0), depth, 0.0, cumulative)
        centroids = wp.empty(len(samples.keys), dtype=positions.dtype, device=device)
        wp.launch(
            kernels._centroids,
            len(centroids),
            inputs=[samples.position_sums, samples.weights, centroids],
            device=device,
        )
        value, _, _ = cube_field.evaluate(centroids)
        iso = _compute_isovalue(value, samples.weights)
        transforms = wp.empty(2, dtype=mat44, device=device)
        wp.launch(
            transform_kernels(dtype).evaluation_transform,
            1,
            inputs=[samples.transform, transforms],
            device=device,
        )
        start = time.perf_counter()
        transform = transforms.numpy().tolist()
        isovalue = float(iso.numpy()[0])
        status_sync = time.perf_counter() - start
        matrix = transform[1]
        field = LevelField(
            keys,
            coefficients,
            tuple(matrix[r][c] for r in range(3) for c in range(3)),
            tuple(matrix[r][3] for r in range(3)),
            depth,
            isovalue,
            cumulative,
        )
    metadata = {
        "iso_value": isovalue,
        "precision": dtype.__name__,
        "device": str(device),
        "basis_counts": [len(l.keys) for l in tree.basis],
        "integration_cells": next(
            (level["operator_integration_cells"] for level in levels if level["depth"] == depth), 0
        ),
        "integration_scope": "Materialized FEM cells; analytic stencils do not allocate an integration grid",
        "timings_seconds": {
            name: wp.get_event_elapsed_time(start, end, synchronize=False) / 1000
            for name, (start, end) in stage_events.items()
        },
        "stage_timing_scope": "CUDA stream elapsed time, including submission gaps; no stage-boundary synchronization",
        "operator_integration_cells": {level["depth"]: level["operator_integration_cells"] for level in levels},
        "measured_host_sync_seconds": samples.host_sync_seconds
        + density.sync_seconds
        + tree.host_sync_seconds
        + normal.records.sync_seconds
        + solver_sync
        + status_sync,
        "sync_measurement_scope": "sample/density/normal/tree count reads, solver row counts and CG residual checks, final metadata; excludes Warp-internal synchronization",
        "exact_interpolation_behavior": "empty constraints in pinned reference"
        if exact_interpolation
        else "level-clustered approximate constraints",
        "cube_to_model": transform[0],
        "parameters": {
            "depth": depth,
            "full_depth": full_depth,
            "base_depth": base_depth,
            "solve_depth": solve_depth,
            "kernel_depth": kernel_depth,
            "samples_per_node": samples_per_node,
            "scale": scale,
            "point_weight": point_weight,
            "exact_interpolation": exact_interpolation,
            "iters": iters,
            "base_v_cycles": base_v_cycles,
            "cg_accuracy": cg_accuracy,
        },
        "profile": "pinned cascadic degree-1 Neumann",
        "constant_mode": "reference root mean term"
        if base_depth == 0 and point_weight == 0
        else "reference solve without an added mean term",
    }
    return Reconstruction(
        field,
        metadata,
        {"samples": samples, "tree": tree, "density": density, "normals": normal, "levels": levels},
        iso,
    )
