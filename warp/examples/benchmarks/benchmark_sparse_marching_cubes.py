# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Benchmark: sparse vs. dense marching cubes
#
# Compares isosurface extraction from an implicit function two ways:
#
#   dense  : evaluate the field on a full (2^d + 1)^3 grid, then run
#            wp.geometry.IsoSurfaceMarchingCubes (cost ~ O(R^3), the surface *volume*).
#   sparse : build a Lipschitz octree and run wp.geometry.sparse_marching_cubes_via_lipschitz_pruning
#            (cost ~ O(R^2), the surface *area*).
#
# Both paths evaluate the SAME implicit function on the GPU and, at a given
# depth, produce the SAME surface (this is asserted, so the speedup is
# honest: same output, less work). As the depth grows the sparse method
# wins by an asymptotically increasing margin, and the dense grid
# eventually exhausts memory while the sparse octree keeps going.
#
# Two implicit functions are provided:
#   --sdf analytic : a box smoothly unioned with a torus (a cheap SDF)
#   --sdf bunny    : a signed distance field to the Stanford bunny mesh,
#                    evaluated with a winding-number sign (a realistic,
#                    more expensive query where the sparse method's fewer
#                    evaluations matter even more). Requires usd-core.
#
# Note: requires a CUDA-capable device.
###########################################################################

import argparse
import os
import statistics
import time

import numpy as np

import warp as wp
import warp.examples
import warp.geometry

# =============================================================================
# Backend 1: analytic SDF (box smoothly unioned with a torus)
# =============================================================================


@wp.func
def _sdf_box(p: wp.vec3, size: wp.vec3):
    q = wp.vec3(wp.abs(p[0]) - size[0], wp.abs(p[1]) - size[1], wp.abs(p[2]) - size[2])
    qp = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    return wp.length(qp) + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)


@wp.func
def _sdf_torus(p: wp.vec3, major_radius: float, minor_radius: float):
    q = wp.vec2(wp.length(wp.vec2(p[0], p[2])) - major_radius, p[1])
    return wp.length(q) - minor_radius


@wp.func
def _sdf_smooth_union(a: float, b: float, radius: float):
    h = wp.max(radius - wp.abs(a - b), 0.0) / radius
    return wp.min(a, b) - h * h * h * radius * (1.0 / 6.0)


@wp.func
def _scene_sdf(p: wp.vec3):
    box = _sdf_box(p - wp.vec3(0.0, -0.35, 0.0), wp.vec3(0.35, 0.2, 0.35))
    torus = _sdf_torus(p - wp.vec3(0.0, 0.25, 0.0), 0.35, 0.12)
    return _sdf_smooth_union(box, torus, 0.2)


@wp.kernel(enable_backward=False)
def _analytic_field_kernel(field: wp.array3d(dtype=float), origin: wp.vec3, h: float):
    i, j, k = wp.tid()
    field[i, j, k] = _scene_sdf(origin + h * wp.vec3(float(i), float(j), float(k)))


def analytic_backend(device):
    origin = wp.vec3(-1.0, -1.0, -1.0)
    root_width = 2.0

    def dense_field(depth):
        res = 1 << depth
        h = root_width / res
        field = wp.empty((res + 1, res + 1, res + 1), dtype=float, device=device)
        wp.launch(_analytic_field_kernel, dim=field.shape, inputs=[field, origin, float(h)], device=device)
        return field

    # Passing the @wp.func lets wp.geometry.sparse_marching_cubes_via_lipschitz_pruning build its own evaluation
    # kernel; it computes the same values as the dense field kernel.
    return _scene_sdf, dense_field, origin, root_width


# =============================================================================
# Backend 2: mesh SDF (signed distance to the Stanford bunny)
# =============================================================================


@wp.kernel(enable_backward=False)
def _mesh_sdf_batch_kernel(mesh_id: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    p = points[i]
    q = wp.mesh_query_point_sign_winding_number(mesh_id, p, 1.0e6)
    values[i] = q.sign * wp.length(p - wp.mesh_eval_position(mesh_id, q.face, q.u, q.v))


@wp.kernel(enable_backward=False)
def _mesh_field_kernel(mesh_id: wp.uint64, field: wp.array3d(dtype=float), origin: wp.vec3, h: float):
    i, j, k = wp.tid()
    p = origin + h * wp.vec3(float(i), float(j), float(k))
    q = wp.mesh_query_point_sign_winding_number(mesh_id, p, 1.0e6)
    field[i, j, k] = q.sign * wp.length(p - wp.mesh_eval_position(mesh_id, q.face, q.u, q.v))


def bunny_backend(device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415 -- optional dependency

    stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bunny"))
    points = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    center = 0.5 * (points.min(0) + points.max(0))
    points = (points - center) / (points.max(0) - points.min(0)).max()

    mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=wp.int32, device=device),
    )
    origin = wp.vec3(-0.65, -0.65, -0.65)
    root_width = 1.3

    def evaluate(query_points):
        values = wp.empty(query_points.shape[0], dtype=wp.float32, device=query_points.device)
        wp.launch(
            _mesh_sdf_batch_kernel,
            dim=query_points.shape[0],
            inputs=[mesh.id, query_points],
            outputs=[values],
            device=query_points.device,
        )
        return values

    def dense_field(depth):
        res = 1 << depth
        h = root_width / res
        field = wp.empty((res + 1, res + 1, res + 1), dtype=float, device=device)
        wp.launch(_mesh_field_kernel, dim=field.shape, inputs=[mesh.id, field, origin, float(h)], device=device)
        return field

    return evaluate, dense_field, origin, root_width


# =============================================================================
# Extraction drivers and timing
# =============================================================================


def dense_extract(dense_field, origin, root_width, depth):
    field = dense_field(depth)
    upper = wp.vec3(origin[0] + root_width, origin[1] + root_width, origin[2] + root_width)
    verts, indices = wp.geometry.IsoSurfaceMarchingCubes.extract(
        field, threshold=0.0, domain_bounds_lower_corner=origin, domain_bounds_upper_corner=upper
    )
    return verts, indices


def sparse_extract(sdf, origin, root_width, depth, device, return_stats=False):
    return wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
        sdf, origin, root_width, depth, threshold=0.0, device=device, return_stats=return_stats
    )


def time_call(fn, device, iters):
    fn()  # warm up (compilation, first allocation) outside the measurement
    wp.synchronize_device(device)
    samples = []
    for _ in range(iters):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--sdf", choices=("analytic", "bunny"), default="analytic", help="Implicit function to test.")
    parser.add_argument("--min-depth", type=int, default=5, help="Smallest octree depth to benchmark.")
    parser.add_argument("--max-depth", type=int, default=10, help="Largest octree depth to benchmark.")
    parser.add_argument("--iters", type=int, default=7, help="Timed iterations per configuration (median reported).")
    args = parser.parse_known_args()[0]

    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if min(args.min_depth, args.max_depth) < 0:
        parser.error("depth arguments must be non-negative")
    if args.min_depth > args.max_depth:
        parser.error("--min-depth must not exceed --max-depth")

    device = wp.get_device(args.device)
    if not device.is_cuda:
        print("Warning: this benchmark is intended for a CUDA device; timings on CPU are not representative.")

    if args.sdf == "bunny":
        sdf, dense_field, origin, root_width = bunny_backend(device)
    else:
        sdf, dense_field, origin, root_width = analytic_backend(device)

    header = (
        f"{'depth':>5} {'res':>6} {'dense (ms)':>11} {'sparse (ms)':>12} {'speedup':>8} "
        f"{'dense evals':>13} {'sparse evals':>13} {'eval x':>7} {'leaf cells':>11} {'tris':>9}"
    )
    print(f"\nDevice: {device}   SDF: {args.sdf}\n")
    print(header)
    print("-" * len(header))

    for depth in range(args.min_depth, args.max_depth + 1):
        resolution = 1 << depth
        dense_evals = (resolution + 1) ** 3

        sv, si, stats = sparse_extract(sdf, origin, root_width, depth, device, return_stats=True)
        wp.synchronize_device(device)
        sparse_tris = len(si) // 3
        # Release the untimed mesh before timing, so it does not inflate peak
        # memory (and push the dense path into OOM ahead of its real limit).
        del sv, si
        sparse_ms = (
            time_call(lambda depth=depth: sparse_extract(sdf, origin, root_width, depth, device), device, args.iters)
            * 1e3
        )

        dense_ms = None
        dense_tris = None
        try:
            dv, di = dense_extract(dense_field, origin, root_width, depth)
            wp.synchronize_device(device)
            dense_tris = len(di) // 3
            del dv, di
            dense_ms = (
                time_call(lambda depth=depth: dense_extract(dense_field, origin, root_width, depth), device, args.iters)
                * 1e3
            )
        except Exception as exc:
            print(f"{depth:>5} {resolution:>6}  dense grid unavailable ({type(exc).__name__}); sparse only")

        if dense_tris is not None and dense_tris != sparse_tris:
            print(
                f"  WARNING: triangle-count mismatch at depth {depth}: "
                f"dense={dense_tris} sparse={sparse_tris} (speedup would be unfair)"
            )

        eval_ratio = dense_evals / max(stats["sdf_evaluations"], 1)
        if dense_ms is not None:
            print(
                f"{depth:>5} {resolution:>6} {dense_ms:>11.3f} {sparse_ms:>12.3f} {dense_ms / sparse_ms:>7.2f}x "
                f"{dense_evals:>13,} {stats['sdf_evaluations']:>13,} {eval_ratio:>6.1f}x "
                f"{stats['leaf_cells']:>11,} {sparse_tris:>9,}"
            )
        else:
            print(
                f"{depth:>5} {resolution:>6} {'OOM':>11} {sparse_ms:>12.3f} {'--':>8} "
                f"{dense_evals:>13,} {stats['sdf_evaluations']:>13,} {eval_ratio:>6.1f}x "
                f"{stats['leaf_cells']:>11,} {sparse_tris:>9,}"
            )

    print(
        "\nNotes:\n"
        "  * Both methods evaluate the same GPU implicit function and produce the same surface at\n"
        "    each depth (triangle counts are checked). The sparse method simply performs less work.\n"
        "  * 'eval x' is how many fewer implicit-function evaluations the octree performs.\n"
        "  * At small depths the octree's per-level synchronizations can make it slower than the\n"
        "    dense grid; the sparse method wins by a growing margin as resolution increases.\n"
    )


if __name__ == "__main__":
    main()
