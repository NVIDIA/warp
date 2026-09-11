# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from collections import defaultdict

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import *

# =============================================================================
# Analytic implicit functions (signed distance functions)
# =============================================================================


@wp.func
def sphere_sdf(p: wp.vec3):
    return wp.length(p) - 0.5


@wp.func
def torus_sdf(p: wp.vec3):
    # Torus centered at the origin in the xz-plane.
    q = wp.vec2(wp.length(wp.vec2(p[0], p[2])) - 0.5, p[1])
    return wp.length(q) - 0.2


@wp.kernel(enable_backward=False)
def _sphere_batch_kernel(points: wp.array[wp.vec3], values: wp.array[wp.float32]):
    i = wp.tid()
    values[i] = wp.float32(sphere_sdf(points[i]))


@wp.kernel(enable_backward=False)
def _torus_batch_kernel(points: wp.array[wp.vec3], values: wp.array[wp.float32]):
    i = wp.tid()
    values[i] = wp.float32(torus_sdf(points[i]))


def sphere_evaluate(points):
    """Evaluate ``sphere_sdf`` in batches, entirely on-device.

    ``sparse_marching_cubes``/``lipschitz_octree`` only
    accept a batched callable, not a bare single-point ``@wp.func`` -- this is
    the pattern callers use to batch one, matching
    ``warp/examples/geometry/example_sparse_marching_cubes.py``.
    """
    values = wp.empty(points.shape[0], dtype=wp.float32, device=points.device)
    wp.launch(_sphere_batch_kernel, dim=points.shape[0], inputs=[points], outputs=[values], device=points.device)
    return values


def torus_evaluate(points):
    """Evaluate ``torus_sdf`` in batches, entirely on-device (see ``sphere_evaluate``)."""
    values = wp.empty(points.shape[0], dtype=wp.float32, device=points.device)
    wp.launch(_torus_batch_kernel, dim=points.shape[0], inputs=[points], outputs=[values], device=points.device)
    return values


@wp.kernel(enable_backward=False)
def sphere_field_kernel(field: wp.array3d[float], origin: wp.vec3, h: float):
    i, j, k = wp.tid()
    p = origin + h * wp.vec3(float(i), float(j), float(k))
    field[i, j, k] = wp.length(p) - 0.5


@wp.kernel(enable_backward=False)
def torus_field_kernel(field: wp.array3d[float], origin: wp.vec3, h: float):
    i, j, k = wp.tid()
    p = origin + h * wp.vec3(float(i), float(j), float(k))
    q = wp.vec2(wp.length(wp.vec2(p[0], p[2])) - 0.5, p[1])
    field[i, j, k] = wp.length(q) - 0.2


@wp.kernel(enable_backward=False)
def sphere_field_kernel_aniso(field: wp.array3d[float], lower: wp.vec3, delta: wp.vec3):
    i, j, k = wp.tid()
    p = lower + wp.cw_mul(delta, wp.vec3(float(i), float(j), float(k)))
    field[i, j, k] = wp.length(p) - 0.5


# =============================================================================
# Differentiable fixtures: parametrized-by-radius sphere, dense (field) and
# sparse (per-cell corner values), plus a shared surface-area reduction.
# =============================================================================

_CUBE_CORNER_OFFSETS = wp.geometry.IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS


@wp.kernel
def sphere_field_kernel_grad(field: wp.array3d[float], lower: wp.vec3, delta: wp.vec3, radius: wp.array[wp.float32]):
    i, j, k = wp.tid()
    p = lower + wp.cw_mul(delta, wp.vec3(float(i), float(j), float(k)))
    field[i, j, k] = wp.length(p) - radius[0]


@wp.kernel
def sphere_corner_values_kernel(
    cells: wp.array[wp.vec3i],
    lower: wp.vec3,
    delta: wp.vec3,
    radius: wp.array[wp.float32],
    out: wp.array2d[wp.float32],
):
    """Differentiable per-cell corner values for every cell of a dense grid.

    Corners are ordered per ``IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS``, as
    ``sparse_marching_cubes_from_cells`` expects.
    """
    cell = wp.tid()
    c = cells[cell]
    for corner in range(8):
        di = wp.static(_CUBE_CORNER_OFFSETS[corner][0])
        dj = wp.static(_CUBE_CORNER_OFFSETS[corner][1])
        dk = wp.static(_CUBE_CORNER_OFFSETS[corner][2])
        p = lower + wp.cw_mul(delta, wp.vec3(float(c[0] + di), float(c[1] + dj), float(c[2] + dk)))
        out[cell, corner] = wp.length(p) - radius[0]


@wp.kernel
def compute_surface_area_kernel(verts: wp.array[wp.vec3], faces: wp.array[wp.int32], out_area: wp.array[wp.float32]):
    """Heron's-formula surface area reduction, matching test_marching_cubes.py's ``compute_surface_area``."""
    tid = wp.tid()
    p0 = verts[faces[3 * tid + 0]]
    p1 = verts[faces[3 * tid + 1]]
    p2 = verts[faces[3 * tid + 2]]
    a = wp.length(p1 - p0)
    b = wp.length(p2 - p0)
    c = wp.length(p2 - p1)
    s = (a + b + c) / 2.0
    area = wp.sqrt(s * (s - a) * (s - b) * (s - c))
    wp.atomic_add(out_area, 0, area)


@wp.kernel
def sphere_batch_kernel_grad(points: wp.array[wp.vec3], radius: wp.array[wp.float32], out: wp.array[wp.float32]):
    i = wp.tid()
    out[i] = wp.length(points[i]) - radius[0]


def sphere_evaluate_grad(radius_wp):
    """Build a batched sphere evaluator parametrized by ``radius_wp``.

    For testing that gradient flows through sparse_marching_cubes. The output
    array must be allocated with ``requires_grad=True`` for Warp's
    tape to track it -- the same requirement as any other differentiable Warp
    kernel output; ``sparse_marching_cubes`` cannot infer this on the
    caller's behalf.
    """

    def evaluate(points):
        out = wp.empty(points.shape[0], dtype=wp.float32, device=points.device, requires_grad=True)
        wp.launch(
            sphere_batch_kernel_grad,
            dim=points.shape[0],
            inputs=[points, radius_wp],
            outputs=[out],
            device=points.device,
        )
        return out

    return evaluate


# =============================================================================
# Helpers
# =============================================================================


def _triangle_areas(verts, faces):
    p0 = verts[faces[:, 0]]
    p1 = verts[faces[:, 1]]
    p2 = verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _validate_mesh(test, verts, faces, check_nonempty=True):
    """Reproduce the structural invariants checked for dense marching cubes."""
    if check_nonempty:
        test.assertGreater(faces.shape[0], 0)
        test.assertGreater(verts.shape[0], 0)
    test.assertEqual(faces.shape[1], 3)
    test.assertTrue((faces >= 0).all())
    test.assertTrue((faces < verts.shape[0]).all())
    test.assertTrue((faces[:, 0] != faces[:, 1]).all())
    test.assertTrue((faces[:, 0] != faces[:, 2]).all())
    test.assertTrue((faces[:, 1] != faces[:, 2]).all())
    # every emitted vertex is referenced by at least one face
    test.assertTrue((np.unique(faces.flatten()) == np.arange(verts.shape[0])).all())
    test.assertTrue(np.isfinite(verts).all())


def _one_sided_hausdorff(a_points, b_points, bucket):
    """Compute the max over ``a`` of the nearest distance to ``b`` via a spatial hash.

    Exact set comparison is too brittle here: the ``@wp.func`` and the dense
    field kernel compile the same arithmetic with slightly different
    floating-point contraction, so shared vertices can differ by ~1e-4. This
    tolerance-based match is the robust way to assert the surfaces coincide.
    """
    grid = defaultdict(list)
    for b in b_points:
        key = tuple(np.floor(b / bucket).astype(np.int64))
        grid[key].append(b)

    max_dist = 0.0
    for a in a_points:
        cell = np.floor(a / bucket).astype(np.int64)
        best = np.inf
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for b in grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                        best = min(best, float(np.linalg.norm(a - b)))
        max_dist = max(max_dist, best)
    return max_dist


def _assert_surfaces_equivalent(test, sv, sf, dv, df, tol=1e-3):
    """Assert two meshes describe the same surface (counts, geometry, area)."""
    # Identical marching cubes case decisions everywhere.
    test.assertEqual(sf.shape[0], df.shape[0], "triangle count differs")
    test.assertEqual(sv.shape[0], dv.shape[0], "vertex count differs")
    # Every vertex of each mesh coincides with a vertex of the other.
    bucket = max(tol * 4.0, 1e-2)
    test.assertLess(_one_sided_hausdorff(sv, dv, bucket), tol)
    test.assertLess(_one_sided_hausdorff(dv, sv, bucket), tol)
    # Total surface area agrees.
    np.testing.assert_allclose(_triangle_areas(sv, sf).sum(), _triangle_areas(dv, df).sum(), rtol=1e-4)


def _edge_manifold_stats(faces):
    """Return (edge_count, boundary_edge_count, nonmanifold_edge_count)."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    n_edges = counts.shape[0]
    n_boundary = int((counts == 1).sum())
    n_nonmanifold = int((counts > 2).sum())
    return n_edges, n_boundary, n_nonmanifold


def _dense_surface(field_kernel, origin, root_width, max_depth, threshold, device):
    """Extract the reference surface from a dense grid at the finest resolution."""
    resolution = 1 << max_depth
    n_nodes = resolution + 1
    h = root_width / resolution
    field = wp.empty((n_nodes, n_nodes, n_nodes), dtype=float, device=device)
    wp.launch(field_kernel, dim=field.shape, inputs=[field, wp.vec3(origin), float(h)], device=device)
    upper = wp.vec3(origin[0] + root_width, origin[1] + root_width, origin[2] + root_width)
    verts, indices = wp.geometry.IsoSurfaceMarchingCubes.extract(
        field,
        threshold=threshold,
        lower=wp.vec3(origin),
        upper=upper,
    )
    return verts.numpy(), indices.numpy().reshape(-1, 3)


def _dense_surface_aniso(field_kernel, lower, upper, nx, ny, nz, threshold, device):
    """Extract the reference surface from an anisotropic dense grid."""
    lower = wp.vec3(lower)
    upper = wp.vec3(upper)
    delta = wp.vec3(
        (upper[0] - lower[0]) / (nx - 1),
        (upper[1] - lower[1]) / (ny - 1),
        (upper[2] - lower[2]) / (nz - 1),
    )
    field = wp.empty((nx, ny, nz), dtype=float, device=device)
    wp.launch(field_kernel, dim=field.shape, inputs=[field, lower, delta], device=device)
    verts, indices = wp.geometry.IsoSurfaceMarchingCubes.extract(
        field,
        threshold=threshold,
        lower=lower,
        upper=upper,
    )
    return verts.numpy(), indices.numpy().reshape(-1, 3)


# =============================================================================
# Tests
# =============================================================================


def _bounds_from_origin_width(origin, root_width):
    """Convert the legacy cubic (origin, root_width) parameterization to corner bounds."""
    lower = tuple(origin)
    upper = tuple(o + root_width for o in origin)
    return lower, upper


def _sparse_mc(sdf, origin, root_width, max_depth, **kwargs):
    """Call the new nx/ny/nz entry point with the cubic (origin, root_width, max_depth) shape.

    A thin adapter so the existing tests, which were written against the cubic
    octree parameterization, keep exercising the exact same grids after the
    signature change to match the dense `nx, ny, nz` + bounds convention.
    """
    n = (1 << max_depth) + 1
    lower, upper = _bounds_from_origin_width(origin, root_width)
    return wp.geometry.sparse_marching_cubes(sdf, n, n, n, lower=lower, upper=upper, **kwargs)


def test_sparse_mc_sphere(test, device):
    """Check that vertices lie on the sphere and that the mesh is structurally valid."""
    origin = (-1.0, -1.0, -1.0)
    verts, indices = _sparse_mc(sphere_evaluate, origin, 2.0, 6, device=device)
    v = verts.numpy()
    f = indices.numpy().reshape(-1, 3)
    _validate_mesh(test, v, f)

    radius = np.linalg.norm(v, axis=1)
    # Finest cell width is 2/64; interpolated vertices land well within it.
    test.assertLess(np.abs(radius - 0.5).max(), 2.0 / 64.0)

    area = _triangle_areas(v, f).sum()
    np.testing.assert_allclose(area, 4.0 * np.pi * 0.25, rtol=2e-2)


def test_sparse_mc_matches_dense(test, device):
    """Check that sparse extraction reproduces the dense surface at equal resolution.

    The finest octree leaves are a subset of the dense grid cells and share the
    same corner sampling and lookup tables, so the extracted triangles must be
    identical as a set. This is the fairness anchor for the benchmark: same
    surface, far fewer evaluations.
    """
    cases = [
        ("sphere", sphere_evaluate, sphere_field_kernel, (-1.0, -1.0, -1.0), 2.0, 5),
        ("sphere", sphere_evaluate, sphere_field_kernel, (-1.0, -1.0, -1.0), 2.0, 6),
        ("torus", torus_evaluate, torus_field_kernel, (-1.0, -1.0, -1.0), 2.0, 6),
    ]
    for name, evaluate, field_kernel, origin, width, depth in cases:
        with test.subTest(sdf=name, depth=depth):
            verts, indices = _sparse_mc(evaluate, origin, width, depth, device=device)
            sv = verts.numpy()
            sf = indices.numpy().reshape(-1, 3)
            dv, df = _dense_surface(field_kernel, origin, width, depth, 0.0, device)

            _validate_mesh(test, sv, sf)
            _assert_surfaces_equivalent(test, sv, sf, dv, df)


def test_sparse_mc_anisotropic_matches_dense(test, device):
    """Check that unequal, non-power-of-two nx/ny/nz and anisotropic bounds still match dense.

    ``nx, ny, nz`` here don't share a common power-of-two cell count, so the
    octree must pad each axis independently up to a shared ``max_depth`` and
    then cull the padding cells before extraction. This is the correctness
    proof that the padding + culling reconciliation exactly reproduces the
    dense grid, the whole point of matching the two APIs' parameters.
    """
    lower = (-1.0, -1.2, -0.9)
    upper = (1.0, 1.3, 1.1)
    for nx, ny, nz in ((11, 15, 21), (17, 17, 17), (9, 33, 13)):
        with test.subTest(nx=nx, ny=ny, nz=nz):
            verts, indices = wp.geometry.sparse_marching_cubes(
                sphere_evaluate,
                nx,
                ny,
                nz,
                lower=lower,
                upper=upper,
                device=device,
            )
            sv = verts.numpy()
            sf = indices.numpy().reshape(-1, 3)
            dv, df = _dense_surface_aniso(sphere_field_kernel_aniso, lower, upper, nx, ny, nz, 0.0, device)

            _validate_mesh(test, sv, sf)
            _assert_surfaces_equivalent(test, sv, sf, dv, df)


def test_sparse_mc_threshold(test, device):
    """Check that a non-zero isovalue extracts a concentric, larger sphere."""
    origin = (-1.5, -1.5, -1.5)
    threshold = 0.25  # sphere_sdf == 0.25 -> radius 0.75
    verts, indices = _sparse_mc(sphere_evaluate, origin, 3.0, 6, threshold=threshold, device=device)
    v = verts.numpy()
    f = indices.numpy().reshape(-1, 3)
    _validate_mesh(test, v, f)
    radius = np.linalg.norm(v, axis=1)
    test.assertLess(np.abs(radius - 0.75).max(), 3.0 / 64.0)


def test_sparse_mc_empty(test, device):
    """Check that a level set outside the domain yields an empty but valid mesh."""
    origin = (-1.0, -1.0, -1.0)
    # Isovalue -10 is never attained by the sphere SDF in this box.
    verts, indices = _sparse_mc(sphere_evaluate, origin, 2.0, 5, threshold=-10.0, device=device)
    v = verts.numpy()
    f = indices.numpy().reshape(-1, 3)
    test.assertEqual(v.shape[0], 0)
    test.assertEqual(f.shape[0], 0)


def test_sparse_mc_stats_fewer_evaluations(test, device):
    """Check that the Lipschitz octree evaluates far fewer points than a dense grid."""
    origin = (-1.0, -1.0, -1.0)
    max_depth = 7
    _, _, stats = _sparse_mc(sphere_evaluate, origin, 2.0, max_depth, return_stats=True, device=device)
    resolution = 1 << max_depth
    dense_evals = (resolution + 1) ** 3
    test.assertEqual(stats["resolution"], resolution)
    test.assertGreater(stats["leaf_cells"], 0)
    # nx = ny = nz = resolution + 1 is already an exact power-of-two-plus-one
    # grid, so the octree needs no padding and the cull pass is a no-op.
    test.assertEqual(stats["culled_cells"], 0)
    # Surface work is ~O(R^2); dense is O(R^3). Require a comfortable margin.
    test.assertLess(stats["field_evaluations"], dense_evals // 4)


def test_sparse_mc_no_padding_when_power_of_two_plus_one(test, device):
    """Confirm the anisotropic/cull generalization is a no-op for exact grids.

    ``nx = ny = nz = 2**depth + 1`` requires no power-of-two padding, so the
    leaf-cell and evaluation counts must exactly match what the pre-refactor
    cubic-only octree produced for the same depth. Fixed expected counts are a
    deterministic stand-in for a performance-regression test, since
    timing-based assertions are unreliable under parallel test execution on
    shared CI runners: if the generalized, anisotropic-capable code path ever
    adds overhead for the common isotropic case, these counts would move.
    """
    origin = (-1.0, -1.0, -1.0)
    for max_depth, expected_leaf_cells, expected_field_evaluations in (
        (5, 1304, 6275),
        (6, 5360, 24387),
        (7, 22088, 98443),
    ):
        with test.subTest(max_depth=max_depth):
            _, _, stats = _sparse_mc(sphere_evaluate, origin, 2.0, max_depth, return_stats=True, device=device)
            test.assertEqual(stats["culled_cells"], 0)
            test.assertEqual(stats["leaf_cells"], expected_leaf_cells)
            test.assertEqual(stats["field_evaluations"], expected_field_evaluations)


def test_sparse_mc_numpy_evaluator(test, device):
    """Check that an off-device (NumPy) batched evaluator is still correct.

    The batched-callable contract only requires the *returned* array to be a
    ``wp.array[wp.float32]`` on the query points' device -- a callable is
    free to round-trip through host memory (NumPy, PyTorch, ...) internally.
    This is slower (every call pays a device/host sync), but must produce the
    exact same surface as an all-device Warp evaluator.
    """

    def evaluate_numpy(points):
        p = points.numpy()  # device -> host sync
        values = np.linalg.norm(p, axis=1) - 0.5
        return wp.array(values.astype(np.float32), device=points.device)

    origin = (-1.0, -1.0, -1.0)
    v_np, f_np = _sparse_mc(evaluate_numpy, origin, 2.0, 6, device=device)
    v_ref, f_ref = _sparse_mc(sphere_evaluate, origin, 2.0, 6, device=device)

    _validate_mesh(test, v_np.numpy(), f_np.numpy().reshape(-1, 3))
    # Both paths evaluate the identical expression, so the meshes must coincide.
    _assert_surfaces_equivalent(
        test,
        v_np.numpy(),
        f_np.numpy().reshape(-1, 3),
        v_ref.numpy(),
        f_ref.numpy().reshape(-1, 3),
    )


def test_sparse_mc_mesh_sdf(test, device):
    """Extract a surface from a mesh-based SDF end to end.

    Builds a closed triangle mesh, then uses ``wp.mesh_query_point_sign_normal``
    as the batched implicit function. Reproduces the workflow of the libigl
    Lipschitz-octree tutorial on the GPU.
    """
    # Build a sphere mesh via dense marching cubes to use as the SDF source.
    dv, df = _dense_surface(sphere_field_kernel, (-1.0, -1.0, -1.0), 2.0, 6, 0.0, device)
    mesh = wp.Mesh(
        points=wp.array(dv, dtype=wp.vec3, device=device),
        indices=wp.array(df.flatten(), dtype=wp.int32, device=device),
    )

    @wp.kernel(enable_backward=False)
    def mesh_sdf_kernel(mesh_id: wp.uint64, points: wp.array[wp.vec3], out: wp.array[wp.float32]):
        i = wp.tid()
        p = points[i]
        query = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6)
        cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        out[i] = query.sign * wp.length(p - cp)

    def evaluate(points):
        out = wp.empty(points.shape[0], dtype=wp.float32, device=points.device)
        wp.launch(mesh_sdf_kernel, dim=points.shape[0], inputs=[mesh.id, points], outputs=[out], device=points.device)
        return out

    verts, indices = _sparse_mc(evaluate, (-1.0, -1.0, -1.0), 2.0, 6, device=device)
    v = verts.numpy()
    f = indices.numpy().reshape(-1, 3)
    _validate_mesh(test, v, f)
    # Recovered surface should still be a sphere of radius ~0.5.
    radius = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(np.median(radius), 0.5, atol=3.0 / 64.0)


@wp.kernel(enable_backward=False)
def _mesh_minus_sphere_kernel(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    sphere_radius: wp.float32,
    out: wp.array[wp.float32],
):
    """CSG subtraction of an analytic sphere from a mesh SDF, entirely on-device.

    No ``.numpy()``/host round trip anywhere in this kernel or the launch that
    wraps it below: this is the pattern ``sparse_marching_cubes``
    expects for an implicit function that should stay on the GPU end to end.
    """
    i = wp.tid()
    p = points[i]
    query = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6)
    cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    mesh_sdf = query.sign * wp.length(p - cp)
    sphere_sdf_val = wp.length(p) - sphere_radius
    # CSG "A minus B" for signed distances: max(d_A, -d_B).
    out[i] = wp.max(mesh_sdf, -sphere_sdf_val)


def test_sparse_mc_mesh_minus_sphere_on_device(test, device):
    """Extract a mesh SDF with a concentric sphere subtracted, evaluated entirely on the GPU.

    Demonstrates and checks the recommended pattern for composing implicit
    functions in Warp: a single kernel combines a mesh query
    (``wp.mesh_query_point_sign_normal``) with an analytic sphere via CSG
    subtraction, launched from a batched evaluator with no host round trip.
    Carving a sphere of a quarter the mesh's bounding-box radius out of its
    center turns the solid sphere into a hollow shell, so the extracted
    surface should be two disjoint, watertight, genus-0 spheres (outer and
    inner) rather than one.
    """
    # Build a sphere mesh via dense marching cubes to use as the SDF source
    # (bounding-box radius 0.5, so a quarter of that is 0.125).
    dv, df = _dense_surface(sphere_field_kernel, (-1.0, -1.0, -1.0), 2.0, 6, 0.0, device)
    mesh = wp.Mesh(
        points=wp.array(dv, dtype=wp.vec3, device=device),
        indices=wp.array(df.flatten(), dtype=wp.int32, device=device),
    )
    bbox_radius = 0.5
    sphere_radius = 0.25 * bbox_radius

    def evaluate(points):
        out = wp.empty(points.shape[0], dtype=wp.float32, device=points.device)
        wp.launch(
            _mesh_minus_sphere_kernel,
            dim=points.shape[0],
            inputs=[mesh.id, points, sphere_radius],
            outputs=[out],
            device=points.device,
        )
        return out

    verts, indices = _sparse_mc(evaluate, (-1.0, -1.0, -1.0), 2.0, 6, device=device)
    v = verts.numpy()
    f = indices.numpy().reshape(-1, 3)
    _validate_mesh(test, v, f)

    n_edges, n_boundary, n_nonmanifold = _edge_manifold_stats(f)
    test.assertEqual(n_boundary, 0, "watertight shell must have no boundary (hole) edges")
    test.assertEqual(n_nonmanifold, 0, "manifold shell must have no edges shared by >2 faces")
    # Two disjoint genus-0 spheres: Euler characteristic sums to 2 + 2 = 4.
    test.assertEqual(v.shape[0] - n_edges + f.shape[0], 4, "expected two disjoint sphere shells")

    # Every vertex lies on the outer (mesh) or inner (carved-out) sphere.
    radius = np.linalg.norm(v, axis=1)
    on_outer = np.abs(radius - bbox_radius) < 3.0 / 64.0
    on_inner = np.abs(radius - sphere_radius) < 3.0 / 64.0
    test.assertTrue(np.all(on_outer | on_inner))
    test.assertTrue(np.any(on_outer))
    test.assertTrue(np.any(on_inner))


def test_sparse_mc_watertight(test, device):
    """Check that a closed SDF strictly inside the domain yields a watertight, manifold mesh.

    A boundary edge (used by a single triangle) is exactly a hole; it would mean
    the octree dropped a surface-crossing cell or the shared-edge vertex
    de-duplication cracked. We also check the Euler characteristic to confirm the
    genus is recovered (sphere -> 2, torus -> 0).
    """
    cases = [
        ("sphere", sphere_evaluate, 2),  # genus 0
        ("torus", torus_evaluate, 0),  # genus 1
    ]
    for name, evaluate, euler in cases:
        with test.subTest(sdf=name):
            verts, indices = _sparse_mc(evaluate, (-1.0, -1.0, -1.0), 2.0, 6, device=device)
            v = verts.numpy()
            f = indices.numpy().reshape(-1, 3)
            _validate_mesh(test, v, f)

            n_edges, n_boundary, n_nonmanifold = _edge_manifold_stats(f)
            test.assertEqual(n_boundary, 0, "watertight surface must have no boundary (hole) edges")
            test.assertEqual(n_nonmanifold, 0, "manifold surface must have no edges shared by >2 faces")
            test.assertEqual(v.shape[0] - n_edges + f.shape[0], euler, "unexpected Euler characteristic")


def test_sparse_mc_from_cells(test, device):
    """Check that the explicit-cells entry point reproduces the octree-driven result.

    Exercises the decomposition Nicholas Sharp / Alec Jacobson described: the
    Lipschitz octree only *chooses* the cells, and sparse marching cubes runs on
    an explicit ``(cells, corner_values)`` list -- so the extractor can be driven
    directly from a marked voxel set (e.g. from a vision or generative model).
    """
    origin = (-1.0, -1.0, -1.0)
    root_width = 2.0
    depth = 6
    corner_offsets = np.array(wp.geometry.IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS, dtype=np.int32)  # (8, 3)

    # Reference surface from the full octree-driven path.
    v_ref, f_ref = _sparse_mc(sphere_evaluate, origin, root_width, depth, device=device)
    v_ref = v_ref.numpy()
    f_ref = f_ref.numpy().reshape(-1, 3)

    # Recover the octree cells, then sample the field at their corners ourselves.
    cells_wp, cell_width = wp.geometry.lipschitz_octree(
        sphere_evaluate, origin, root_width, max_depth=depth, device=device
    )
    cells = cells_wp.numpy()
    corner_pos = np.array(origin) + cell_width * (cells[:, None, :] + corner_offsets[None, :, :])
    corner_vals = (np.linalg.norm(corner_pos, axis=2) - 0.5).astype(np.float32)  # (N, 8)

    verts, indices = wp.geometry.sparse_marching_cubes_from_cells(
        cells, corner_vals, origin=origin, cell_width=float(cell_width), threshold=0.0, device=device
    )
    v = verts.numpy()
    f = indices.numpy().reshape(-1, 3)
    _validate_mesh(test, v, f)

    # Same surface as the octree-driven call, and watertight.
    test.assertEqual(v.shape[0], v_ref.shape[0])
    test.assertEqual(f.shape[0], f_ref.shape[0])
    _, n_boundary, _ = _edge_manifold_stats(f)
    test.assertEqual(n_boundary, 0)
    _assert_surfaces_equivalent(test, v, f, v_ref, f_ref)

    # Cells already resident on the device are consumed in place (no host round
    # trip), in either the vec3i or the (N, 3) int32 layout.
    corner_vals_wp = wp.array(corner_vals.reshape(-1), dtype=wp.float32, device=device)
    for cells_wp in (
        wp.array(cells, dtype=wp.vec3i, device=device),
        wp.array(cells, dtype=wp.int32, device=device),
    ):
        verts_d, indices_d = wp.geometry.sparse_marching_cubes_from_cells(
            cells_wp, corner_vals_wp, origin=origin, cell_width=float(cell_width), device=device
        )
        np.testing.assert_allclose(verts_d.numpy(), v, atol=0.0)
        np.testing.assert_array_equal(indices_d.numpy(), indices.numpy())

    # Arbitrary (large, negative) subscripts with a compensating origin must give
    # the same world-space surface, confirming the offset-relative corner packing.
    shift = np.array([1000, -500, 7], dtype=np.int32)
    origin_shifted = tuple(np.array(origin) - shift * cell_width)
    verts_s, _ = wp.geometry.sparse_marching_cubes_from_cells(
        cells + shift, corner_vals, origin=origin_shifted, cell_width=float(cell_width), device=device
    )
    vs = verts_s.numpy()
    test.assertEqual(vs.shape[0], v_ref.shape[0])
    np.testing.assert_allclose(np.sort(np.linalg.norm(vs, axis=1)), np.sort(np.linalg.norm(v_ref, axis=1)), atol=1e-4)


def _sphere_area_grad_dense(node_dim, radius, device):
    """Compute d(area)/d(radius) of a sphere via dense marching cubes (for cross-checking)."""
    lower = wp.vec3(-1.0, -1.0, -1.0)
    upper = wp.vec3(1.0, 1.0, 1.0)
    delta = wp.vec3(2.0 / (node_dim - 1), 2.0 / (node_dim - 1), 2.0 / (node_dim - 1))
    radius_wp = wp.full((1,), value=radius, dtype=wp.float32, device=device, requires_grad=True)

    with wp.Tape() as tape:
        field = wp.zeros((node_dim, node_dim, node_dim), dtype=float, device=device, requires_grad=True)
        wp.launch(sphere_field_kernel_grad, dim=field.shape, inputs=[field, lower, delta, radius_wp], device=device)
        verts, faces = wp.geometry.IsoSurfaceMarchingCubes.extract(field, threshold=0.0, lower=lower, upper=upper)
        area = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        wp.launch(compute_surface_area_kernel, dim=faces.shape[0] // 3, inputs=[verts, faces, area], device=device)

    tape.backward(area)
    return float(area.numpy()[0]), float(radius_wp.grad.numpy()[0])


def _sphere_area_grad_sparse(node_dim, radius, device):
    """Compute d(area)/d(radius) of a sphere via sparse_marching_cubes_from_cells.

    On every cell of the same dense grid (no octree --
    lipschitz_octree is not differentiated).
    """
    lower = wp.vec3(-1.0, -1.0, -1.0)
    upper = wp.vec3(1.0, 1.0, 1.0)
    ncells = node_dim - 1
    delta = wp.vec3((upper[0] - lower[0]) / ncells, (upper[1] - lower[1]) / ncells, (upper[2] - lower[2]) / ncells)
    radius_wp = wp.full((1,), value=radius, dtype=wp.float32, device=device, requires_grad=True)

    grid = np.arange(ncells)
    cells_np = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.int32)
    cells = wp.array(cells_np, dtype=wp.vec3i, device=device)
    n_cells = cells.shape[0]

    with wp.Tape() as tape:
        corner_values = wp.empty((n_cells, 8), dtype=wp.float32, device=device, requires_grad=True)
        wp.launch(
            sphere_corner_values_kernel,
            dim=n_cells,
            inputs=[cells, lower, delta, radius_wp],
            outputs=[corner_values],
            device=device,
        )
        verts, faces = wp.geometry.sparse_marching_cubes_from_cells(
            cells, corner_values, origin=lower, cell_width=float(delta[0]), threshold=0.0, device=device
        )
        area = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        wp.launch(compute_surface_area_kernel, dim=faces.shape[0] // 3, inputs=[verts, faces, area], device=device)

    tape.backward(area)
    return float(area.numpy()[0]), float(radius_wp.grad.numpy()[0])


def test_sparse_mc_from_cells_differentiable(test, device):
    """Check that sparse_marching_cubes_from_cells has reasonable gradients.

    Mirrors ``test_marching_cubes_differentiable`` in test_marching_cubes.py:
    constructs a sphere via a differentiable per-cell corner-value kernel
    (every cell of a dense grid -- not an octree, since
    lipschitz_octree is deliberately not differentiated),
    extracts a surface, computes its area, and differentiates the area with
    respect to the sphere's radius.
    """
    node_dim = 33
    radius = 0.5
    area, grad = _sphere_area_grad_sparse(node_dim, radius, device)
    test.assertLess(abs(area - 4.0 * np.pi * radius * radius), 5e-2)
    test.assertLess(abs(grad - 8.0 * np.pi * radius), 5e-1)


def test_sparse_mc_gradient_matches_dense(test, device):
    """Check that sparse and dense marching cubes give matching d(area)/d(radius).

    Now that sparse_marching_cubes_from_cells and IsoSurfaceMarchingCubes.extract
    agree exactly on the mesh for the same grid (test_sparse_mc_matches_dense),
    their gradients should agree too -- the direct cross-implementation
    regression the nx/ny/nz-matching work enables.
    """
    node_dim = 33
    radius = 0.5
    area_dense, grad_dense = _sphere_area_grad_dense(node_dim, radius, device)
    area_sparse, grad_sparse = _sphere_area_grad_sparse(node_dim, radius, device)
    np.testing.assert_allclose(area_sparse, area_dense, rtol=1e-3)
    np.testing.assert_allclose(grad_sparse, grad_dense, rtol=1e-2)


def test_sparse_mc_gradient_deterministic(test, device):
    """Check that the gather-based corner-value gradient doesn't have an arbitrary-winner problem.

    The scatter this replaced did a plain (non-atomic) many-to-one write; an
    empirical check showed Warp's generated backward gives the *entire*
    downstream gradient to whichever write happened to run last, and zero to
    the others -- correct per forward semantics, but on the GPU the winning
    write is not guaranteed deterministic across launches, so which corner
    "wins" (and thus the gradient value itself) could silently vary run to
    run. The replacement gather picks one deterministic canonical source per
    unique corner instead, so repeated calls with identical inputs should
    match closely -- differing only by ordinary GPU floating-point
    non-associativity (e.g. reduction order), not by an arbitrary duplicate
    winning.
    """
    node_dim = 17  # small; this checks determinism, not gradient accuracy
    radius = 0.5
    _, grad1 = _sphere_area_grad_sparse(node_dim, radius, device)
    _, grad2 = _sphere_area_grad_sparse(node_dim, radius, device)
    np.testing.assert_allclose(grad1, grad2, rtol=1e-5)


def test_sparse_mc_via_lipschitz_pruning_differentiable(test, device):
    """Check that sparse_marching_cubes inherits differentiability, with no tape warnings.

    It reuses sparse_marching_cubes_from_cells's extraction core directly on
    already-deduplicated corner values, so gradient flows through as soon as
    the caller's ``field`` evaluator itself returns a requires_grad array --
    with no changes needed to the octree machinery. That octree machinery
    (lipschitz_octree) is still not differentiated; its
    kernels are launched with record_tape=False, so they never appear on the
    wp.Tape() at all and recording+running this call inside one prints no
    'enable_backward=False' warnings, unlike a plain enable_backward=False
    kernel that gets recorded and then skipped.
    """
    radius = 0.5
    radius_wp = wp.full((1,), value=radius, dtype=wp.float32, device=device, requires_grad=True)
    evaluate = sphere_evaluate_grad(radius_wp)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with wp.Tape() as tape:
            verts, indices = wp.geometry.sparse_marching_cubes(
                evaluate,
                33,
                33,
                33,
                lower=(-1.0, -1.0, -1.0),
                upper=(1.0, 1.0, 1.0),
                device=device,
            )
            area = wp.zeros(1, dtype=float, device=device, requires_grad=True)
            wp.launch(
                compute_surface_area_kernel, dim=indices.shape[0] // 3, inputs=[verts, indices, area], device=device
            )

        tape.backward(area)

    test.assertEqual(len(caught), 0, f"Expected no warnings, got: {[str(w.message) for w in caught]}")

    grad = float(radius_wp.grad.numpy()[0])
    test.assertLess(abs(grad - 8.0 * np.pi * radius), 5e-1)


def test_lipschitz_octree_compose_with_from_cells(test, device):
    """Verify lipschitz_octree()'s cells feed sparse_marching_cubes_from_cells() directly.

    lipschitz_octree() used to return world-space cell origins (wp.vec3),
    which required rounding to recover integer subscripts before calling
    sparse_marching_cubes_from_cells() -- a lossy round trip once subscripts
    exceed float32's exact-integer range. It now returns the same wp.vec3i
    subscripts sparse_marching_cubes_from_cells() expects, so the array can be
    passed straight through with no conversion and no host round trip.
    """
    origin = (-1.0, -1.0, -1.0)
    root_width = 2.0
    depth = 5
    corner_offsets = np.array(wp.geometry.IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS, dtype=np.int32)

    cells_wp, cell_width = wp.geometry.lipschitz_octree(sphere_evaluate, origin, root_width, depth, device=device)
    test.assertEqual(cells_wp.dtype, wp.vec3i)

    cells = cells_wp.numpy()
    corner_pos = np.array(origin) + cell_width * (cells[:, None, :] + corner_offsets[None, :, :])
    corner_vals = wp.array((np.linalg.norm(corner_pos, axis=2) - 0.5).astype(np.float32).reshape(-1), device=device)

    # cells_wp is passed straight through -- no .numpy()/np.round() conversion.
    verts, indices = wp.geometry.sparse_marching_cubes_from_cells(
        cells_wp, corner_vals, origin=origin, cell_width=float(cell_width), device=device
    )
    _validate_mesh(test, verts.numpy(), indices.numpy().reshape(-1, 3))


def test_sparse_mc_large_subscripts(test, device):
    """Check that subscripts beyond float32's exact-integer range still resolve.

    Corner positions are decoded from subscripts relative to the cell set's
    minimum, with the offset folded into the origin in double precision. Adding
    the absolute subscript in float32 instead would round adjacent corners onto
    the same position once subscripts pass 2**24, collapsing triangles.
    """
    origin = (-1.0, -1.0, -1.0)
    cell_width = 2.0 / 32
    corner_offsets = np.array(wp.geometry.IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS, dtype=np.int32)

    # Cells straddling a radius-0.5 sphere on a 32^3 grid.
    grid = np.arange(32)
    cells = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.int32)
    corner_pos = np.array(origin) + cell_width * (cells[:, None, :] + corner_offsets[None, :, :])
    corner_vals = (np.linalg.norm(corner_pos, axis=2) - 0.5).astype(np.float32)
    keep = (corner_vals.min(axis=1) < 0.0) & (corner_vals.max(axis=1) >= 0.0)
    cells, corner_vals = cells[keep], corner_vals[keep]

    verts, indices = wp.geometry.sparse_marching_cubes_from_cells(
        cells, corner_vals, origin=origin, cell_width=cell_width, threshold=0.0, device=device
    )
    v_ref, f_ref = verts.numpy(), indices.numpy().reshape(-1, 3)

    # The same surface, addressed with subscripts well past 2**24 and a
    # compensating origin, must come out identical to within float32 rounding.
    #
    # The shifts are multiples of 16 so that ``shift * cell_width`` is a whole
    # number and the compensating origin stays exactly representable: ``origin``
    # is itself a float32 wp.vec3, so a shift that pushed it off the float32 grid
    # would move the surface by up to half an ULP regardless of how the corner
    # subscripts are decoded.
    for shift in (1 << 24, (1 << 25) + 12352):
        origin_shifted = tuple(np.array(origin) - shift * cell_width)
        verts_s, indices_s = wp.geometry.sparse_marching_cubes_from_cells(
            cells + np.int32(shift),
            corner_vals,
            origin=origin_shifted,
            cell_width=cell_width,
            threshold=0.0,
            device=device,
        )
        vs, fs = verts_s.numpy(), indices_s.numpy().reshape(-1, 3)

        test.assertEqual(vs.shape[0], v_ref.shape[0], f"vertex count changed at shift {shift}")
        np.testing.assert_array_equal(fs, f_ref)
        np.testing.assert_allclose(vs, v_ref, atol=1e-4)

        # No triangle may collapse to a degenerate sliver.
        for a, b in ((0, 1), (1, 2), (0, 2)):
            test.assertFalse(
                np.any(np.all(vs[fs[:, a]] == vs[fs[:, b]], axis=1)),
                f"degenerate triangles at shift {shift}",
            )


def test_sparse_mc_noncontiguous_corner_values(test, device):
    """Check that a strided view of corner values is accepted.

    A caller slicing corner values out of a larger device-resident structure
    hands us a non-contiguous array, which cannot be reshaped directly.
    """
    origin = (-1.0, -1.0, -1.0)
    depth = 5
    corner_offsets = np.array(wp.geometry.IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS, dtype=np.int32)

    cells_wp, cell_width = wp.geometry.lipschitz_octree(sphere_evaluate, origin, 2.0, max_depth=depth, device=device)
    cells = cells_wp.numpy()
    corner_pos = np.array(origin) + cell_width * (cells[:, None, :] + corner_offsets[None, :, :])
    corner_vals = (np.linalg.norm(corner_pos, axis=2) - 0.5).astype(np.float32)

    v_ref, f_ref = wp.geometry.sparse_marching_cubes_from_cells(
        cells, corner_vals, origin=origin, cell_width=float(cell_width), device=device
    )

    # Interleave the values into a wider buffer, then take a strided view of it.
    padded = np.repeat(corner_vals[:, :, None], 2, axis=2)
    strided = wp.array(np.ascontiguousarray(padded), dtype=wp.float32, device=device)[:, :, 0]
    test.assertFalse(strided.is_contiguous)

    verts, indices = wp.geometry.sparse_marching_cubes_from_cells(
        cells, strided, origin=origin, cell_width=float(cell_width), device=device
    )
    np.testing.assert_array_equal(verts.numpy(), v_ref.numpy())
    np.testing.assert_array_equal(indices.numpy(), f_ref.numpy())


def test_lipschitz_octree_brackets_surface(test, device):
    """Check that the octree keeps every cell the surface actually passes through.

    Correctness of the sparse extractor rests on this conservative guarantee:
    the returned leaves must be a superset of the dense grid cells that contain
    a sign change, or the surface would develop holes.
    """
    origin = (-1.0, -1.0, -1.0)
    root_width = 2.0
    max_depth = 5
    resolution = 1 << max_depth

    cells_wp, cell_width = wp.geometry.lipschitz_octree(sphere_evaluate, origin, root_width, max_depth, device=device)
    cells = cells_wp.numpy()
    test.assertGreater(cells.shape[0], 0)
    np.testing.assert_allclose(cell_width, root_width / resolution)

    # Every kept cell's center is within the Lipschitz band of the surface.
    origins = np.array(origin) + cell_width * cells
    centers = origins + 0.5 * cell_width
    band = np.sqrt(3.0) / 2.0 * cell_width
    center_sdf = np.linalg.norm(centers, axis=1) - 0.5
    test.assertLessEqual(np.abs(center_sdf).max(), band + 1e-5)

    # Confirm completeness against a dense grid.
    kept = {tuple(c) for c in cells}

    n_nodes = resolution + 1
    xs = origin[0] + cell_width * np.arange(n_nodes)
    ys = origin[1] + cell_width * np.arange(n_nodes)
    zs = origin[2] + cell_width * np.arange(n_nodes)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    field = np.sqrt(gx**2 + gy**2 + gz**2) - 0.5
    inside = field < 0.0
    # A cell has a crossing if its 8 corners are not all inside or all outside.
    c = inside
    corner_sum = (
        c[:-1, :-1, :-1].astype(int)
        + c[1:, :-1, :-1]
        + c[:-1, 1:, :-1]
        + c[1:, 1:, :-1]
        + c[:-1, :-1, 1:]
        + c[1:, :-1, 1:]
        + c[:-1, 1:, 1:]
        + c[1:, 1:, 1:]
    )
    crossing = np.argwhere((corner_sum > 0) & (corner_sum < 8))
    missing = [tuple(ijk) for ijk in crossing if tuple(ijk) not in kept]
    test.assertEqual(len(missing), 0, f"{len(missing)} surface cells were not bracketed by the octree")


def test_sparse_mc_invalid_arguments(test, device):
    """Check that argument validation raises informative errors."""
    with test.assertRaises(ValueError):
        wp.geometry.sparse_marching_cubes(sphere_evaluate, 1, 17, 17, device=device)
    with test.assertRaises(ValueError):
        wp.geometry.sparse_marching_cubes(sphere_evaluate, 17, 0, 17, device=device)
    with test.assertRaises(ValueError):
        wp.geometry.sparse_marching_cubes(sphere_evaluate, 17, 17, 17, lipschitz_bound=-1.0, device=device)
    with test.assertRaises(ValueError):
        wp.geometry.lipschitz_octree(
            sphere_evaluate, (0.0, 0.0, 0.0), 2.0, max_depth=4, lipschitz_bound=-1.0, device=device
        )
    with test.assertRaises(TypeError):
        wp.geometry.sparse_marching_cubes(42, 17, 17, 17, device=device)
    # A bare single-point @wp.func is rejected -- only a batched callable is
    # accepted, so that the on-GPU-or-not choice is explicit to the caller.
    with test.assertRaises(TypeError):
        wp.geometry.sparse_marching_cubes(sphere_sdf, 17, 17, 17, device=device)
    with test.assertRaises(TypeError):
        wp.geometry.lipschitz_octree(sphere_sdf, (0.0, 0.0, 0.0), 2.0, max_depth=4, device=device)

    # A non-positive cell width collapses every corner onto the origin (or
    # mirrors the cell), so the explicit-cells entry point rejects it.
    cells = np.zeros((1, 3), dtype=np.int32)
    corner_values = np.zeros((1, 8), dtype=np.float32)
    for bad_width in (0.0, -1.0):
        with test.assertRaises(ValueError):
            wp.geometry.sparse_marching_cubes_from_cells(cells, corner_values, cell_width=bad_width, device=device)

    # An empty cell array still validates corner_values -- a mismatched size
    # is rejected rather than silently ignored just because there are no
    # cells to attach it to.
    with test.assertRaises(ValueError):
        wp.geometry.sparse_marching_cubes_from_cells(
            np.zeros((0, 3), dtype=np.int32), np.zeros((3,), dtype=np.float32), device=device
        )


devices = get_test_devices()


class TestSparseMarchingCubes(unittest.TestCase):
    pass


add_function_test(TestSparseMarchingCubes, "test_sparse_mc_sphere", test_sparse_mc_sphere, devices=devices)
add_function_test(
    TestSparseMarchingCubes, "test_sparse_mc_matches_dense", test_sparse_mc_matches_dense, devices=devices
)
add_function_test(
    TestSparseMarchingCubes,
    "test_sparse_mc_anisotropic_matches_dense",
    test_sparse_mc_anisotropic_matches_dense,
    devices=devices,
)
add_function_test(TestSparseMarchingCubes, "test_sparse_mc_threshold", test_sparse_mc_threshold, devices=devices)
add_function_test(TestSparseMarchingCubes, "test_sparse_mc_empty", test_sparse_mc_empty, devices=devices)
add_function_test(
    TestSparseMarchingCubes,
    "test_sparse_mc_stats_fewer_evaluations",
    test_sparse_mc_stats_fewer_evaluations,
    devices=devices,
)
add_function_test(
    TestSparseMarchingCubes,
    "test_sparse_mc_no_padding_when_power_of_two_plus_one",
    test_sparse_mc_no_padding_when_power_of_two_plus_one,
    devices=devices,
)
add_function_test(
    TestSparseMarchingCubes, "test_sparse_mc_numpy_evaluator", test_sparse_mc_numpy_evaluator, devices=devices
)
add_function_test(TestSparseMarchingCubes, "test_sparse_mc_mesh_sdf", test_sparse_mc_mesh_sdf, devices=devices)
add_function_test(
    TestSparseMarchingCubes,
    "test_sparse_mc_mesh_minus_sphere_on_device",
    test_sparse_mc_mesh_minus_sphere_on_device,
    devices=devices,
)
add_function_test(TestSparseMarchingCubes, "test_sparse_mc_watertight", test_sparse_mc_watertight, devices=devices)
add_function_test(TestSparseMarchingCubes, "test_sparse_mc_from_cells", test_sparse_mc_from_cells, devices=devices)
add_function_test(
    TestSparseMarchingCubes,
    "test_sparse_mc_from_cells_differentiable",
    test_sparse_mc_from_cells_differentiable,
    devices=devices,
)
add_function_test(
    TestSparseMarchingCubes,
    "test_sparse_mc_gradient_matches_dense",
    test_sparse_mc_gradient_matches_dense,
    devices=devices,
)
add_function_test(
    TestSparseMarchingCubes,
    "test_sparse_mc_gradient_deterministic",
    test_sparse_mc_gradient_deterministic,
    devices=devices,
)
add_function_test(
    TestSparseMarchingCubes,
    "test_sparse_mc_via_lipschitz_pruning_differentiable",
    test_sparse_mc_via_lipschitz_pruning_differentiable,
    devices=devices,
)
add_function_test(
    TestSparseMarchingCubes,
    "test_lipschitz_octree_compose_with_from_cells",
    test_lipschitz_octree_compose_with_from_cells,
    devices=devices,
)
add_function_test(
    TestSparseMarchingCubes, "test_sparse_mc_large_subscripts", test_sparse_mc_large_subscripts, devices=devices
)
add_function_test(
    TestSparseMarchingCubes,
    "test_sparse_mc_noncontiguous_corner_values",
    test_sparse_mc_noncontiguous_corner_values,
    devices=devices,
)
add_function_test(
    TestSparseMarchingCubes,
    "test_lipschitz_octree_brackets_surface",
    test_lipschitz_octree_brackets_surface,
    devices=devices,
)
add_function_test(
    TestSparseMarchingCubes, "test_sparse_mc_invalid_arguments", test_sparse_mc_invalid_arguments, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
