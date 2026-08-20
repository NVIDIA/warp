# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
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
def sphere_field_kernel(field: wp.array3d(dtype=float), origin: wp.vec3, h: float):
    i, j, k = wp.tid()
    p = origin + h * wp.vec3(float(i), float(j), float(k))
    field[i, j, k] = wp.length(p) - 0.5


@wp.kernel(enable_backward=False)
def torus_field_kernel(field: wp.array3d(dtype=float), origin: wp.vec3, h: float):
    i, j, k = wp.tid()
    p = origin + h * wp.vec3(float(i), float(j), float(k))
    q = wp.vec2(wp.length(wp.vec2(p[0], p[2])) - 0.5, p[1])
    field[i, j, k] = wp.length(q) - 0.2


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
    """Max over ``a`` of the nearest distance to ``b`` via a spatial hash.

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
    verts, indices = wp.geometry.IsoSurfaceMarchingCubes.extract_surface_marching_cubes(
        field,
        threshold=threshold,
        domain_bounds_lower_corner=wp.vec3(origin),
        domain_bounds_upper_corner=upper,
    )
    return verts.numpy(), indices.numpy().reshape(-1, 3)


# =============================================================================
# Tests
# =============================================================================


def test_sparse_mc_sphere(test, device):
    """Check that vertices lie on the sphere and that the mesh is structurally valid."""
    origin = (-1.0, -1.0, -1.0)
    verts, indices = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
        sphere_sdf, origin, 2.0, max_depth=6, device=device
    )
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
        (sphere_sdf, sphere_field_kernel, (-1.0, -1.0, -1.0), 2.0, 5),
        (sphere_sdf, sphere_field_kernel, (-1.0, -1.0, -1.0), 2.0, 6),
        (torus_sdf, torus_field_kernel, (-1.0, -1.0, -1.0), 2.0, 6),
    ]
    for sdf, field_kernel, origin, width, depth in cases:
        with test.subTest(sdf=sdf.key, depth=depth):
            verts, indices = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
                sdf, origin, width, max_depth=depth, device=device
            )
            sv = verts.numpy()
            sf = indices.numpy().reshape(-1, 3)
            dv, df = _dense_surface(field_kernel, origin, width, depth, 0.0, device)

            _validate_mesh(test, sv, sf)
            _assert_surfaces_equivalent(test, sv, sf, dv, df)


def test_sparse_mc_threshold(test, device):
    """Check that a non-zero isovalue extracts a concentric, larger sphere."""
    origin = (-1.5, -1.5, -1.5)
    threshold = 0.25  # sphere_sdf == 0.25 -> radius 0.75
    verts, indices = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
        sphere_sdf, origin, 3.0, max_depth=6, threshold=threshold, device=device
    )
    v = verts.numpy()
    f = indices.numpy().reshape(-1, 3)
    _validate_mesh(test, v, f)
    radius = np.linalg.norm(v, axis=1)
    test.assertLess(np.abs(radius - 0.75).max(), 3.0 / 64.0)


def test_sparse_mc_empty(test, device):
    """Check that a level set outside the domain yields an empty but valid mesh."""
    origin = (-1.0, -1.0, -1.0)
    # Isovalue -10 is never attained by the sphere SDF in this box.
    verts, indices = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
        sphere_sdf, origin, 2.0, max_depth=5, threshold=-10.0, device=device
    )
    v = verts.numpy()
    f = indices.numpy().reshape(-1, 3)
    test.assertEqual(v.shape[0], 0)
    test.assertEqual(f.shape[0], 0)


def test_sparse_mc_stats_fewer_evaluations(test, device):
    """Check that the Lipschitz octree evaluates far fewer points than a dense grid."""
    origin = (-1.0, -1.0, -1.0)
    max_depth = 7
    _, _, stats = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
        sphere_sdf, origin, 2.0, max_depth=max_depth, return_stats=True, device=device
    )
    resolution = 1 << max_depth
    dense_evals = (resolution + 1) ** 3
    test.assertEqual(stats["resolution"], resolution)
    test.assertGreater(stats["leaf_cells"], 0)
    # Surface work is ~O(R^2); dense is O(R^3). Require a comfortable margin.
    test.assertLess(stats["sdf_evaluations"], dense_evals // 4)


def test_sparse_mc_python_callable(test, device):
    """Check that the batched Python-callable interface matches the @wp.func interface."""

    @wp.kernel(enable_backward=False)
    def batched_sphere(points: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        out[i] = wp.length(points[i]) - 0.5

    def evaluate(points):
        out = wp.empty(points.shape[0], dtype=wp.float32, device=points.device)
        wp.launch(batched_sphere, dim=points.shape[0], inputs=[points], outputs=[out], device=points.device)
        return out

    origin = (-1.0, -1.0, -1.0)
    v_func, f_func = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
        sphere_sdf, origin, 2.0, max_depth=6, device=device
    )
    v_call, f_call = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
        evaluate, origin, 2.0, max_depth=6, device=device
    )

    # Both paths evaluate the identical expression, so the meshes must coincide.
    _assert_surfaces_equivalent(
        test,
        v_func.numpy(),
        f_func.numpy().reshape(-1, 3),
        v_call.numpy(),
        f_call.numpy().reshape(-1, 3),
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
    def mesh_sdf_kernel(mesh_id: wp.uint64, points: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        p = points[i]
        query = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6)
        cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        out[i] = query.sign * wp.length(p - cp)

    def evaluate(points):
        out = wp.empty(points.shape[0], dtype=wp.float32, device=points.device)
        wp.launch(mesh_sdf_kernel, dim=points.shape[0], inputs=[mesh.id, points], outputs=[out], device=points.device)
        return out

    verts, indices = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
        evaluate, (-1.0, -1.0, -1.0), 2.0, max_depth=6, device=device
    )
    v = verts.numpy()
    f = indices.numpy().reshape(-1, 3)
    _validate_mesh(test, v, f)
    # Recovered surface should still be a sphere of radius ~0.5.
    radius = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(np.median(radius), 0.5, atol=3.0 / 64.0)


def test_sparse_mc_watertight(test, device):
    """Check that a closed SDF strictly inside the domain yields a watertight, manifold mesh.

    A boundary edge (used by a single triangle) is exactly a hole; it would mean
    the octree dropped a surface-crossing cell or the shared-edge vertex
    de-duplication cracked. We also check the Euler characteristic to confirm the
    genus is recovered (sphere -> 2, torus -> 0).
    """
    cases = [
        (sphere_sdf, 2),  # genus 0
        (torus_sdf, 0),  # genus 1
    ]
    for sdf, euler in cases:
        with test.subTest(sdf=sdf.key):
            verts, indices = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
                sdf, (-1.0, -1.0, -1.0), 2.0, max_depth=6, device=device
            )
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
    v_ref, f_ref = wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
        sphere_sdf, origin, root_width, max_depth=depth, device=device
    )
    v_ref = v_ref.numpy()
    f_ref = f_ref.numpy().reshape(-1, 3)

    # Recover the octree cells, then sample the field at their corners ourselves.
    cell_origins, cell_width = wp.geometry.lipschitz_octree(
        sphere_sdf, origin, root_width, max_depth=depth, device=device
    )
    co = cell_origins.numpy()
    cells = np.round((co - np.array(origin)) / cell_width).astype(np.int32)
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

    cell_origins, cell_width = wp.geometry.lipschitz_octree(sphere_sdf, origin, 2.0, max_depth=depth, device=device)
    cells = np.round((cell_origins.numpy() - np.array(origin)) / cell_width).astype(np.int32)
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

    cell_origins, cell_width = wp.geometry.lipschitz_octree(sphere_sdf, origin, root_width, max_depth, device=device)
    origins = cell_origins.numpy()
    test.assertGreater(origins.shape[0], 0)
    np.testing.assert_allclose(cell_width, root_width / resolution)

    # Every kept cell's center is within the Lipschitz band of the surface.
    centers = origins + 0.5 * cell_width
    band = np.sqrt(3.0) / 2.0 * cell_width
    center_sdf = np.linalg.norm(centers, axis=1) - 0.5
    test.assertLessEqual(np.abs(center_sdf).max(), band + 1e-5)

    # Recover integer subscripts and confirm completeness against a dense grid.
    kept = {tuple(np.round((o - np.array(origin)) / cell_width).astype(int)) for o in origins}

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
        wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
            sphere_sdf, (0.0, 0.0, 0.0), 2.0, max_depth=-1, device=device
        )
    with test.assertRaises(ValueError):
        wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
            sphere_sdf, (0.0, 0.0, 0.0), 0.0, max_depth=4, device=device
        )
    with test.assertRaises(ValueError):
        wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(
            sphere_sdf, (0.0, 0.0, 0.0), 2.0, max_depth=4, lipschitz_bound=-1.0, device=device
        )
    with test.assertRaises(ValueError):
        wp.geometry.lipschitz_octree(sphere_sdf, (0.0, 0.0, 0.0), 2.0, max_depth=4, lipschitz_bound=-1.0, device=device)
    with test.assertRaises(TypeError):
        wp.geometry.sparse_marching_cubes_via_lipschitz_pruning(42, (0.0, 0.0, 0.0), 2.0, max_depth=4, device=device)

    # A non-positive cell width collapses every corner onto the origin (or
    # mirrors the cell), so the explicit-cells entry point rejects it.
    cells = np.zeros((1, 3), dtype=np.int32)
    corner_values = np.zeros((1, 8), dtype=np.float32)
    for bad_width in (0.0, -1.0):
        with test.assertRaises(ValueError):
            wp.geometry.sparse_marching_cubes_from_cells(cells, corner_values, cell_width=bad_width, device=device)


devices = get_test_devices()


class TestSparseMarchingCubes(unittest.TestCase):
    pass


add_function_test(TestSparseMarchingCubes, "test_sparse_mc_sphere", test_sparse_mc_sphere, devices=devices)
add_function_test(
    TestSparseMarchingCubes, "test_sparse_mc_matches_dense", test_sparse_mc_matches_dense, devices=devices
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
    TestSparseMarchingCubes, "test_sparse_mc_python_callable", test_sparse_mc_python_callable, devices=devices
)
add_function_test(TestSparseMarchingCubes, "test_sparse_mc_mesh_sdf", test_sparse_mc_mesh_sdf, devices=devices)
add_function_test(TestSparseMarchingCubes, "test_sparse_mc_watertight", test_sparse_mc_watertight, devices=devices)
add_function_test(TestSparseMarchingCubes, "test_sparse_mc_from_cells", test_sparse_mc_from_cells, devices=devices)
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
