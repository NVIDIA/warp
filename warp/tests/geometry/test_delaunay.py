# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp._src.geometry as _src_geometry
import warp.geometry
from warp.tests.unittest_utils import *

# ---------------------------------------------------------------------------
# NumPy reference helpers (independent of the Warp implementation)
# ---------------------------------------------------------------------------


def _signed_area(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_circle_det(a, b, c, d):
    """Return the in-circle determinant for counterclockwise triangle ``abc`` and point ``d``."""
    ad = a - d
    bd = b - d
    cd = c - d
    ad2 = ad @ ad
    bd2 = bd @ bd
    cd2 = cd @ cd
    return (
        ad[0] * (bd[1] * cd2 - cd[1] * bd2)
        - ad[1] * (bd[0] * cd2 - cd[0] * bd2)
        + ad2 * (bd[0] * cd[1] - bd[1] * cd[0])
    )


def _edge_map(tris):
    """Map each undirected edge to the list of ``(tri_index, apex_local)`` incident to it."""
    edges = {}
    for ti, tri in enumerate(tris):
        for j in range(3):
            a = int(tri[(j + 1) % 3])
            b = int(tri[(j + 2) % 3])
            edges.setdefault((min(a, b), max(a, b)), []).append((ti, j))
    return edges


def _assert_valid_mesh(test, points, tris):
    # Every triangle counterclockwise (no inversions/degeneracies).
    for tri in tris:
        area = _signed_area(points[tri[0]], points[tri[1]], points[tri[2]])
        test.assertGreater(area, 0.0, f"non-positive triangle area {area} for {tri}")

    # Every edge shared by at most two triangles (manifold).
    for edge, incident in _edge_map(tris).items():
        test.assertLessEqual(len(incident), 2, f"non-manifold edge {edge}")


def _assert_delaunay(test, points, tris, tol=1e-9):
    for edge, incident in _edge_map(tris).items():
        if len(incident) != 2:
            continue
        (t0, _apex0), (t1, apex1) = incident
        tri0 = tris[t0]
        d = points[tris[t1][apex1]]
        det = _in_circle_det(points[tri0[0]], points[tri0[1]], points[tri0[2]], d)
        test.assertLessEqual(det, tol, f"edge {edge} violates Delaunay condition (det={det})")


def _grid_mesh(nx, ny, jitter=0.0, seed=0):
    """Build a jittered grid triangulation using the (bottom-left, top-right) diagonal."""
    xs, ys = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    points = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        interior = (xs.ravel() > 0) & (xs.ravel() < nx) & (ys.ravel() > 0) & (ys.ravel() < ny)
        points[interior] += rng.uniform(-jitter, jitter, size=(int(interior.sum()), 2)).astype(np.float32)

    def vid(i, j):
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            bl, br, tr, tl = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris.append([bl, br, tr])
            tris.append([bl, tr, tl])
    return points, np.array(tris, dtype=np.int32)


def _convex_fan_mesh(n, a=2.0, b=1.0, seed=0):
    """Points on an ellipse (convex, not cocircular) triangulated as a fan from vertex 0.

    The ellipse guarantees a convex polygon so the fan is a valid triangulation,
    while ``a != b`` keeps the points off a common circle so the Delaunay
    triangulation is non-degenerate and the fan is far from it.
    """
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n))
    points = np.stack([a * np.cos(angles), b * np.sin(angles)], axis=1).astype(np.float32)
    tris = np.array([[0, i, i + 1] for i in range(1, n - 1)], dtype=np.int32)
    return points, tris


def _star_mesh(num_points=6):
    """A non-convex star-shaped polygon triangulated as a fan from an added center vertex.

    The polygon is star-shaped about the origin, so the center fan is a valid
    triangulation of a non-convex simple polygon. Its boundary edges are shared
    by a single triangle each (never flipped), while the interior spokes are.
    """
    outer_r, inner_r = 1.0, 0.45
    boundary = []
    for i in range(2 * num_points):
        r = outer_r if i % 2 == 0 else inner_r
        ang = np.pi * float(i) / float(num_points)
        boundary.append([r * np.cos(ang), r * np.sin(ang)])
    boundary = np.array(boundary, dtype=np.float32)  # counterclockwise
    points = np.concatenate([np.zeros((1, 2), dtype=np.float32), boundary], axis=0)  # index 0 = center
    m = boundary.shape[0]
    tris = np.array([[0, 1 + i, 1 + (i + 1) % m] for i in range(m)], dtype=np.int32)
    return points, tris


def _flip(*args, **kwargs):
    """Call ``delaunay_edge_flip`` and convert its device accumulator to a Python int.

    ``delaunay_edge_flip`` always returns a single-element ``warp.array`` (see its
    docstring), so eager-mode tests read it back here rather than at every call site.
    """
    return int(warp.geometry.delaunay_edge_flip(*args, **kwargs).numpy()[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@wp.kernel
def _eval_predicates(out_area: wp.array[float], out_in: wp.array[wp.int32]):
    a = wp.vec2(0.0, 0.0)
    b = wp.vec2(1.0, 0.0)
    c = wp.vec2(0.0, 1.0)
    out_area[0] = _src_geometry.signed_area(a, b, c)  # +0.5 (counterclockwise)
    out_area[1] = _src_geometry.signed_area(a, c, b)  # -0.5 (clockwise)
    # Circumcircle of the unit right triangle is centered at (0.5, 0.5), radius ~0.707.
    out_in[0] = wp.int32(_src_geometry.in_circle(a, b, c, wp.vec2(0.4, 0.4)))  # inside -> 1
    out_in[1] = wp.int32(_src_geometry.in_circle(a, b, c, wp.vec2(2.0, 2.0)))  # outside -> 0


def test_predicates(test, device):
    """Verify the signed-area and in-circle predicates on a unit right triangle."""
    out_area = wp.empty(2, dtype=float, device=device)
    out_in = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(_eval_predicates, dim=1, inputs=[out_area, out_in], device=device)
    area = out_area.numpy()
    inside = out_in.numpy()
    np.testing.assert_allclose(area[0], 0.5, atol=1e-6)
    np.testing.assert_allclose(area[1], -0.5, atol=1e-6)
    test.assertEqual(inside[0], 1)
    test.assertEqual(inside[1], 0)


def test_adjacency_single_pair(test, device):
    """Verify triangle-triangle adjacency on the smallest interior-edge mesh.

    Two triangles sharing edge (0, 2): tri 0 = (0,1,2), tri 1 = (0,2,3).
    """
    indices = wp.array(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), dtype=wp.int32, device=device)
    triangle_neighbors, neighbor_edge_indices = warp.geometry.tri_tri_adjacency(indices, vertex_count=4)
    triangle_neighbors_np = triangle_neighbors.numpy()
    neighbor_edge_indices_np = neighbor_edge_indices.numpy()

    # Shared edge (0,2) is opposite vertex 1 in tri 0 (local edge 1) and opposite
    # vertex 3 in tri 1 (local edge 2).
    test.assertEqual(triangle_neighbors_np[0, 1], 1)
    test.assertEqual(neighbor_edge_indices_np[0, 1], 2)
    test.assertEqual(triangle_neighbors_np[1, 2], 0)
    test.assertEqual(neighbor_edge_indices_np[1, 2], 1)

    # All other edges are on the boundary.
    test.assertEqual(triangle_neighbors_np[0, 0], -1)
    test.assertEqual(triangle_neighbors_np[0, 2], -1)
    test.assertEqual(triangle_neighbors_np[1, 0], -1)
    test.assertEqual(triangle_neighbors_np[1, 1], -1)

    # return_neighbor_edge_indices=False yields the same triangle_neighbors as a single array (no neighbor_edge_indices).
    triangle_neighbors_only = warp.geometry.tri_tri_adjacency(
        indices, vertex_count=4, return_neighbor_edge_indices=False
    )
    test.assertFalse(isinstance(triangle_neighbors_only, tuple))
    assert_np_equal(triangle_neighbors_only.numpy(), triangle_neighbors_np)


def test_adjacency_matches_grid(test, device):
    """Verify that adjacency pointers round-trip on a larger grid mesh."""
    points, tris = _grid_mesh(5, 4, jitter=0.2, seed=11)
    indices = wp.array(tris, dtype=wp.int32, device=device)
    triangle_neighbors, neighbor_edge_indices = warp.geometry.tri_tri_adjacency(indices, vertex_count=points.shape[0])
    triangle_neighbors_only = warp.geometry.tri_tri_adjacency(
        indices, vertex_count=points.shape[0], return_neighbor_edge_indices=False
    )
    triangle_neighbors_np = triangle_neighbors.numpy()
    neighbor_edge_indices_np = neighbor_edge_indices.numpy()
    # return_neighbor_edge_indices=False must agree with the full build.
    assert_np_equal(triangle_neighbors_only.numpy(), triangle_neighbors_np)

    # For every interior edge, the neighbor edge index round-trips.
    for t in range(tris.shape[0]):
        for j in range(3):
            n = triangle_neighbors_np[t, j]
            if n < 0:
                continue
            jn = neighbor_edge_indices_np[t, j]
            test.assertEqual(triangle_neighbors_np[n, jn], t)
            test.assertEqual(neighbor_edge_indices_np[n, jn], j)


def test_adjacency_infers_vertex_count(test, device):
    """Verify that omitting vertex_count (device-side max-vertex-index reduction) matches passing it explicitly."""
    points, tris = _grid_mesh(5, 4, jitter=0.2, seed=11)
    indices = wp.array(tris, dtype=wp.int32, device=device)
    triangle_neighbors_inferred, neighbor_edge_indices_inferred = warp.geometry.tri_tri_adjacency(indices)
    triangle_neighbors_explicit, neighbor_edge_indices_explicit = warp.geometry.tri_tri_adjacency(
        indices, vertex_count=points.shape[0]
    )
    assert_np_equal(triangle_neighbors_inferred.numpy(), triangle_neighbors_explicit.numpy())
    assert_np_equal(neighbor_edge_indices_inferred.numpy(), neighbor_edge_indices_explicit.numpy())


def test_adjacency_requires_vertex_count_under_capture(test, device):
    """Reject inferring vertex_count (a host sync) during graph capture."""
    indices = wp.array(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), dtype=wp.int32, device=device)

    wp.load_module(warp.geometry, device=device)
    with wp.ScopedDevice(device):
        with test.assertRaises(RuntimeError):
            with wp.ScopedCapture(force_module_load=False):
                warp.geometry.tri_tri_adjacency(indices)


def test_flip_single_edge(test, device):
    """Flip the one non-Delaunay edge of a thin quad."""
    points = np.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    # tri 0 = (0,1,2) above edge, tri 1 = (1,0,3) below edge; both counterclockwise.
    tris = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)

    _assert_valid_mesh(test, points, tris)  # sanity on the input
    test.assertFalse(_is_delaunay(points, tris))

    positions = wp.array(points, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)

    num_flips = _flip(positions, indices)
    test.assertEqual(num_flips, 1)

    out = indices.numpy()
    _assert_valid_mesh(test, points, out)
    _assert_delaunay(test, points, out)

    # The shared edge is now the vertical diagonal (2, 3).
    edges = _edge_map(out)
    test.assertIn((2, 3), edges)
    test.assertEqual(len(edges[(2, 3)]), 2)


def _is_delaunay(points, tris, tol=1e-9):
    for _edge, incident in _edge_map(tris).items():
        if len(incident) != 2:
            continue
        (t0, _j0), (t1, apex1) = incident
        tri0 = tris[t0]
        d = points[tris[t1][apex1]]
        det = _in_circle_det(points[tri0[0]], points[tri0[1]], points[tri0[2]], d)
        if det > tol:
            return False
    return True


def test_flip_grid(test, device):
    """Flip a jittered grid to a Delaunay triangulation and confirm it is a fixed point."""
    points, tris = _grid_mesh(6, 5, jitter=0.3, seed=1234)
    _assert_valid_mesh(test, points, tris)
    test.assertFalse(_is_delaunay(points, tris), "input grid should not already be Delaunay")

    positions = wp.array(points, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)

    total_area_before = sum(_signed_area(points[t[0]], points[t[1]], points[t[2]]) for t in tris)
    verts_before = set(np.unique(tris).tolist())

    num_flips = _flip(positions, indices)
    test.assertGreater(num_flips, 0)

    out = indices.numpy()
    _assert_valid_mesh(test, points, out)
    _assert_delaunay(test, points, out)

    # Connectivity stays a valid triangulation of the same vertices with conserved area.
    test.assertEqual(out.shape, tris.shape)
    test.assertEqual(set(np.unique(out).tolist()), verts_before)
    total_area_after = sum(_signed_area(points[t[0]], points[t[1]], points[t[2]]) for t in out)
    np.testing.assert_allclose(total_area_after, total_area_before, rtol=1e-5)

    # A Delaunay triangulation is a fixed point: flipping again changes nothing.
    test.assertEqual(_flip(positions, indices), 0)


def test_flip_grid_large(test, device):
    """Stress the parallel independent set on a large perturbed grid.

    Many flips are applied per pass, so this is the key check that concurrent
    flips never corrupt adjacency (race-freedom) and still converge to a valid
    Delaunay mesh.
    """
    points, tris = _grid_mesh(40, 40, jitter=0.25, seed=99)
    _assert_valid_mesh(test, points, tris)
    test.assertFalse(_is_delaunay(points, tris))

    positions = wp.array(points, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)
    area_before = sum(_signed_area(points[t[0]], points[t[1]], points[t[2]]) for t in tris)

    num_flips = _flip(positions, indices)
    test.assertGreater(num_flips, 0)

    out = indices.numpy()
    _assert_valid_mesh(test, points, out)
    _assert_delaunay(test, points, out)
    test.assertEqual(set(np.unique(out).tolist()), set(np.unique(tris).tolist()))
    area_after = sum(_signed_area(points[t[0]], points[t[1]], points[t[2]]) for t in out)
    np.testing.assert_allclose(area_after, area_before, rtol=1e-4)
    test.assertEqual(_flip(positions, indices), 0)


def test_flip_grid_sheared_converges_in_few_passes(test, device):
    """Converge in O(1) passes on a sheared grid, not O(num_rows).

    A regularly-numbered grid (triangle index increasing row by row) sheared
    so that every internal diagonal is non-Delaunay is the textbook adversarial
    case for a maximal-independent-set edge flipper: a naive priority that is
    monotonic in triangle index (e.g. plain ``t * 3 + j``) makes claims cascade
    along each column and collapses the whole independent set to one row
    resolved per pass, taking ``num_rows - 1`` passes to converge instead of
    the expected ``O(log num_rows)``. This asserts a small constant bound
    instead of pinning an exact pass count, so it stays robust to unrelated
    convergence-order changes while still catching a regression back to
    linear-in-mesh-size behavior.
    """
    nx = ny = 40
    points, tris = _grid_mesh(nx, ny, jitter=0.0)
    # The sign of the shear that breaks Delaunay-ness depends on which way
    # _grid_mesh's fixed (bottom-left, top-right) diagonal runs; +0.1 does it here.
    points = points @ np.array([[1.0, 0.1], [0.0, 1.0]], dtype=np.float32)
    test.assertFalse(_is_delaunay(points, tris))

    positions = wp.array(points, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)

    max_passes = 5
    num_flips = warp.geometry.delaunay_edge_flip(positions, indices, max_passes=max_passes)
    num_flips = int(num_flips.numpy()[0])
    test.assertGreater(num_flips, 0)

    out = indices.numpy()
    _assert_valid_mesh(test, points, out)
    # If this starts failing, check whether the claim priority computed in
    # _EdgeFlipper._edge_priority() regressed back to being monotonic in
    # triangle index (e.g. plain t * 3 + j).
    _assert_delaunay(test, points, out)


def test_flip_already_delaunay(test, device):
    """Leave an already-Delaunay (right-triangulated axis-aligned grid) mesh untouched."""
    points, tris = _grid_mesh(4, 4, jitter=0.0)
    positions = wp.array(points, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)

    num_flips = _flip(positions, indices)
    test.assertEqual(num_flips, 0)
    assert_np_equal(indices.numpy(), tris)


def test_flip_reference_rejection(test, device):
    """Reject a flip that would degenerate the reference configuration.

    Same thin quad as the single-edge test, but the reference config is
    degenerate (all four points collinear), so the otherwise-valid flip must not
    be applied.
    """
    points = np.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    tris = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
    ref = np.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    positions = wp.array(points, dtype=wp.vec2, device=device)
    ref_positions = wp.array(ref, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)

    num_flips = _flip(positions, indices, ref_positions=ref_positions)
    test.assertEqual(num_flips, 0)
    assert_np_equal(indices.numpy(), tris)


def test_flip_empty(test, device):
    """Report zero flips for a mesh with no triangles."""
    positions = wp.zeros(0, dtype=wp.vec2, device=device)
    indices = wp.zeros((0, 3), dtype=wp.int32, device=device)
    num_flips = _flip(positions, indices)
    test.assertEqual(num_flips, 0)


def test_flip_invalid_arguments(test, device):
    """Reject malformed arguments before any flip pass runs.

    ``max_passes`` in particular must be validated up front: the convergence loop
    is seeded to run once, so a non-positive budget would otherwise still modify
    the connectivity.
    """
    points, tris = _grid_mesh(3, 3, jitter=0.3, seed=5)
    positions = wp.array(points, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)

    with test.assertRaises(ValueError):
        warp.geometry.delaunay_edge_flip(positions, indices, max_passes=0)
    with test.assertRaises(ValueError):
        warp.geometry.delaunay_edge_flip(positions, indices, max_passes=-1)

    # Negative or non-finite epsilons would silently loosen (or NaN-disable) the
    # guards they are meant to enforce, so they are rejected rather than accepted.
    for bad_eps in (-1.0, float("nan"), float("inf")):
        with test.assertRaises(ValueError):
            warp.geometry.delaunay_edge_flip(positions, indices, area_epsilon=bad_eps)
        with test.assertRaises(ValueError):
            warp.geometry.delaunay_edge_flip(positions, indices, ref_area_epsilon=bad_eps)

    # A rejected call must leave the connectivity untouched.
    assert_np_equal(indices.numpy(), tris)

    flat = wp.zeros(4, dtype=wp.int32, device=device)
    with test.assertRaises(ValueError):
        warp.geometry.delaunay_edge_flip(positions, flat)
    with test.assertRaises(ValueError):
        warp.geometry.tri_tri_adjacency(flat)


def test_flip_convex_fan(test, device):
    """Flip a fan triangulation of a convex polygon to the full Delaunay triangulation.

    The domain is convex and unconstrained, so every non-Delaunay edge is
    flippable and the result must satisfy the empty-circumcircle property.
    """
    points, tris = _convex_fan_mesh(40, seed=3)
    _assert_valid_mesh(test, points, tris)
    test.assertFalse(_is_delaunay(points, tris))

    positions = wp.array(points, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)
    area_before = sum(_signed_area(points[t[0]], points[t[1]], points[t[2]]) for t in tris)

    num_flips = _flip(positions, indices)
    test.assertGreater(num_flips, 0)

    out = indices.numpy()
    _assert_valid_mesh(test, points, out)
    _assert_delaunay(test, points, out)
    test.assertEqual(set(np.unique(out).tolist()), set(np.unique(tris).tolist()))
    area_after = sum(_signed_area(points[t[0]], points[t[1]], points[t[2]]) for t in out)
    np.testing.assert_allclose(area_after, area_before, rtol=1e-4)
    test.assertEqual(_flip(positions, indices), 0)


def test_flip_star_polygon(test, device):
    """Preserve a non-convex domain while flipping to a fixed point.

    Some interior edges of a star-shaped polygon are un-flippable because their
    quad is non-convex, so this verifies the flipper reaches a valid fixed point
    with the boundary intact rather than asserting full Delaunay.
    """
    points, tris = _star_mesh(num_points=7)
    _assert_valid_mesh(test, points, tris)

    positions = wp.array(points, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)
    area_before = sum(_signed_area(points[t[0]], points[t[1]], points[t[2]]) for t in tris)

    num_flips = _flip(positions, indices)
    test.assertGreater(num_flips, 0)

    out = indices.numpy()
    _assert_valid_mesh(test, points, out)
    test.assertEqual(out.shape, tris.shape)
    test.assertEqual(set(np.unique(out).tolist()), set(np.unique(tris).tolist()))
    area_after = sum(_signed_area(points[t[0]], points[t[1]], points[t[2]]) for t in out)
    np.testing.assert_allclose(area_after, area_before, rtol=1e-4)
    # Boundary is preserved: same set of boundary (singly-incident) edges.
    before_boundary = {e for e, inc in _edge_map(tris).items() if len(inc) == 1}
    after_boundary = {e for e, inc in _edge_map(out).items() if len(inc) == 1}
    test.assertEqual(before_boundary, after_boundary)
    # Converged to a fixed point.
    test.assertEqual(_flip(positions, indices), 0)


def test_flip_graph_capture(test, device):
    """Replay a captured flip loop and match the eager result.

    The convergence loop is driven by ``wp.capture_while()``, which records CUDA
    conditional graph nodes, so CUDA devices need Toolkit and driver 12.4+.
    """
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("CUDA conditional graph nodes require Toolkit and driver 12.4+")

    points, tris = _grid_mesh(8, 7, jitter=0.3, seed=7)

    # Eager reference result on a separate copy of the input.
    ref_positions = wp.array(points, dtype=wp.vec2, device=device)
    ref_indices = wp.array(tris, dtype=wp.int32, device=device)
    ref_flips = _flip(ref_positions, ref_indices)
    test.assertGreater(ref_flips, 0)
    expected = ref_indices.numpy()

    # Warm up device allocations (radix-sort scratch, etc.) before capturing so
    # that no first-time allocation happens inside the captured region.
    warp.geometry.delaunay_edge_flip(
        wp.array(points, dtype=wp.vec2, device=device),
        wp.array(tris, dtype=wp.int32, device=device),
    )

    positions = wp.array(points, dtype=wp.vec2, device=device)
    indices = wp.array(tris, dtype=wp.int32, device=device)

    empty_positions = wp.zeros(0, dtype=wp.vec2, device=device)
    empty_indices = wp.zeros((0, 3), dtype=wp.int32, device=device)

    # Load the flip kernels up front so no compilation is triggered inside the
    # capture, which would recursively build modules in single-process test mode.
    wp.load_module(warp.geometry, device=device)
    with wp.ScopedDevice(device):
        with wp.ScopedCapture(force_module_load=False) as capture:
            total = warp.geometry.delaunay_edge_flip(positions, indices)
            empty_total = warp.geometry.delaunay_edge_flip(empty_positions, empty_indices)

    # delaunay_edge_flip always returns a device accumulator, empty mesh or not,
    # capturing or not; under capture its contents are only valid at replay time,
    # so both are read below with the rest of the results.
    test.assertIsInstance(total, wp.array)
    test.assertIsInstance(empty_total, wp.array)

    # Capture records operations without executing them: the mesh is untouched.
    assert_np_equal(indices.numpy(), tris)

    wp.capture_launch(capture.graph)

    out = indices.numpy()
    _assert_delaunay(test, points, out)
    assert_np_equal(out, expected)
    test.assertEqual(int(total.numpy()[0]), ref_flips)
    test.assertEqual(int(empty_total.numpy()[0]), 0)

    # The captured graph rebuilds adjacency from the current connectivity each
    # replay, so replaying on the now-Delaunay mesh is a stable no-op.
    wp.capture_launch(capture.graph)
    assert_np_equal(indices.numpy(), expected)
    test.assertEqual(int(total.numpy()[0]), 0)


devices = get_test_devices()


class TestDelaunay(unittest.TestCase):
    pass


add_function_test(TestDelaunay, "test_predicates", test_predicates, devices=devices)
add_function_test(TestDelaunay, "test_adjacency_single_pair", test_adjacency_single_pair, devices=devices)
add_function_test(TestDelaunay, "test_adjacency_matches_grid", test_adjacency_matches_grid, devices=devices)
add_function_test(
    TestDelaunay, "test_adjacency_infers_vertex_count", test_adjacency_infers_vertex_count, devices=devices
)
add_function_test(
    TestDelaunay,
    "test_adjacency_requires_vertex_count_under_capture",
    test_adjacency_requires_vertex_count_under_capture,
    devices=devices,
)
add_function_test(TestDelaunay, "test_flip_single_edge", test_flip_single_edge, devices=devices)
add_function_test(TestDelaunay, "test_flip_grid", test_flip_grid, devices=devices)
add_function_test(TestDelaunay, "test_flip_grid_large", test_flip_grid_large, devices=devices)
add_function_test(
    TestDelaunay,
    "test_flip_grid_sheared_converges_in_few_passes",
    test_flip_grid_sheared_converges_in_few_passes,
    devices=devices,
)
add_function_test(TestDelaunay, "test_flip_already_delaunay", test_flip_already_delaunay, devices=devices)
add_function_test(TestDelaunay, "test_flip_reference_rejection", test_flip_reference_rejection, devices=devices)
add_function_test(TestDelaunay, "test_flip_convex_fan", test_flip_convex_fan, devices=devices)
add_function_test(TestDelaunay, "test_flip_star_polygon", test_flip_star_polygon, devices=devices)
add_function_test(TestDelaunay, "test_flip_empty", test_flip_empty, devices=devices)
add_function_test(TestDelaunay, "test_flip_invalid_arguments", test_flip_invalid_arguments, devices=devices)
add_function_test(TestDelaunay, "test_flip_graph_capture", test_flip_graph_capture, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
