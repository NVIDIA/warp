# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``warp.geometry.self_intersect_tri_tri``.

Several adjacency fixtures are inspired by PaMO's Appendix 13 and released CUDA code.
Reference: https://arxiv.org/html/2509.05595v1#S13
Code: https://github.com/SarahWeiii/pamo/tree/a10e34351eb7de71f41eb279e7ab9b2b101cf7a4/simp_cuda/src/bvh
In particular: coplanar angle/collinear-ray cases in tri_tri_2d.cuh, and
coordinate-based adjacency and short-intersection filtering in self_intersect.cuh.
Fixtures are independently constructed; PaMO is not a correctness oracle or a
runtime dependency. Our adjacency comes exclusively from IDs, with no length cutoff.
"""

import unittest
from fractions import Fraction
from itertools import combinations, permutations
from typing import Any

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def self_intersect_kernel(points: wp.array2d[Any], ids: wp.array2d[wp.vec3i], tol: wp.array[Any], hits: wp.array[bool]):
    i = wp.tid()
    hits[i] = wp.geometry.self_intersect_tri_tri(
        points[i, 0],
        points[i, 1],
        points[i, 2],
        points[i, 3],
        points[i, 4],
        points[i, 5],
        ids[i, 0],
        ids[i, 1],
        tol[i],
    )


@wp.kernel(enable_backward=False)
def compare_unrelated_kernel(points: wp.array2d[Any], tol: wp.array[Any], hits: wp.array2d[bool]):
    i = wp.tid()
    a, b = wp.vec3i(0, 1, 2), wp.vec3i(3, 4, 5)
    a0, a1, a2 = points[i, 0], points[i, 1], points[i, 2]
    b0, b1, b2 = points[i, 3], points[i, 4], points[i, 5]
    hits[i, 0] = wp.geometry.self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, a, b)
    hits[i, 1] = wp.geometry.intersect_tri_tri(a0, a1, a2, b0, b1, b2)
    hits[i, 2] = wp.geometry.self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, a, b, tol[i])
    hits[i, 3] = wp.geometry.intersect_tri_tri(a0, a1, a2, b0, b1, b2, tol[i])
    hits[i, 4] = wp.geometry.self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, a, b, tol=tol[i])
    hits[i, 5] = wp.geometry.self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, a, b, tol=wp.float64(1e-6))
    hits[i, 6] = wp.geometry.self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, a, b, tol=1e-6)
    hits[i, 7] = wp.geometry.intersect_tri_tri(a0, a1, a2, b0, b1, b2, tol=1e-6)


@wp.kernel(enable_backward=False)
def tolerance_forms_kernel(points: wp.array2d[Any], ids: wp.array2d[wp.vec3i], hits: wp.array2d[bool]):
    i = wp.tid()
    a0, a1, a2 = points[i, 0], points[i, 1], points[i, 2]
    b0, b1, b2 = points[i, 3], points[i, 4], points[i, 5]
    a, b = ids[i, 0], ids[i, 1]
    hits[i, 0] = wp.geometry.self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, a, b)
    hits[i, 1] = wp.geometry.self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, a, b, wp.float64(1e-6))
    hits[i, 2] = wp.geometry.self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, a, b, tol=1e-6)
    hits[i, 3] = wp.geometry.self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, a, b, tol=0.0)


wp.overload(self_intersect_kernel, [wp.array2d[wp.vec3], wp.array2d[wp.vec3i], wp.array[wp.float32], wp.array[bool]])
wp.overload(self_intersect_kernel, [wp.array2d[wp.vec3], wp.array2d[wp.vec3i], wp.array[wp.float64], wp.array[bool]])
wp.overload(self_intersect_kernel, [wp.array2d[wp.vec3d], wp.array2d[wp.vec3i], wp.array[wp.float32], wp.array[bool]])
wp.overload(self_intersect_kernel, [wp.array2d[wp.vec3d], wp.array2d[wp.vec3i], wp.array[wp.float64], wp.array[bool]])
wp.overload(compare_unrelated_kernel, [wp.array2d[wp.vec3], wp.array[wp.float32], wp.array2d[bool]])
wp.overload(compare_unrelated_kernel, [wp.array2d[wp.vec3], wp.array[wp.float64], wp.array2d[bool]])
wp.overload(compare_unrelated_kernel, [wp.array2d[wp.vec3d], wp.array[wp.float32], wp.array2d[bool]])
wp.overload(compare_unrelated_kernel, [wp.array2d[wp.vec3d], wp.array[wp.float64], wp.array2d[bool]])
wp.overload(tolerance_forms_kernel, [wp.array2d[wp.vec3], wp.array2d[wp.vec3i], wp.array2d[bool]])
wp.overload(tolerance_forms_kernel, [wp.array2d[wp.vec3d], wp.array2d[wp.vec3i], wp.array2d[bool]])


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def lerp(a, b, t):
    return tuple(x + t * (y - x) for x, y in zip(a, b, strict=True))


def exact_self_intersection(points, ids):
    """Clip triangle A against B's edge halfspaces, then intersect with B's plane.

    Rational arithmetic on the represented input coordinates gives the entire
    intersection's extreme points, independently of the opposite-edge algorithm.
    Check those points against the topologically shared simplex.
    """
    a, b = [[tuple(Fraction(float(x)) for x in p) for p in tri] for tri in np.reshape(points, (2, 3, 3))]
    shared = set(ids[0]).intersection(ids[1])
    if len(shared) == 3:
        return True
    normal = cross(sub(b[1], b[0]), sub(b[2], b[0]))
    polygon = a
    for j in range(3):
        inward = cross(normal, sub(b[(j + 1) % 3], b[j]))
        if not polygon:
            return False
        output = []
        previous = polygon[-1]
        dp = dot(inward, sub(previous, b[j]))
        for current in polygon:
            dc = dot(inward, sub(current, b[j]))
            if (dp < 0) != (dc < 0):
                output.append(lerp(previous, current, dp / (dp - dc)))
            if dc >= 0:
                output.append(current)
            previous, dp = current, dc
        polygon = output
    if not polygon:
        return False
    section = []
    previous = polygon[-1]
    dp = dot(normal, sub(previous, b[0]))
    for current in polygon:
        dc = dot(normal, sub(current, b[0]))
        if dc == 0:
            section.append(current)
        elif (dp < 0 < dc) or (dc < 0 < dp):
            section.append(lerp(previous, current, dp / (dp - dc)))
        previous, dp = current, dc
    if not shared:
        return bool(section)
    common = [a[j] for j in range(3) if ids[0][j] in shared]
    if len(common) == 1:
        return any(p != common[0] for p in section)
    edge = sub(common[1], common[0])
    for p in section:
        offset = sub(p, common[0])
        if cross(edge, offset) != (0, 0, 0) or not 0 <= dot(edge, offset) <= dot(edge, edge):
            return True
    return False


def run_cases(test, device, points, ids, expected, tolerance=0.0):
    for dtype in (wp.vec3, wp.vec3d):
        data = wp.array(points, dtype=dtype, device=device)
        topology = wp.array(ids, dtype=wp.vec3i, device=device)
        tolerances = wp.full(len(points), tolerance, dtype=wp.float64, device=device)
        hits = wp.empty(len(points), dtype=bool, device=device)
        wp.launch(self_intersect_kernel, dim=len(points), inputs=[data, topology, tolerances, hits], device=device)
        with test.subTest(dtype=dtype):
            np.testing.assert_array_equal(hits.numpy(), expected)


def fixture_cases():
    a = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    cases = [
        # PaMO coplanar vertex-sharing cases: disjoint, crossing sectors,
        # containment, same/opposite collinear rays, and complementary sectors.
        ([(0, 0, 0), (-1, 0, 0), (0, -1, 0)], [0, 3, 4], False),
        ([(0, 0, 0), (1, -1, 0), (1, 1, 0)], [0, 3, 4], True),
        ([(0, 0, 0), (0.25, 0.125, 0), (0.125, 0.25, 0)], [0, 3, 4], True),
        ([(0, 0, 0), (2, 0, 0), (0, -1, 0)], [0, 3, 4], True),
        ([(0, 0, 0), (-1, 0, 0), (0, 1, 0)], [0, 3, 4], True),
        ([(0, 0, 0), (-1, 1, 0), (-1, -1, 0)], [0, 3, 4], False),
        # A supporting line passes through A's sector, but B occupies its
        # opposite rays: cross-product sign checks alone must not accept this.
        ([(0, 0, 0), (-1, 1, 0), (-1, -2, 0)], [0, 3, 4], False),
        # Noncoplanar one-vertex contact and intersection beyond that vertex.
        ([(0, 0, 0), (1, 0, 1), (0, 1, 1)], [0, 3, 4], False),
        ([(0, 0, 0), (0.5, 0.5, -1), (0.5, 0.5, 1)], [0, 3, 4], True),
        ([(0, 0, 0), (-0.5, -0.5, -1), (-0.5, -0.5, 1)], [0, 3, 4], False),
        # Edge-sharing same/opposite side, and genuine 3D folds.
        ([(0, 0, 0), (1, 0, 0), (0, -1, 0)], [0, 1, 3], False),
        ([(0, 0, 0), (1, 0, 0), (0.25, 0.5, 0)], [0, 1, 3], True),
        ([(0, 0, 0), (1, 0, 0), (0, 1, 1)], [0, 1, 3], False),
        ([(0, 0, 0), (1, 0, 0), (0, -1, 1)], [0, 1, 3], False),
        # Identical coordinates do not establish adjacency; equal-ID duplicates do.
        ([(0, 0, 0), (1, 0, 0), (0, -1, 1)], [3, 4, 5], True),
        ([(0, 0, 0), (-1, 0, 0), (0, -1, 0)], [3, 4, 5], True),
        (a, [3, 4, 5], True),
        (a, [0, 1, 2], True),
    ]
    return [(np.array(a + b, dtype=np.float64), np.array([[0, 1, 2], bid]), hit) for b, bid, hit in cases]


def test_adjacency_fixtures(test, device):
    """Classify adjacency fixtures under vertex permutations, swaps, scaling, and translation."""
    points, ids, expected = [], [], []
    for p, topology, hit in fixture_cases():
        test.assertEqual(exact_self_intersection(p, topology), hit)
        for pa in permutations(range(3)):
            for pb in permutations(range(3)):
                order = [*pa, *(i + 3 for i in pb)]
                tp = topology.flatten()[order].reshape(2, 3)
                for swap in (False, True):
                    pp = p[order].reshape(2, 3, 3)
                    tt = tp
                    if swap:
                        pp, tt = pp[::-1], tt[::-1]
                    for scale, shift in ((1, 0), (0.125, 0), (8, 1024)):
                        points.append(pp.reshape(6, 3) * scale + shift)
                        # Noncontiguous IDs exercise matching, including negative IDs.
                        ids.append(tt * 17 - 100)
                        expected.append(hit)
    run_cases(test, device, points, ids, expected)
    run_cases(test, device, points, ids, expected, 1e-6)


def test_exact_random(test, device):
    """Match exact rational clipping on random shared-vertex and shared-edge pairs."""
    rng = np.random.default_rng(38171)
    points, ids, expected = [], [], []
    for shared in (1, 2):
        for planar in (False, True):
            for _ in range(256):
                while True:
                    p = rng.integers(-8, 9, (6, 3)).astype(np.float64) / 8
                    p[3 : 3 + shared] = p[:shared]
                    if planar:
                        p[:, 2] = 0
                    if np.any(np.cross(p[1] - p[0], p[2] - p[0])) and np.any(np.cross(p[4] - p[3], p[5] - p[3])):
                        break
                topology = np.array([[0, 1, 2], [0, 3, 4] if shared == 1 else [0, 1, 3]])
                hit = exact_self_intersection(p, topology)
                # Exercise normals with all dominant axes, plus non-axis-aligned
                # planes via an exactly represented integer shear.
                for matrix in (
                    np.eye(3),
                    np.eye(3)[[0, 2, 1]],
                    np.eye(3)[[2, 1, 0]],
                    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]]),
                ):
                    points.append(p @ matrix)
                    ids.append(topology)
                    expected.append(hit)
    run_cases(test, device, points, ids, expected)


def test_short_intersections(test, device):
    """Report intersections beyond a shared vertex regardless of their length."""
    points, ids, expected = [], [], []
    for exponent in (8, 16, 24, 32, 40):
        delta = 2.0**-exponent
        for sign in (-1, 1):
            # The overlap segment has length delta; the two triangles have
            # substantial noncoplanarity. A negative delta leaves only the origin.
            p = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (sign * delta, 0, -1), (sign * delta, 0, 1)]
            points.append(p)
            ids.append([[0, 1, 2], [0, 3, 4]])
            expected.append(sign > 0)
    run_cases(test, device, points, ids, expected)
    # Positive plane tolerance can project nearly coplanar inputs. Nevertheless,
    # these real overlaps must also survive the default and explicit tolerances.
    positive = np.flatnonzero(expected)
    result = wp.empty((len(positive), 4), dtype=bool, device=device)
    for dtype in (wp.vec3, wp.vec3d):
        wp.launch(
            tolerance_forms_kernel,
            dim=len(positive),
            inputs=[
                wp.array(np.array(points)[positive], dtype=dtype, device=device),
                wp.array(np.array(ids)[positive], dtype=wp.vec3i, device=device),
                result,
            ],
            device=device,
        )
        np.testing.assert_array_equal(result.numpy(), np.ones((len(positive), 4), dtype=bool))


def test_unrelated_equivalence(test, device):
    """Match intersect_tri_tri exactly when the triangles share no vertex IDs."""
    rng = np.random.default_rng(9441)
    p = rng.uniform(-1, 1, (4096, 6, 3))
    p[:1024, :, 2] = 0
    fixtures = [f[0] for f in fixture_cases()]
    p = np.concatenate((p, fixtures))
    for dtype in (wp.vec3, wp.vec3d):
        for tol_type in (wp.float32, wp.float64):
            tolerances = np.resize([0.0, 1e-6, 0.125], len(p))
            hits = wp.empty((len(p), 8), dtype=bool, device=device)
            wp.launch(
                compare_unrelated_kernel,
                dim=len(p),
                inputs=[
                    wp.array(p, dtype=dtype, device=device),
                    wp.array(tolerances, dtype=tol_type, device=device),
                    hits,
                ],
                device=device,
            )
            result = hits.numpy()
            for left, right in ((0, 1), (2, 3), (2, 4), (0, 5), (6, 7)):
                np.testing.assert_array_equal(result[:, left], result[:, right])


def test_tolerance(test, device):
    """Treat slight shared-edge folds as coplanar only within the cubic-scaled tolerance."""
    # Unit shared edge and unit-area-normal A make plane values equal height.
    # Scaling coordinates scales the plane value cubically.
    points, ids, tolerances, expected = [], [], [], []
    for scale in (0.5, 1, 2):
        for height in (-0.25, -0.125, 0, 0.125, 0.25):
            for tolerance in (0, 0.125, 0.25, 0.5):
                for side in (-1, 1):
                    points.append(
                        np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0), (0, side, height)]) * scale
                    )
                    ids.append([[0, 1, 2], [0, 1, 3]])
                    tolerances.append(tolerance)
                    expected.append(side == 1 and (height == 0 or abs(height) * scale**3 < tolerance))
    for dtype in (wp.vec3, wp.vec3d):
        for tol_type in (wp.float32, wp.float64):
            result = wp.empty(len(points), dtype=bool, device=device)
            wp.launch(
                self_intersect_kernel,
                dim=len(points),
                inputs=[
                    wp.array(points, dtype=dtype, device=device),
                    wp.array(ids, dtype=wp.vec3i, device=device),
                    wp.array(tolerances, dtype=tol_type, device=device),
                    result,
                ],
                device=device,
            )
            np.testing.assert_array_equal(result.numpy(), expected)


def test_nearly_collinear_sectors(test, device):
    """Separate nearly collinear coplanar sectors without dilating edges."""
    # PaMO's check_triangle_overlap treats small cross products as collinear.
    # Plane tolerance must not dilate coplanar triangle edges in this API.
    points, ids, expected = [], [], []
    for exponent in (8, 16, 24, 32, 40):
        delta = 2.0**-exponent
        for sign in (-1, 1):
            points.append([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (1, sign * delta, 0), (1, -1, 0)])
            ids.append([[0, 1, 2], [0, 3, 4]])
            expected.append(sign > 0)
    run_cases(test, device, points, ids, expected)
    run_cases(test, device, points, ids, expected, 1e-6)


def test_default_tolerance_boundary(test, device):
    """Distinguish the float64 default tolerance from a float32 literal at the cutoff."""
    cutoff = float(np.float32(1e-6))
    points, ids, expected = [], [], []
    for height in (0.0, cutoff / 2, cutoff, cutoff * 2):
        # One shared vertex: B's opposite edge is parallel to A's plane and
        # projects inside it when its height snaps to zero.
        points.append([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (0.25, 0.125, height), (0.125, 0.25, height)])
        ids.append([[0, 1, 2], [0, 3, 4]])
        expected.append([height < 1e-6, height < 1e-6, height < cutoff, height == 0])
        # Two shared vertices: a slight same-side fold becomes coplanar overlap.
        points.append([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0), (0, 1, height)])
        ids.append([[0, 1, 2], [0, 1, 3]])
        expected.append([height < 1e-6, height < 1e-6, height < cutoff, height == 0])
    for dtype in (wp.vec3, wp.vec3d):
        result = wp.empty((len(points), 4), dtype=bool, device=device)
        wp.launch(
            tolerance_forms_kernel,
            dim=len(points),
            inputs=[wp.array(points, dtype=dtype, device=device), wp.array(ids, dtype=wp.vec3i, device=device), result],
            device=device,
        )
        np.testing.assert_array_equal(result.numpy(), expected)


def test_thin_shared_edge(test, device):
    """Classify thin shared-edge triangles without underflowing projected areas."""
    # Multiplying projected areas would underflow in float32 for the thinner
    # triangles. Nonzero sign comparisons should retain same-side overlaps.
    points, ids, expected = [], [], []
    for exponent in (24, 60, 80, 100):
        width = 2.0**-exponent
        for side in (-1, 1):
            points.append([(0, 0, 0), (1, 0, 0), (0, width, 0), (0, 0, 0), (1, 0, 0), (0.5, side * width, 0)])
            ids.append([[0, 1, 2], [0, 1, 3]])
            expected.append(side > 0)
    run_cases(test, device, points, ids, expected)
    run_cases(test, device, points, ids, expected, 1e-6)


def test_mesh_pairs(test, device):
    """Flag only the duplicate face among all face pairs of valid meshes."""
    # A closed tetrahedron has only valid edge adjacency.
    vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float64)
    faces = np.array([(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
    pairs = np.array(list(combinations(range(len(faces)), 2)))
    ids = faces[pairs]
    run_cases(test, device, vertices[ids].reshape(-1, 6, 3), ids, [False] * len(pairs), 1e-6)

    # Indexed planar grid exercises shared vertices, shared edges, and unrelated
    # faces. A duplicate face is invalid, even with reversed winding.
    vertices = np.array([(x, y, 0) for y in range(4) for x in range(4)], dtype=np.float64)
    faces = []
    for y in range(3):
        for x in range(3):
            v = 4 * y + x
            faces.extend([(v, v + 1, v + 4), (v + 1, v + 5, v + 4)])
    faces.append(tuple(reversed(faces[0])))
    faces = np.array(faces)
    pairs = np.array(list(combinations(range(len(faces)), 2)))
    ids = faces[pairs]
    expected = [(i == 0 and j == len(faces) - 1) for i, j in pairs]
    run_cases(test, device, vertices[ids].reshape(-1, 6, 3), ids, expected, 1e-6)


class TestSelfIntersectTriTri(unittest.TestCase):
    pass


for name, function in (
    ("adjacency_fixtures", test_adjacency_fixtures),
    ("exact_random", test_exact_random),
    ("short_intersections", test_short_intersections),
    ("unrelated_equivalence", test_unrelated_equivalence),
    ("tolerance", test_tolerance),
    ("nearly_collinear_sectors", test_nearly_collinear_sectors),
    ("default_tolerance_boundary", test_default_tolerance_boundary),
    ("mesh_pairs", test_mesh_pairs),
    ("thin_shared_edge", test_thin_shared_edge),
):
    add_function_test(TestSelfIntersectTriTri, f"test_{name}", function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
