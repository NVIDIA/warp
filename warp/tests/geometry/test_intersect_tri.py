# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the triangle-triangle and segment-triangle predicates in ``warp.geometry``."""

import unittest
from fractions import Fraction
from itertools import permutations
from typing import Any

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import add_function_test, get_test_devices

# Native ray-triangle adapters restricted to the segment parameter range [0, 1].
# They add no coplanar handling, tolerance, or backface correction.


@wp.func_native("""
float t, u, v, sign;
bool hit = wp::intersect_ray_tri_woop(p0, p1-p0, a, b, c, t, u, v, sign, nullptr);
return hit && t >= 0.0f && t <= 1.0f;
""")
def native_woop(p0: wp.vec3, p1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> bool: ...


@wp.func_native("""
float t, u, v, w, sign;
bool hit = wp::intersect_ray_tri_moller(p0, p1-p0, a, b, c, t, u, v, w, sign, nullptr);
return hit && t >= 0.0f && t <= 1.0f;
""")
def native_moller(p0: wp.vec3, p1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> bool: ...


@wp.func_native("""
float t, u, v, w, sign;
bool hit = wp::intersect_ray_tri_rtcd(p0, p1-p0, a, b, c, t, u, v, w, sign, nullptr);
return hit && t >= 0.0f && t <= 1.0f;
""")
def native_rtcd(p0: wp.vec3, p1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> bool: ...


def make_segment_kernel(predicate):
    @wp.kernel(module="unique", enable_backward=False)
    def segment_predicate_kernel(points: wp.array2d[wp.vec3], hits: wp.array[bool]):
        i = wp.tid()
        hits[i] = predicate(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4])

    return segment_predicate_kernel


@wp.kernel(enable_backward=False)
def compare_tri_tri_kernel(points: wp.array2d[Any], native: wp.array[int], geometry: wp.array[bool]):
    i = wp.tid()
    native[i] = wp.intersect_tri_tri(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5])
    geometry[i] = wp.geometry.intersect_tri_tri(
        points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5]
    )


wp.overload(compare_tri_tri_kernel, [wp.array2d[wp.vec3], wp.array[int], wp.array[bool]])
wp.overload(compare_tri_tri_kernel, [wp.array2d[wp.vec3d], wp.array[int], wp.array[bool]])


def check_triangles(test, device, points, expected=None):
    for dtype in (wp.vec3, wp.vec3d):
        with test.subTest(dtype=dtype):
            inputs = wp.array(points, dtype=dtype, device=device)
            native = wp.empty(len(points), dtype=int, device=device)
            geometry = wp.empty(len(points), dtype=bool, device=device)
            wp.launch(compare_tri_tri_kernel, dim=len(points), inputs=[inputs, native, geometry], device=device)
            np.testing.assert_array_equal(geometry.numpy(), native.numpy())
            if expected is not None:
                np.testing.assert_array_equal(geometry.numpy(), expected)


def test_tri_random(test, device):
    """Match the native builtin on random and axis-aligned coplanar triangle pairs."""
    rng = np.random.default_rng(4829)
    points = rng.uniform(-1.0, 1.0, (10000, 6, 3))
    check_triangles(test, device, points)
    # Exercise all three dominant projection axes and both triangle windings.
    for axis in range(3):
        points[:, :, axis] = 0.25
        check_triangles(test, device, points)
        points = rng.uniform(-1.0, 1.0, (10000, 6, 3))


def test_tri_contacts(test, device):
    """Classify crossing, coplanar, touching, and separated triangles under permutations."""
    first = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    second = (
        ([[0.25, 0.25, -1], [0.25, 0.25, 1], [0.75, 0.25, 0]], 1),
        ([[0.1, 0.1, 0], [0.6, 0.1, 0], [0.1, 0.6, 0]], 1),
        ([[2, 0, 0], [3, 0, 0], [2, 1, 0]], 0),
        ([[1, 0, 0], [2, 0, 1], [2, 1, 0]], 1),
        ([[1, 0, 0], [0, 1, 0], [1, 1, 1]], 1),
        (first, 1),
        ([[0, 0, 1], [1, 0, 1], [0, 1, 1]], 0),
    )
    points = []
    expected = []
    # Check vertex permutations, swapped triangles, and all coordinate projections.
    for other, hit in second:
        for rotation in range(3):
            for reverse in (False, True):
                a = np.roll(first, rotation, axis=0)
                b = np.asarray(other)[::-1] if reverse else np.asarray(other)
                for axis in range(3):
                    pair = np.roll(np.concatenate((a, b)), axis, axis=1)
                    points.extend((pair, np.concatenate((pair[3:], pair[:3]))))
                    expected.extend((hit, hit))
    check_triangles(test, device, np.asarray(points), expected)


def test_tri_tolerance(test, device):
    """Match the native plane tolerance, including near-parallel and thin triangles."""
    first = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    points = []
    # Match the native tolerance, including the float32 value just below 1e-6.
    above_float32_cutoff = float(np.nextafter(np.float32(1e-6), np.float32(np.inf)))
    for height in (0.0, 0.5e-6, float(np.float32(1e-6)), above_float32_cutoff, 1e-6, 2e-6, -0.5e-6, -2e-6):
        other = first * 0.5 + (0.1, 0.1, height)
        points.append(np.concatenate((first, other)))
    # Almost parallel planes and thin triangles at several coordinate scales.
    for scale in (0.01, 1.0, 100.0):
        for height in (1e-7, 1e-5, 1e-3):
            other = np.array([[0.2, 0.2, -height], [0.2, 0.8, height], [0.8, 0.2, height]])
            points.append(np.concatenate((first, other)) * scale)
    check_triangles(test, device, np.asarray(points))


@wp.kernel(enable_backward=False)
def intersect_seg_tri_kernel(points: wp.array2d[Any], hits: wp.array[bool]):
    i = wp.tid()
    hits[i] = wp.geometry.intersect_seg_tri(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4])


wp.overload(intersect_seg_tri_kernel, [wp.array2d[wp.vec3], wp.array[bool]])
wp.overload(intersect_seg_tri_kernel, [wp.array2d[wp.vec3d], wp.array[bool]])


def check_segments(test, device, points, expected):
    for dtype in (wp.vec3, wp.vec3d):
        with test.subTest(dtype=dtype):
            data = wp.array(points, dtype=dtype, device=device)
            hits = wp.empty(len(points), dtype=bool, device=device)
            wp.launch(intersect_seg_tri_kernel, dim=len(points), inputs=[data, hits], device=device)
            np.testing.assert_array_equal(hits.numpy(), expected)


def segment_triangle_reference(points):
    """Compute an independent NumPy Möller-Trumbore reference for noncoplanar fixtures."""
    p0, p1, a, b, c = np.moveaxis(np.asarray(points, dtype=np.float64), 1, 0)
    direction = p1 - p0
    e1, e2 = b - a, c - a
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    normal = np.cross(e1, e2)
    distances = np.einsum("ij,ij->i", normal, p0 - a)
    # This reference intentionally excludes the coplanar case, covered separately.
    if np.any((np.abs(det) < 1.0e-12) & (np.abs(distances) < 1.0e-6)):
        raise ValueError("The NumPy reference requires noncoplanar segments.")
    inverse = np.divide(1.0, det, out=np.zeros_like(det), where=det != 0.0)
    s = p0 - a
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inverse * np.einsum("ij,ij->i", direction, q)
    t = inverse * np.einsum("ij,ij->i", e2, q)
    return (det != 0.0) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= 1.0)


def test_seg_contacts(test, device):
    """Classify segment contacts, coplanar overlaps, and degenerate inputs under permutations."""
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    cases = (
        # Interior crossing, miss, and an infinite-line hit beyond the segment.
        ((0.25, 0.25, -1), (0.25, 0.25, 1), True),
        ((0.75, 0.75, -1), (0.75, 0.75, 1), False),
        ((0.25, 0.25, 1), (0.25, 0.25, 2), False),
        # Endpoint, edge, and vertex contact from outside the plane.
        ((0.25, 0.25, 0), (0.25, 0.25, 1), True),
        ((0.5, 0.5, -1), (0.5, 0.5, 1), True),
        ((0, 0, -1), (0, 0, 1), True),
        # Coplanar containment, crossing with both endpoints outside, and miss.
        ((0.125, 0.125, 0), (0.5, 0.125, 0), True),
        ((-1, 0.25, 0), (2, 0.25, 0), True),
        ((1, 1, 0), (2, 2, 0), False),
        # Collinear overlap, disjoint collinear edges, and vertex-only contact.
        ((-1, 0, 0), (2, 0, 0), True),
        ((-2, 0, 0), (-1, 0, 0), False),
        ((-1, 0, 0), (0, 0, 0), True),
        ((-1, 1, 0), (1, -1, 0), True),
        ((0, 1, 0), (1, 0, 0), True),
        # Zero-length segments inside, outside, on a vertex, and on an edge.
        ((0.25, 0.25, 0), (0.25, 0.25, 0), True),
        ((1, 1, 0), (1, 1, 0), False),
        ((0, 0, 0), (0, 0, 0), True),
        ((0.5, 0.5, 0), (0.5, 0.5, 0), True),
        ((0.25, 0.25, 1), (0.25, 0.25, 1), False),
    )
    points, expected = [], []
    for p0, p1, hit in cases:
        for order in permutations(range(3)):
            for segment in ((p0, p1), (p1, p0)):
                pair = np.concatenate((segment, triangle[list(order)]))
                for axis in range(3):
                    for scale in (0.125, 1.0, 16.0):
                        points.append(np.roll(pair, axis, axis=1) * scale + (2.0, -3.0, 1.0))
                        expected.append(hit)
    # Degenerate triangles have a defined False result.
    points.extend(
        (
            [[0, 0, -1], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, -1], [0, 0, 1], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
        )
    )
    expected.extend((False, False))
    check_segments(test, device, np.asarray(points), expected)


def test_seg_random(test, device):
    """Match NumPy and native ray-triangle routines on random noncoplanar segments."""
    rng = np.random.default_rng(6927)
    # Quantized inputs are represented exactly in both precisions. Avoid the
    # plane tolerance and exact contact; those have explicit fixtures above.
    points = rng.integers(-16, 17, (10000, 5, 3)).astype(np.float64) / 8.0
    p0, p1, a, b, c = np.moveaxis(points, 1, 0)
    normal = np.cross(b - a, c - a)
    d0 = np.einsum("ij,ij->i", normal, p0 - a)
    d1 = np.einsum("ij,ij->i", normal, p1 - a)
    # Test general position here; exact rational contacts are tested separately.
    keep = (np.abs(d0) > 1.0e-4) & (np.abs(d1) > 1.0e-4) & (np.abs(d0 - d1) > 1.0e-4)
    points = points[keep]
    # Solve independently for intersection barycentrics and exclude contacts
    # susceptible to different floating-point rounding between algorithms.
    p0, p1, a, b, c = np.moveaxis(points, 1, 0)
    parameters = np.linalg.solve(np.stack((p1 - p0, a - b, a - c), axis=2), (a - p0)[..., None])[..., 0]
    t, u, v = parameters.T
    margin = np.min(np.abs(np.stack((t, 1 - t, u, v, 1 - u - v))), axis=0)
    points = points[margin > 1.0e-4]
    expected = segment_triangle_reference(points)
    check_segments(test, device, points, expected)
    p0, p1, a, b, c = np.moveaxis(points, 1, 0)
    front_facing = np.einsum("ij,ij->i", p0 - p1, np.cross(b - a, c - a)) > 0.0
    for name, predicate, native_expected in (
        ("woop", native_woop, expected),
        ("moller", native_moller, expected),
        ("rtcd", native_rtcd, expected & front_facing),
    ):
        with test.subTest(native=name):
            data = wp.array(points, dtype=wp.vec3, device=device)
            hits = wp.empty(len(points), dtype=bool, device=device)
            wp.launch(make_segment_kernel(predicate), dim=len(points), inputs=[data, hits], device=device)
            np.testing.assert_array_equal(hits.numpy(), native_expected)


def test_seg_native_semantics(test, device):
    """Document where segment results differ from the native ray-triangle routines."""
    triangle = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    segments = [
        [[0.25, 0.25, 1], [0.25, 0.25, -1]],  # Front-facing crossing.
        [[0.25, 0.25, -1], [0.25, 0.25, 1]],  # Back-facing crossing.
        [[0.25, 0.25, 1], [0.25, 0.25, 0]],  # Endpoint contact.
        [[0.5, 0.5, 1], [0.5, 0.5, -1]],  # Edge contact.
        [[0, 0, 1], [0, 0, -1]],  # Vertex contact.
        [[0.25, 0.25, 2], [0.25, 0.25, 1]],  # Ray hits beyond segment.
        [[-1, 0.25, 0], [2, 0.25, 0]],  # Coplanar crossing.
        [[0.25, 0.25, 0], [0.25, 0.25, 0]],  # Point inside triangle.
        [[0.25, 0.25, 0.5e-6], [0.5, 0.25, 0.5e-6]],  # Within plane tolerance.
    ]
    points = np.array([segment + triangle for segment in segments], dtype=np.float32)
    expectations = (
        ("geometry", wp.geometry.intersect_seg_tri, [1, 1, 1, 1, 1, 0, 1, 1, 1]),
        ("woop", native_woop, [1, 1, 1, 1, 1, 0, 0, 0, 0]),
        ("moller", native_moller, [1, 1, 1, 1, 1, 0, 0, 0, 0]),
        ("rtcd", native_rtcd, [1, 0, 1, 1, 1, 0, 0, 0, 0]),
    )
    data = wp.array(points, dtype=wp.vec3, device=device)
    for name, predicate, expected in expectations:
        with test.subTest(routine=name):
            hits = wp.empty(len(points), dtype=bool, device=device)
            wp.launch(make_segment_kernel(predicate), dim=len(points), inputs=[data, hits], device=device)
            np.testing.assert_array_equal(hits.numpy(), expected)


def test_seg_coplanar_random(test, device):
    """Match an exact rational oracle on random coplanar segments."""
    rng = np.random.default_rng(137)
    points, expected = [], []
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    # Exact rational oracle: a segment hits the unit right triangle iff an
    # endpoint is inside or it meets one of the three boundary lines inside it.
    for segment in rng.integers(-8, 17, (1000, 2, 2)):
        p, q = [[Fraction(int(x), 8) for x in endpoint] for endpoint in segment]
        direction = [q[i] - p[i] for i in range(2)]
        parameters = [Fraction(0), Fraction(1)]
        for numerator, denominator in (
            (-p[0], direction[0]),
            (-p[1], direction[1]),
            (1 - p[0] - p[1], direction[0] + direction[1]),
        ):
            if denominator:
                parameters.append(numerator / denominator)
        hit = False
        for t in parameters:
            x, y = [p[i] + t * direction[i] for i in range(2)]
            hit |= 0 <= t <= 1 and x >= 0 and y >= 0 and x + y <= 1
        pair = np.concatenate((np.column_stack((segment / 8.0, [0.0, 0.0])), triangle))
        for axis in range(3):
            points.append(np.roll(pair, axis, axis=1))
            expected.append(hit)
    check_segments(test, device, np.asarray(points), expected)


def test_seg_tolerance(test, device):
    """Snap segment endpoints using the native default plane tolerance in both precisions."""
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    points = []
    cutoff = np.float32(1.0e-6)
    below = float(np.nextafter(cutoff, np.float32(0.0)))
    above = float(np.nextafter(cutoff, np.float32(np.inf)))
    heights = (
        0.0,
        0.5e-6,
        below,
        float(cutoff),
        above,
        1.0e-6,
        2.0e-6,
        -0.5e-6,
        -below,
        -float(cutoff),
        -above,
        -2.0e-6,
    )
    for height in heights:
        points.append(np.concatenate(([[0.25, 0.25, height], [0.5, 0.25, height]], triangle)))
    # Float32 rounds 1e-6 to just below the double-precision threshold.
    for dtype, scalar in ((wp.vec3, np.float32), (wp.vec3d, np.float64)):
        expected = np.abs(np.asarray(heights, dtype=scalar).astype(np.float64)) < 1.0e-6
        with test.subTest(dtype=dtype):
            data = wp.array(points, dtype=dtype, device=device)
            hits = wp.empty(len(points), dtype=bool, device=device)
            wp.launch(intersect_seg_tri_kernel, dim=len(points), inputs=[data, hits], device=device)
            np.testing.assert_array_equal(hits.numpy(), expected)


def test_seg_snapped_endpoints(test, device):
    """Locate snapped endpoints by dominant-axis projection on an oblique triangle."""
    # Oblique triangle z=x: snapping uses its dominant-axis projection, not a
    # closest-point projection or the unsnapped ray-plane intersection.
    triangle = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    points, expected = [], []
    for height, hit in ((0.0, True), (0.5e-6, True), (2.0e-6, False)):
        cases = (
            ([[0.25, 0.25, 0.25 + height], [2, 2, 3]], hit),
            ([[1, 1, 1 + height], [0.25, 0.25, 1.25]], False),
        )
        for segment, result in cases:
            for reverse in (False, True):
                endpoints = np.asarray(segment)[::-1] if reverse else np.asarray(segment)
                for axis in range(3):
                    points.append(np.roll(np.concatenate((endpoints, triangle)), axis, axis=1))
                    expected.append(result)
    check_segments(test, device, np.asarray(points), expected)


@wp.kernel(enable_backward=False)
def intersect_custom_tolerances_kernel(
    triangles: wp.array2d[Any], segments: wp.array2d[Any], tolerances: wp.array[Any], hits: wp.array2d[bool]
):
    i = wp.tid()
    v0, v1, v2 = triangles[i, 0], triangles[i, 1], triangles[i, 2]
    u0, u1, u2 = triangles[i, 3], triangles[i, 4], triangles[i, 5]
    p0, p1 = segments[i, 0], segments[i, 1]
    tol = tolerances[i]
    hits[i, 0] = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol)
    hits[i, 1] = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol=tol)
    hits[i, 2] = wp.geometry.intersect_seg_tri(p0, p1, v0, v1, v2, tol)
    hits[i, 3] = wp.geometry.intersect_seg_tri(p0, p1, v0, v1, v2, tol=tol)


wp.overload(
    intersect_custom_tolerances_kernel,
    [wp.array2d[wp.vec3], wp.array2d[wp.vec3], wp.array[wp.float32], wp.array2d[bool]],
)
wp.overload(
    intersect_custom_tolerances_kernel,
    [wp.array2d[wp.vec3], wp.array2d[wp.vec3], wp.array[wp.float64], wp.array2d[bool]],
)
wp.overload(
    intersect_custom_tolerances_kernel,
    [wp.array2d[wp.vec3d], wp.array2d[wp.vec3d], wp.array[wp.float32], wp.array2d[bool]],
)
wp.overload(
    intersect_custom_tolerances_kernel,
    [wp.array2d[wp.vec3d], wp.array2d[wp.vec3d], wp.array[wp.float64], wp.array2d[bool]],
)


def test_custom_tolerances(test, device):
    """Apply runtime float32 and float64 tolerances with cubic coordinate scaling."""
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    triangles, segments, tolerances, expected = [], [], [], []
    # Binary-exact values test the strict cutoff without rounding ambiguity.
    # Keeping tol fixed while scaling the geometry also tests its cubic units.
    for scale in (0.5, 1.0, 4.0):
        for height in (0.0, 0.0625, 0.125, 0.25, -0.0625, -0.125, -0.25):
            for tol in (0.0, 0.0625, 0.125, 0.25, 1.0):
                triangles.append(np.concatenate((triangle, triangle + np.array([0, 0, height]))) * scale)
                segments.append(np.array([[0.25, 0.25, height], [0.5, 0.25, height]]) * scale)
                tolerances.append(tol)
                expected.append([height == 0.0 or abs(height) * scale**3 < tol] * 4)
    for dtype in (wp.vec3, wp.vec3d):
        for tol_dtype in (wp.float32, wp.float64):
            with test.subTest(dtype=dtype, tol_dtype=tol_dtype):
                inputs = [
                    wp.array(triangles, dtype=dtype, device=device),
                    wp.array(segments, dtype=dtype, device=device),
                    wp.array(tolerances, dtype=tol_dtype, device=device),
                ]
                hits = wp.empty((len(triangles), 4), dtype=bool, device=device)
                wp.launch(intersect_custom_tolerances_kernel, dim=len(triangles), inputs=[*inputs, hits], device=device)
                np.testing.assert_array_equal(hits.numpy(), expected)


@wp.kernel(enable_backward=False)
def intersect_tolerance_call_forms_kernel(points: wp.array2d[Any], hits: wp.array2d[bool]):
    i = wp.tid()
    v0, v1, v2 = points[i, 0], points[i, 1], points[i, 2]
    u0, u1, u2 = points[i, 3], points[i, 4], points[i, 5]
    hits[i, 0] = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2)
    hits[i, 1] = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol=wp.float64(1.0e-6))
    hits[i, 2] = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol=1.0e-6)
    hits[i, 3] = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2, 0.0)
    hits[i, 4] = wp.geometry.intersect_seg_tri(u0, u1, v0, v1, v2)
    hits[i, 5] = wp.geometry.intersect_seg_tri(u0, u1, v0, v1, v2, tol=wp.float64(1.0e-6))
    hits[i, 6] = wp.geometry.intersect_seg_tri(u0, u1, v0, v1, v2, tol=1.0e-6)
    hits[i, 7] = wp.geometry.intersect_seg_tri(u0, u1, v0, v1, v2, 0.0)


wp.overload(intersect_tolerance_call_forms_kernel, [wp.array2d[wp.vec3], wp.array2d[bool]])
wp.overload(intersect_tolerance_call_forms_kernel, [wp.array2d[wp.vec3d], wp.array2d[bool]])


def test_tolerance_call_forms(test, device):
    """Distinguish the float64 default tolerance from a float32 literal at the cutoff."""
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    cutoff = float(np.float32(1.0e-6))
    points = np.array([np.concatenate((triangle, triangle + np.array([0, 0, height]))) for height in (0.0, cutoff)])
    expected = [[True] * 8, [True, True, False, False] * 2]
    for dtype in (wp.vec3, wp.vec3d):
        with test.subTest(dtype=dtype):
            data = wp.array(points, dtype=dtype, device=device)
            hits = wp.empty((2, 8), dtype=bool, device=device)
            wp.launch(intersect_tolerance_call_forms_kernel, dim=2, inputs=[data, hits], device=device)
            np.testing.assert_array_equal(hits.numpy(), expected)


PREDICATES = (
    "intersect_tri_tri",
    "intersect_seg_tri",
    "intersect_aabb_aabb",
    "intersect_seg_aabb",
    "intersect_tri_aabb",
    "self_intersect_tri_tri",
)


class TestIntersectTri(unittest.TestCase):
    def test_python_scope(self):
        """Evaluate every intersection predicate overload at Python scope."""
        for vec in (wp.vec3, wp.vec3d):
            with self.subTest(dtype=vec):
                a, b, c = vec(0.0, 0.0, 0.0), vec(1.0, 0.0, 0.0), vec(0.0, 1.0, 0.0)
                p, q, r = vec(0.25, 0.25, -1.0), vec(0.25, 0.25, 1.0), vec(0.75, 0.25, 0.0)
                far = vec(5.0, 5.0, 5.0)
                ids = (wp.vec3i(0, 1, 2), wp.vec3i(3, 4, 5))
                for tol in ((), (wp.float32(0.0),), (wp.float64(0.0),)):
                    self.assertTrue(wp.geometry.intersect_tri_tri(a, b, c, p, q, r, *tol))
                    self.assertFalse(wp.geometry.intersect_tri_tri(a, b, c, far, far + p, far + r, *tol))
                    self.assertTrue(wp.geometry.intersect_seg_tri(p, q, a, b, c, *tol))
                    self.assertFalse(wp.geometry.intersect_seg_tri(far, far + q, a, b, c, *tol))
                    self.assertTrue(wp.geometry.self_intersect_tri_tri(a, b, c, p, q, r, *ids, *tol))
                lower, upper = vec(0.0, 0.0, 0.0), vec(1.0, 1.0, 1.0)
                self.assertTrue(wp.geometry.intersect_aabb_aabb(lower, upper, p, upper))
                self.assertFalse(wp.geometry.intersect_aabb_aabb(lower, upper, far, far + upper))
                self.assertTrue(wp.geometry.intersect_seg_aabb(p, q, lower, upper))
                self.assertFalse(wp.geometry.intersect_seg_aabb(far, far + q, lower, upper))
                self.assertTrue(wp.geometry.intersect_tri_aabb(a, b, c, lower, upper))
                self.assertFalse(wp.geometry.intersect_tri_aabb(far + a, far + b, far + c, lower, upper))

    def test_predicates_not_differentiable(self):
        """Mark every intersection predicate overload as non-differentiable."""
        for name in PREDICATES:
            function = getattr(wp.geometry, name)
            for overload in (function, *function.user_overloads.values()):
                with self.subTest(name=name, signature=overload.key):
                    self.assertFalse(overload.is_differentiable)


devices = get_test_devices()
for name, function in (
    ("test_tri_random", test_tri_random),
    ("test_tri_contacts", test_tri_contacts),
    ("test_tri_tolerance", test_tri_tolerance),
    ("test_seg_contacts", test_seg_contacts),
    ("test_seg_random", test_seg_random),
    ("test_seg_native_semantics", test_seg_native_semantics),
    ("test_seg_coplanar_random", test_seg_coplanar_random),
    ("test_seg_tolerance", test_seg_tolerance),
    ("test_seg_snapped_endpoints", test_seg_snapped_endpoints),
    ("test_custom_tolerances", test_custom_tolerances),
    ("test_tolerance_call_forms", test_tolerance_call_forms),
):
    add_function_test(TestIntersectTri, name, function, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
