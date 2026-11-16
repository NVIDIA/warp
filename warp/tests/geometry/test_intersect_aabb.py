# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the box intersection predicates in ``warp.geometry``."""

import unittest
from fractions import Fraction
from itertools import permutations
from typing import Any

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.func_native("return wp::intersect_aabb_aabb(a, b, lower, upper);")
def native_aabb(a: wp.vec3, b: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> bool: ...


@wp.func_native("""
wp::vec3 d = p1-p0;
wp::vec3 rcp(1.0f/d[0], 1.0f/d[1], 1.0f/d[2]);
float t;
return wp::intersect_ray_aabb_robust(p0, d, rcp, lower, upper, t) && t <= 1.0f;
""")
def native_ray(p0: wp.vec3, p1: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> bool: ...


@wp.kernel(enable_backward=False)
def native_aabb_kernel(points: wp.array2d[wp.vec3], hits: wp.array[bool]):
    i = wp.tid()
    hits[i] = native_aabb(points[i, 0], points[i, 1], points[i, 2], points[i, 3])


@wp.kernel(enable_backward=False)
def native_ray_kernel(points: wp.array2d[wp.vec3], hits: wp.array[bool]):
    i = wp.tid()
    hits[i] = native_ray(points[i, 0], points[i, 1], points[i, 2], points[i, 3])


@wp.kernel(enable_backward=False)
def intersect_aabb_aabb_kernel(points: wp.array2d[Any], hits: wp.array[bool]):
    i = wp.tid()
    hits[i] = wp.geometry.intersect_aabb_aabb(points[i, 0], points[i, 1], points[i, 2], points[i, 3])


@wp.kernel(enable_backward=False)
def intersect_seg_aabb_kernel(points: wp.array2d[Any], hits: wp.array[bool]):
    i = wp.tid()
    hits[i] = wp.geometry.intersect_seg_aabb(points[i, 0], points[i, 1], points[i, 2], points[i, 3])


@wp.kernel(enable_backward=False)
def intersect_tri_aabb_kernel(points: wp.array2d[Any], hits: wp.array[bool]):
    i = wp.tid()
    hits[i] = wp.geometry.intersect_tri_aabb(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4])


wp.overload(intersect_aabb_aabb_kernel, [wp.array2d[wp.vec3], wp.array[bool]])
wp.overload(intersect_aabb_aabb_kernel, [wp.array2d[wp.vec3d], wp.array[bool]])
wp.overload(intersect_seg_aabb_kernel, [wp.array2d[wp.vec3], wp.array[bool]])
wp.overload(intersect_seg_aabb_kernel, [wp.array2d[wp.vec3d], wp.array[bool]])
wp.overload(intersect_tri_aabb_kernel, [wp.array2d[wp.vec3], wp.array[bool]])
wp.overload(intersect_tri_aabb_kernel, [wp.array2d[wp.vec3d], wp.array[bool]])


def clipped_polygon_intersects_box(points):
    """Clip a polygon to a box using exact rational arithmetic, independently of the separating-axis tests.

    One- and two-vertex polygons cover point and segment degeneracies as well.
    Input dyadic coordinates have exact representations in both tested precisions.
    """
    polygon = [tuple(Fraction(float(x)) for x in p) for p in points[:-2]]
    lower, upper = points[-2:]
    for axis in range(3):
        for slab_bound, sign in ((lower[axis], 1), (upper[axis], -1)):
            if not polygon:
                return False
            bound = Fraction(float(slab_bound))
            output = []
            previous = polygon[-1]
            d_previous = sign * (previous[axis] - bound)
            for current in polygon:
                d_current = sign * (current[axis] - bound)
                if (d_previous < 0) != (d_current < 0):
                    t = d_previous / (d_previous - d_current)
                    output.append(tuple(a + t * (b - a) for a, b in zip(previous, current, strict=True)))
                if d_current >= 0:
                    output.append(current)
                previous, d_previous = current, d_current
            polygon = output
    return bool(polygon)


def check_cases(test, device, kernel, cases, expected):
    for dtype in (wp.vec3, wp.vec3d):
        data = wp.array(cases, dtype=dtype, device=device)
        out = wp.empty(len(cases), dtype=bool, device=device)
        wp.launch(kernel, dim=len(cases), inputs=[data, out], device=device)
        with test.subTest(dtype=dtype):
            np.testing.assert_array_equal(out.numpy(), expected)


def test_boxes(test, device):
    """Classify random and explicit box pairs, including contact and zero-width boxes."""
    rng = np.random.default_rng(419)
    points = rng.integers(-8, 9, (3000, 4, 3)).astype(np.float64) / 4
    points[:, :2] = np.sort(points[:, :2], axis=1)
    points[:, 2:] = np.sort(points[:, 2:], axis=1)
    # Explicit disjoint, containing, identical, touching, and zero-width boxes.
    fixtures = []
    for lower, upper in (
        ((1, 0, 0), (2, 1, 1)),
        ((1, 1, 0), (2, 2, 1)),
        ((1, 1, 1), (2, 2, 2)),
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ((1.25, 0, 0), (2, 1, 1)),
        ((0, 0, 0), (1, 1, 1)),
    ):
        fixtures.append(((0, 0, 0), (1, 1, 1), lower, upper))
    points = np.concatenate((points, fixtures))
    expected = np.all((points[:, 0] <= points[:, 3]) & (points[:, 2] <= points[:, 1]), axis=1)
    check_cases(test, device, intersect_aabb_aabb_kernel, points, expected)
    check_cases(test, device, intersect_aabb_aabb_kernel, points[:, [2, 3, 0, 1]], expected)


def primitive_cases(arity):
    rng = np.random.default_rng(871 + arity)
    points = rng.integers(-8, 9, (1200, arity + 2, 3)).astype(np.float64) / 4
    points[:, -2:] = np.sort(points[:, -2:], axis=1)
    # Collapsed boxes, points, segments, and planar geometry.
    points[::7, -1, 2] = points[::7, -2, 2]
    points[::11, -1] = points[::11, -2]
    points[::13, 1] = points[::13, 0]
    if arity == 3:
        points[::17, 2] = (points[::17, 0] + points[::17, 1]) / 2
        fixtures = [
            [(-2, -2, 0), (2, -2, 0), (0, 2, 0), (-0.25, -0.25, -0.25), (0.25, 0.25, 0.25)],
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.75, 0.75, 0), (0.75, 0.75, 0)],
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.25, 0.25, 0), (0.25, 0.25, 0)],
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.5, 0.5, 0), (1, 1, 1)],
            [(-2, 0, 0), (2, 0, 0), (0, 0, 0), (-1, -1, -1), (1, 1, 1)],
        ]
    else:
        fixtures = [
            [(-2, 0, 0), (2, 0, 0), (-1, -1, -1), (1, 1, 1)],
            [(-2, 1, 1), (2, 1, 1), (-1, -1, -1), (1, 1, 1)],
            [(0, 0, 0), (0, 0, 0), (-1, -1, -1), (1, 1, 1)],
            [(2, 0, 0), (2, 0, 0), (-1, -1, -1), (1, 1, 1)],
            [(-2, 0, 0), (0, 2, 0), (-1, -1, -1), (1, 1, 1)],
            [(-2, 0, 0), (0, 2, 0), (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)],
        ]
    return np.concatenate((points, fixtures))


def check_primitive(test, device, arity, kernel):
    points = primitive_cases(arity)
    expected = [clipped_polygon_intersects_box(p) for p in points]
    for order in permutations(range(arity)):
        permuted = points[:, [*order, arity, arity + 1]]
        for scale, shift in ((1, 0), (0.125, 0), (8, 1024)):
            check_cases(test, device, kernel, permuted * scale + shift, expected)


def test_segments(test, device):
    """Match exact clipping for segments and boxes under permutations, scaling, and translation."""
    check_primitive(test, device, 2, intersect_seg_aabb_kernel)


def test_triangles(test, device):
    """Match exact clipping for triangles and boxes under permutations, scaling, and translation."""
    check_primitive(test, device, 3, intersect_tri_aabb_kernel)


def test_coplanar_and_degenerate(test, device):
    """Match exact clipping for planar inputs and point-sized boxes outside the triangle."""
    points = primitive_cases(3)
    points[:, :, 2] = 0
    extra = np.array(
        [
            [(0, 0, 0), (0, 0, 0), (0, 0, 0), (-1, -1, -1), (1, 1, 1)],
            [(2, 0, 0), (2, 0, 0), (2, 0, 0), (-1, -1, -1), (1, 1, 1)],
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.25, 0.25, -1), (0.25, 0.25, 1)],
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.75, 0.75, -1), (0.75, 0.75, 1)],
        ]
    )
    points = np.concatenate((points, extra))
    expected = [clipped_polygon_intersects_box(p) for p in points]
    for order in permutations(range(3)):
        check_cases(test, device, intersect_tri_aabb_kernel, points[:, :, order], expected)


def test_native_comparison(test, device):
    """Match the native box-box and robust ray-box routines on continuous random inputs."""
    # Continuous random coordinates avoid the exact-contact rounding differences
    # between division-based clipping and division-free separating-axis tests.
    rng = np.random.default_rng(20261029)
    for kind, kernel, native_kernels in (
        ("aabb", intersect_aabb_aabb_kernel, (native_aabb_kernel,)),
        ("seg", intersect_seg_aabb_kernel, (native_ray_kernel,)),
    ):
        cases = rng.uniform(-1, 1, (8192, 4, 3)).astype(np.float32)
        cases[:, 2:] = np.sort(cases[:, 2:], axis=1)
        if kind == "aabb":
            cases[:, :2] = np.sort(cases[:, :2], axis=1)
        data = wp.array(cases, dtype=wp.vec3, device=device)
        result = wp.empty(len(cases), dtype=bool, device=device)
        wp.launch(kernel, dim=len(cases), inputs=[data, result], device=device)
        expected = result.numpy()
        for native_kernel in native_kernels:
            wp.launch(native_kernel, dim=len(cases), inputs=[data, result], device=device)
            with test.subTest(kind=kind, native=native_kernel.key):
                np.testing.assert_array_equal(result.numpy(), expected)


def test_no_implicit_margin(test, device):
    """Reject primitives one representable step outside the box."""
    for dtype, scalar in ((wp.vec3, np.float32), (wp.vec3d, np.float64)):
        outside = np.nextafter(scalar(1), scalar(2))
        lower = (0, 0, 0)
        upper = (1, 1, 1)
        cases = {
            intersect_aabb_aabb_kernel: [[(outside, 0, 0), (2, 1, 1), lower, upper]],
            intersect_seg_aabb_kernel: [[(outside, 0, 0), (outside, 1, 1), lower, upper]],
            intersect_tri_aabb_kernel: [[(outside, 0, 0), (outside, 1, 0), (outside, 0, 1), lower, upper]],
        }
        for kernel, points in cases.items():
            result = wp.empty(1, dtype=bool, device=device)
            wp.launch(kernel, dim=1, inputs=[wp.array(points, dtype=dtype, device=device), result], device=device)
            test.assertFalse(result.numpy()[0])


class TestIntersectAABB(unittest.TestCase):
    pass


devices = get_test_devices()
for name, function in (
    ("test_boxes", test_boxes),
    ("test_segments", test_segments),
    ("test_triangles", test_triangles),
    ("test_coplanar_and_degenerate", test_coplanar_and_degenerate),
    ("test_native_comparison", test_native_comparison),
    ("test_no_implicit_margin", test_no_implicit_margin),
):
    add_function_test(TestIntersectAABB, name, function, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
