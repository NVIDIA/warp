# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-triangle coordinates, distances, features, and derivatives against exact active sets."""

import unittest
from fractions import Fraction
from itertools import permutations
from typing import Any

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def query(points: wp.array2d[Any], barycentrics: wp.array[Any], distances: wp.array[Any], features: wp.array[int]):
    i = wp.tid()
    bary, d2, feature = wp.geometry.closest_point_point_tri(points[i, 0], points[i, 1], points[i, 2], points[i, 3])
    barycentrics[i] = bary
    distances[i] = d2
    features[i] = feature


@wp.kernel
def objective(bary: wp.array[Any], d2: wp.array[Any], loss: wp.array[Any]):
    loss[0] = d2[0] + type(d2[0])(0.3) * bary[0][1] + type(d2[0])(0.7) * bary[0][2]


for vector, scalar in ((wp.vec3, wp.float32), (wp.vec3d, wp.float64)):
    wp.overload(query, [wp.array2d[vector], wp.array[vector], wp.array[scalar], wp.array[int]])
    wp.overload(objective, [wp.array[vector], wp.array[scalar], wp.array[scalar]])


def feature_of(bary):
    support = tuple(i for i, value in enumerate(bary) if value > 0)
    return {(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5, (0, 1, 2): 6}[support]


def exact_reference(points):
    """Enumerate all convex-hull active sets using exact arithmetic on the supplied floats."""
    p, a, b, c = [[Fraction(float(value)) for value in point] for point in points]
    vertices = [a, b, c]

    def sub(x, y):
        return [u - v for u, v in zip(x, y, strict=True)]

    def dot(x, y):
        return sum(u * v for u, v in zip(x, y, strict=True))

    candidates = [[Fraction(int(i == j)) for i in range(3)] for j in range(3)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        edge = sub(vertices[j], vertices[i])
        denom = dot(edge, edge)
        if denom:
            t = dot(sub(p, vertices[i]), edge) / denom
            if 0 <= t <= 1:
                bary = [Fraction(0)] * 3
                bary[i], bary[j] = 1 - t, t
                candidates.append(bary)
    e, f, r = sub(b, a), sub(c, a), sub(p, a)
    ee, ef, ff = dot(e, e), dot(e, f), dot(f, f)
    det = ee * ff - ef * ef
    if det:
        er, fr = dot(e, r), dot(f, r)
        v, w = (ff * er - ef * fr) / det, (ee * fr - ef * er) / det
        if v >= 0 and w >= 0 and v + w <= 1:
            candidates.append([1 - v - w, v, w])

    def result(bary):
        closest = [sum(bary[j] * vertices[j][k] for j in range(3)) for k in range(3)]
        delta = sub(closest, p)
        return dot(delta, delta), feature_of(bary)

    bary = min(candidates, key=result)
    d2, feature = result(bary)
    return np.array(bary, dtype=np.float64), float(d2), feature, bool(det)


def evaluate(points, device, precision, requires_grad=False):
    vector, scalar = (wp.vec3, wp.float32) if precision == np.float32 else (wp.vec3d, wp.float64)
    data = wp.array(np.asarray(points, dtype=precision), dtype=vector, device=device, requires_grad=requires_grad)
    bary = wp.empty(len(points), dtype=vector, device=device, requires_grad=requires_grad)
    d2 = wp.empty(len(points), dtype=scalar, device=device, requires_grad=requires_grad)
    ids = wp.empty(len(points), dtype=int, device=device)
    wp.launch(query, len(points), inputs=[data, bary, d2, ids], device=device)
    return bary.numpy(), d2.numpy(), ids.numpy()


def check_cases(test, points, device, precision, features=True):
    points = np.asarray(points, dtype=precision)
    bary, d2, ids = evaluate(points, device, precision)
    test.assertTrue(np.isfinite(bary).all())
    test.assertTrue(np.isfinite(d2).all())
    test.assertTrue((bary >= 0).all())
    tol = 4e-6 if precision == np.float32 else 2e-13
    np.testing.assert_allclose(bary.sum(axis=1), 1, atol=tol, rtol=0)
    for i, case in enumerate(points):
        expected, expected_d2, expected_id, full_rank = exact_reference(case)
        scale2 = max(np.sum((case[j].astype(np.float64) - case[0]) ** 2) for j in range(1, 4))
        test.assertLessEqual(abs(float(d2[i]) - expected_d2), tol * max(scale2, expected_d2), msg=f"case {i}: {case}")
        np.testing.assert_allclose(
            bary[i].astype(np.float64) @ case[1:].astype(np.float64),
            expected @ case[1:].astype(np.float64),
            atol=tol * np.sqrt(scale2),
            rtol=tol,
            err_msg=f"case {i}",
        )
        test.assertEqual(ids[i], feature_of(bary[i]), msg=f"inconsistent returned feature, case {i}")
        if features and full_rank:
            test.assertEqual(ids[i], expected_id, msg=f"case {i}: {case}; bary {bary[i]} vs {expected}")


def test_point_triangle_regions(test, device):
    tri = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    points = [
        [-1, -1, 1],
        [2, -1, 1],
        [-1, 2, 1],
        [0.25, -1, 1],
        [-1, 0.25, 1],
        [1, 1, 1],
        [0.25, 0.25, 1],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0.5, 0, 1],
        [0, 0.5, 1],
        [0.5, 0.5, 1],
    ]
    cases = [[p, *order] for p in points for order in permutations(tri)]
    for precision in (np.float32, np.float64):
        check_cases(test, cases, device, precision)


def test_point_triangle_exact_random(test, device):
    rng = np.random.default_rng(2397)
    # Generic floats as well as dyadic coordinates, whose exact boundaries occur frequently.
    cases = np.concatenate([rng.normal(size=(256, 4, 3)), rng.integers(-8, 9, size=(256, 4, 3)) * 0.125])
    for precision in (np.float32, np.float64):
        check_cases(test, cases, device, precision)


def test_point_triangle_degenerate(test, device):
    cases = []
    for tri in (
        [[0, 0, 0]] * 3,
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        [[0, 0, 0], [1, 1, 1], [-2, -2, -2]],
    ):
        for order in permutations(tri):
            for p in ([0, 1, 0], [0.5, 1, 0], [2, 2, 2], [-3, -3, -3]):
                cases.append([p, *order])
    for precision in (np.float32, np.float64):
        check_cases(test, cases, device, precision)


def test_point_triangle_collinear_float(test, device):
    rng = np.random.default_rng(4183)
    for precision in (np.float32, np.float64):
        cases = []
        for _ in range(128):
            b = rng.normal(size=3).astype(precision)
            c = 2 * b
            p = rng.normal(size=3).astype(precision)
            for tri in permutations([np.zeros(3, dtype=precision), b, c]):
                cases.append([p, *tri])
        check_cases(test, cases, device, precision)
        _, _, ids = evaluate(cases, device, precision)
        test.assertFalse(np.any(ids == int(wp.geometry.TriangleFeature.FACE)))


def test_point_triangle_scales_and_slivers(test, device):
    cases = []
    for exponent in (-15, -12, -6, 0, 6, 12, 15):
        for thickness in (1, 1e-3, 1e-7, 1e-12):
            tri = np.array([[0, 0, 0], [1, 0, 0], [0.5, thickness, 0]])
            for p in ([0.5, 0.25 * thickness, 0.125], [0.5, -thickness, 1], [-1, 0, 0]):
                for order in permutations(tri):
                    cases.append(np.array([p, *order]) * 10.0**exponent)
    for precision in (np.float32, np.float64):
        check_cases(test, cases, device, precision)


def test_point_triangle_boundary_classification(test, device):
    for precision in (np.float32, np.float64):
        eps = np.finfo(precision).eps
        tri = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        cases = []
        for z in (0, 1, 1e6):
            for offset in (-eps, 0, eps):
                cases.extend([[p, *tri] for p in ([0.5, offset, z], [offset, 0.5, z], [0.5, 0.5 + offset, z])])
        check_cases(test, cases, device, precision)


def test_point_triangle_exact_boundaries(test, device):
    rng = np.random.default_rng(8241)
    cases = []
    for _ in range(256):
        a, b, c = rng.integers(-16, 17, size=(3, 3)).astype(np.float64)
        normal = np.cross(b - a, c - a)
        if np.any(normal):
            for left, right in ((a, b), (a, c), (b, c)):
                cases.append([0.5 * (left + right) + 0.25 * normal, a, b, c])
    for precision in (np.float32, np.float64):
        check_cases(test, cases, device, precision)


def test_point_triangle_rotated_slivers(test, device):
    rng = np.random.default_rng(783)
    cases = []
    for thickness in (1e-3, 1e-5):
        for _ in range(128):
            frame, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            a = rng.normal(size=3)
            ab, transverse, normal = frame.T
            b = a + ab
            c = a + 0.5 * ab + thickness * transverse
            p = a + 0.5 * ab + 0.3 * thickness * transverse + 1e-4 * normal
            cases.append([p, a, b, c])
    for precision in (np.float32, np.float64):
        check_cases(test, cases, device, precision)


def test_point_triangle_translation(test, device):
    tri = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    cases = []
    for p in ([0.25, 0.5, 0.75], [0.5, -0.25, 0.75], [3, -1, 1]):
        for offset in (0, 1024, 65536):
            cases.append(np.array([p, *tri]) + offset)
    for precision in (np.float32, np.float64):
        check_cases(test, cases, device, precision)


def test_point_triangle_gradients(test, device):
    for precision, vector, scalar, step, tol in (
        (np.float32, wp.vec3, wp.float32, 2e-3, 2e-3),
        (np.float64, wp.vec3d, wp.float64, 1e-5, 2e-8),
    ):
        for p in (
            [0.2, 0.3, 0.7],
            [0.3, -0.4, 0.6],
            [-0.3, -0.4, 0.7],
            [-0.4, 0.3, 0.6],
            [0.8, 0.9, 0.6],
            [1.4, -0.1, 0.7],
            [-0.1, 1.4, 0.7],
        ):
            raw = np.array([[p, [0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=precision)
            data = wp.array(raw, dtype=vector, device=device, requires_grad=True)
            bary = wp.empty(1, dtype=vector, device=device, requires_grad=True)
            d2 = wp.empty(1, dtype=scalar, device=device, requires_grad=True)
            ids = wp.empty(1, dtype=int, device=device)
            loss = wp.zeros(1, dtype=scalar, device=device, requires_grad=True)
            with wp.Tape() as tape:
                wp.launch(query, 1, inputs=[data, bary, d2, ids], device=device)
                wp.launch(objective, 1, inputs=[bary, d2, loss], device=device)
            tape.backward(loss)
            actual = data.grad.numpy()
            numerical = np.zeros_like(raw)
            for j in range(4):
                for k in range(3):
                    plus, minus = raw.copy(), raw.copy()
                    plus[0, j, k] += step
                    minus[0, j, k] -= step
                    bp, dp, _ = evaluate(plus, device, precision)
                    bm, dm, _ = evaluate(minus, device, precision)
                    numerical[0, j, k] = (dp[0] - dm[0] + 0.3 * (bp[0, 1] - bm[0, 1]) + 0.7 * (bp[0, 2] - bm[0, 2])) / (
                        2 * step
                    )
            np.testing.assert_allclose(actual, numerical, atol=tol, rtol=tol)


def test_point_triangle_distance_gradients(test, device):
    cases = []
    for scale in (1e-6, 1.0, 1e6):
        for p in ([0.25, 0.25, 0.75], [0.25, -0.5, 0.75], [-0.5, -0.25, 0.75]):
            cases.append(np.array([p, [0, 0, 0], [1, 0, 0], [0, 1, 0]]) * scale)
        cases.append(np.array([[0.5, 2.5e-6, 0.75], [0, 0, 0], [1, 0, 0], [0.5, 1e-5, 0]]) * scale)
    for precision, vector, scalar, tol in (
        (np.float32, wp.vec3, wp.float32, 2e-5),
        (np.float64, wp.vec3d, wp.float64, 2e-13),
    ):
        raw = np.asarray(cases, dtype=precision)
        data = wp.array(raw, dtype=vector, device=device, requires_grad=True)
        bary = wp.empty(len(raw), dtype=vector, device=device, requires_grad=True)
        d2 = wp.empty(len(raw), dtype=scalar, device=device, requires_grad=True)
        ids = wp.empty(len(raw), dtype=int, device=device)
        with wp.Tape() as tape:
            wp.launch(query, len(raw), inputs=[data, bary, d2, ids], device=device)
        tape.backward(grads={d2: wp.ones(len(raw), dtype=scalar, device=device)})
        weights = bary.numpy().astype(np.float64)
        xyz = raw.astype(np.float64)
        closest = xyz[:, 1] + weights[:, 1:2] * (xyz[:, 2] - xyz[:, 1]) + weights[:, 2:3] * (xyz[:, 3] - xyz[:, 1])
        point_grad = 2 * (xyz[:, 0] - closest)
        expected = np.concatenate([point_grad[:, None], -weights[:, :, None] * point_grad[:, None]], axis=1)
        scales = np.max(np.abs(xyz), axis=(1, 2))[:, None, None]
        np.testing.assert_allclose(data.grad.numpy() / scales, expected / scales, atol=tol, rtol=tol)


class TestClosestPointTriangle(unittest.TestCase):
    pass


for test in (
    test_point_triangle_regions,
    test_point_triangle_exact_random,
    test_point_triangle_degenerate,
    test_point_triangle_collinear_float,
    test_point_triangle_scales_and_slivers,
    test_point_triangle_boundary_classification,
    test_point_triangle_exact_boundaries,
    test_point_triangle_rotated_slivers,
    test_point_triangle_translation,
    test_point_triangle_gradients,
    test_point_triangle_distance_gradients,
):
    add_function_test(TestClosestPointTriangle, test.__name__, test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
