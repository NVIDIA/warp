# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent convex-hull oracle tests for closest pairs."""

import unittest
from fractions import Fraction
from itertools import combinations, permutations
from typing import Any

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def pairs_kernel(points: wp.array2d[Any], coordinates: wp.array2d[Any], distances: wp.array[Any], triangle: bool):
    i = wp.tid()
    if triangle:
        bary, t, d2 = wp.geometry.closest_points_tri_seg(
            points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4]
        )
        coordinates[i, 0] = bary[0]
        coordinates[i, 1] = bary[1]
        coordinates[i, 2] = bary[2]
        coordinates[i, 3] = t
        distances[i] = d2
    else:
        s, t, d2 = wp.geometry.closest_points_seg_seg(points[i, 0], points[i, 1], points[i, 2], points[i, 3])
        coordinates[i, 0] = s
        coordinates[i, 1] = t
        coordinates[i, 2] = type(points[i, 0][0])(0.0)
        coordinates[i, 3] = type(points[i, 0][0])(0.0)
        distances[i] = d2


@wp.kernel
def objective(coordinates: wp.array2d[Any], distances: wp.array[Any], loss: wp.array[Any]):
    loss[0] = distances[0]
    for j in range(4):
        loss[0] += type(distances[0])(0.17 * float(j + 1)) * coordinates[0, j]


@wp.kernel
def scaled_objective(coordinates: wp.array2d[Any], distances: wp.array[Any], loss: wp.array[Any], parameters: bool):
    loss[0] = distances[0]
    if parameters:
        loss[0] = type(distances[0])(0.0)
        for j in range(4):
            loss[0] += type(distances[0])(0.17 * float(j + 1)) * coordinates[0, j]


for vector, scalar in ((wp.vec3, wp.float32), (wp.vec3d, wp.float64)):
    wp.overload(pairs_kernel, [wp.array2d[vector], wp.array2d[scalar], wp.array[scalar], bool])
    wp.overload(objective, [wp.array2d[scalar], wp.array[scalar], wp.array[scalar]])
    wp.overload(scaled_objective, [wp.array2d[scalar], wp.array[scalar], wp.array[scalar], bool])


def _solve_exact(matrix, rhs):
    """Gaussian elimination over rationals; singular faces are covered by smaller faces."""
    n = len(rhs)
    rows = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if rows[i][j]), None)
        if pivot is None:
            return None
        rows[j], rows[pivot] = rows[pivot], rows[j]
        value = rows[j][j]
        rows[j] = [x / value for x in rows[j]]
        for i in range(n):
            if i != j:
                value = rows[i][j]
                rows[i] = [x - value * y for x, y in zip(rows[i], rows[j], strict=True)]
    return [row[-1] for row in rows]


def hull_distance_squared(first, second, exact=False):
    """Enumerate active sets of a convex quadratic program independently of feature code."""
    if exact:
        first = [[Fraction(float(x)) for x in row] for row in first]
        second = [[Fraction(float(x)) for x in row] for row in second]
    best = float("inf")
    for na in range(1, len(first) + 1):
        for ia in combinations(range(len(first)), na):
            for nb in range(1, len(second) + 1):
                for ib in combinations(range(len(second)), nb):
                    r = [first[ia[0]][k] - second[ib[0]][k] for k in range(3)]
                    columns = [[first[j][k] - first[ia[0]][k] for k in range(3)] for j in ia[1:]]
                    columns += [[second[ib[0]][k] - second[j][k] for k in range(3)] for j in ib[1:]]
                    if columns:
                        if exact:
                            gram = [[sum(x * y for x, y in zip(a, b, strict=True)) for b in columns] for a in columns]
                            rhs = [-sum(x * y for x, y in zip(a, r, strict=True)) for a in columns]
                            z = _solve_exact(gram, rhs)
                            if z is None:
                                continue
                        else:
                            z = np.linalg.lstsq(np.array(columns).T, -np.array(r), rcond=None)[0]
                        if any(x < 0 for x in z) or sum(z[: na - 1]) > 1 or sum(z[na - 1 :]) > 1:
                            continue
                        r = [r[k] + sum(z[j] * columns[j][k] for j in range(len(columns))) for k in range(3)]
                    best = min(best, sum(x * x for x in r))
    return float(best)


def evaluate(points, triangle, device, dtype=np.float64, requires_grad=False):
    vector, scalar = (wp.vec3, wp.float32) if dtype == np.float32 else (wp.vec3d, wp.float64)
    data = wp.array(np.asarray(points, dtype=dtype), dtype=vector, device=device, requires_grad=requires_grad)
    coords = wp.empty((len(points), 4), dtype=scalar, device=device, requires_grad=requires_grad)
    d2 = wp.empty(len(points), dtype=scalar, device=device, requires_grad=requires_grad)
    wp.launch(pairs_kernel, len(points), inputs=[data, coords, d2, triangle], device=device)
    return data, coords, d2


def check_cases(test, points, triangle, device, exact=False):
    n = 3 if triangle else 2
    for dtype in (np.float32, np.float64):
        p = np.asarray(points, dtype=dtype)
        _, coords_wp, d2_wp = evaluate(p, triangle, device, dtype)
        coords, d2 = coords_wp.numpy(), d2_wp.numpy()
        reference = np.array([hull_distance_squared(row[:n], row[n:], exact=exact) for row in p.astype(float)])
        scale2 = np.max(np.sum((p.astype(float) - p[:, :1].astype(float)) ** 2, axis=-1), axis=1)
        tolerance = 3e-6 if dtype == np.float32 else 2e-13
        np.testing.assert_allclose(
            d2 / np.maximum(scale2, 1e-100), reference / np.maximum(scale2, 1e-100), atol=tolerance
        )
        np.testing.assert_allclose(
            np.sqrt(d2 / np.maximum(scale2, 1e-100)),
            np.sqrt(reference / np.maximum(scale2, 1e-100)),
            atol=tolerance,
        )
        test.assertTrue(np.isfinite(coords).all())
        test.assertTrue(np.isfinite(d2).all())
        test.assertTrue((coords >= 0).all() and (coords <= 1).all())
        if triangle:
            np.testing.assert_allclose(coords[:, :3].sum(axis=1), 1, atol=tolerance)
            x = np.sum(coords[:, :3, None] * p[:, :3], axis=1)
            y = p[:, 3] + coords[:, 3, None] * (p[:, 4] - p[:, 3])
        else:
            x = p[:, 0] + coords[:, 0, None] * (p[:, 1] - p[:, 0])
            y = p[:, 2] + coords[:, 1, None] * (p[:, 3] - p[:, 2])
        np.testing.assert_allclose(
            np.sum((x - y) ** 2, axis=1) / np.maximum(scale2, 1e-100), d2 / np.maximum(scale2, 1e-100), atol=tolerance
        )


def test_closest_random(test, device):
    rng = np.random.default_rng(25)
    for triangle in (False, True):
        n = 5 if triangle else 4
        points = rng.normal(size=(300, n, 3))
        points[::7, 1] = points[::7, 0]
        points[::11, -1] = points[::11, -2]
        if triangle:
            points[::13, 2] = (points[::13, 0] + points[::13, 1]) * 0.5
        check_cases(test, points, triangle, device)


def test_closest_exact(test, device):
    rng = np.random.default_rng(26)
    for triangle in (False, True):
        points = rng.integers(-4, 5, size=(40, 5 if triangle else 4, 3))
        points[::5, 1] = points[::5, 0]
        points[::7, -1] = points[::7, -2]
        check_cases(test, points, triangle, device, exact=True)


def test_closest_features(test, device):
    tri = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    segments = [
        [[0.25, 0.25, -1], [0.25, 0.25, 1]],  # piercing
        [[-1, 0.25, 1], [0.5, 0.25, 1]],  # parallel, projected interior
        [[-1, 0.25, 0], [0.5, 0.25, 0]],  # coplanar overlap
        [[3, 3, 0], [4, 4, 0]],  # coplanar separated
        [[0, 0, 0], [0, 0, 0]],  # point at vertex
        [[0.5, 0.5, 1], [0.5, 0.5, 2]],  # endpoint over interior
        [[0, 0, -1], [0, 0, 1]],  # vertex piercing
    ]
    points = [list(order) + seg[::sign] for order in permutations(tri) for seg in segments for sign in (1, -1)]
    for scale in (1e-6, 1.0, 1e6):
        check_cases(test, np.array(points) * scale, True, device)
    # Collinear overlap, disjointness, coincident endpoints, and collapsed pairs.
    edge_cases = [
        [[0, 0, 0], [2, 0, 0], [1, 0, 0], [3, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
    ]
    variants = []
    for points in edge_cases:
        first, second = points[:2], points[2:]
        for a, b in ((first, second), (second, first)):
            for sa in (1, -1):
                for sb in (1, -1):
                    variants.append(a[::sa] + b[::sb])
    check_cases(test, variants, False, device, exact=True)
    for angle in (0.0, 1e-3, 1e-6, 1e-9):
        p = np.array([[-1, 0, 0], [1, 0, 0], [-1, -angle, 0.1], [1, angle, 0.1]])
        cases = [np.concatenate((a, b)) for a, b in ((p[:2], p[2:]), (p[2:], p[:2]))]
        check_cases(test, cases, False, device)
        if angle:
            for dtype in (np.float32, np.float64):
                _, coords, _ = evaluate([p], False, device, dtype)
                np.testing.assert_allclose(coords.numpy()[0, :2], [0.5, 0.5], atol=1e-6)


def test_closest_near_parallel(test, device):
    rng = np.random.default_rng(2026)
    points = []
    for angle in (1e-2, 1e-4, 1e-6, 1e-8):
        for _ in range(12):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            v = rng.normal(size=3)
            v -= np.dot(u, v) * u
            v /= np.linalg.norm(v)
            # Interior minima even when the Gram determinant rounds to zero.
            offset = np.cross(u, v) * 1e-3
            points.append([-u, u, -u - angle * v + offset, u + angle * v + offset])
    check_cases(test, points, False, device, exact=True)


def test_closest_edge_piercing(test, device):
    # Segments piercing exactly on edge bc: v + w <= 1 can hold after rounding
    # while 1 - v - w is negative, so the returned weight must be the one tested.
    rng = np.random.default_rng(30)
    tri = rng.normal(size=(256, 3, 3))
    lam = np.round(rng.uniform(0, 1, (256, 1)) * 4096) / 4096
    x = (1 - lam) * tri[:, 1] + lam * tri[:, 2]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    check_cases(test, np.concatenate((tri, np.stack((x - normal, x + normal), axis=1)), axis=1), True, device)


def test_closest_extreme_scales(test, device):
    # Segment pairs must not lose accuracy when the fourth power of their length
    # leaves the float32 range (crossing segments used to report s = 0).
    rng = np.random.default_rng(31)
    points = rng.normal(size=(64, 4, 3))
    points[0] = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
    for scale in (1e-15, 1e-12, 1e10, 1e15):
        check_cases(test, points * scale, False, device)
    triangles = rng.normal(size=(64, 5, 3))
    triangles[0] = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0.5, 0.5, -1], [0.5, 0.5, 1]]
    for scale in (1e-15, 1e-12, 1e10, 1e15):
        check_cases(test, triangles * scale, True, device)


def test_closest_mixed_segment_lengths(test, device):
    # Normalizing v by its squared length overflows the determinant when the
    # first segment is long and the second short, despite a well-conditioned
    # right angle and representable squared lengths. Check absolute distance:
    # a tolerance relative to the longest edge would hide this regression.
    for length in (1e3, 1e10, 1e15):
        first = [[-length, 0, 1], [length, 0, 1]]
        second = [[0, -1 / length, 0], [0, 1 / length, 0]]
        points = [first + second, second + first]
        for dtype in (np.float32, np.float64):
            _, coordinates, d2 = evaluate(points, False, device, dtype)
            np.testing.assert_allclose(coordinates.numpy()[:, :2], 0.5, atol=1e-6)
            np.testing.assert_allclose(d2.numpy(), 1.0, atol=1e-6)


def test_closest_thin_triangles(test, device):
    cases = []
    for width in (1e-3, 1e-6, 1e-9):
        tri = np.array([[0, 0, 0], [1, 0, 0], [0.5, width, 0]])
        for seg in (
            [[0.5, width * 0.25, -1], [0.5, width * 0.25, 1]],
            [[0.4, width * 0.25, 0.1], [0.6, width * 0.25, 0.1]],
        ):
            cases.append(np.concatenate((tri, seg)))
    check_cases(test, cases, True, device, exact=True)


def test_closest_capsule_witnesses(test, device):
    # Validate the capsule identity and reconstructed contacts in local/world
    # frames, including a zero-height capsule (sphere). Axis hits have no unique
    # normal; do not pretend a deterministic fallback is a surface projection.
    triangles = np.array(
        [
            [[0.4, -0.2, -0.1], [0.4, 0.2, -0.1], [0.4, 0, 0.2]],  # side, separated
            [[0.2, -0.2, -0.1], [0.2, 0.2, -0.1], [0.2, 0, 0.2]],  # side, penetrating
            [[-0.2, -0.2, 0.7], [0.2, -0.2, 0.7], [0, 0.2, 0.7]],  # cap
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]],  # crosses axis
            [[0.349, 0, 0], [0.349, 0, 0], [0.349, 0, 0]],  # inside contact margin
            [[0.351, 0, 0], [0.351, 0, 0], [0.351, 0, 0]],  # outside contact margin
        ]
    )
    rotation, _ = np.linalg.qr(np.random.default_rng(29).normal(size=(3, 3)))
    for triangle in (False, True):
        for height in (0.0, 0.5):
            for transform, translation in ((np.eye(3), np.zeros(3)), (rotation, np.array([2, -3, 4]))):
                soft = triangles if triangle else triangles[:, :2]
                axis = np.broadcast_to([[0, 0, -height], [0, 0, height]], (len(soft), 2, 3))
                points = np.concatenate((soft, axis), axis=1) @ transform.T + translation
                check_cases(test, points, triangle, device)
                for dtype in (np.float32, np.float64):
                    quantized = points.astype(dtype)
                    _, coordinates, _ = evaluate(quantized, triangle, device, dtype)
                    coords = coordinates.numpy().astype(float)
                    p = quantized.astype(float)
                    if triangle:
                        x = np.sum(coords[:, :3, None] * p[:, :3], axis=1)
                        t = coords[:, 3, None]
                    else:
                        x = p[:, 0] + coords[:, :1] * (p[:, 1] - p[:, 0])
                        t = coords[:, 1:2]
                    y = p[:, -2] + t * (p[:, -1] - p[:, -2])
                    distance = np.linalg.norm(x - y, axis=1)
                    phi = distance - 0.3
                    n = 3 if triangle else 2
                    reference = np.array([np.sqrt(hull_distance_squared(row[:n], row[n:])) - 0.3 for row in p])
                    tolerance = 2e-6 if dtype == np.float32 else 2e-13
                    np.testing.assert_allclose(phi, reference, atol=tolerance)
                    np.testing.assert_array_equal(phi < 0.05, reference < 0.05)
                    unique = distance > 1e-5
                    normal = (x[unique] - y[unique]) / distance[unique, None]
                    projected = x[unique] - phi[unique, None] * normal
                    surface_distance = np.array(
                        [
                            np.sqrt(hull_distance_squared([point], row[-2:]))
                            for point, row in zip(projected, p[unique], strict=True)
                        ]
                    )
                    np.testing.assert_allclose(surface_distance, 0.3, atol=tolerance)


def test_closest_gradients(test, device):
    cases = [
        (False, [[0, 0, 0], [2, 0, 0], [0.7, -1, 1], [0.7, 1, 1]]),
        (True, [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0.3, 0.4, 1], [0.5, 0.6, 2]]),
        (True, [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 1, 1], [3, 2, 2]]),
        (True, [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0.3, 0.4, -1], [0.5, 0.6, 2]]),
    ]
    for triangle, points in cases:
        for dtype in (np.float32, np.float64):
            p = np.array([points], dtype=dtype)
            scalar = wp.float32 if dtype == np.float32 else wp.float64
            with wp.Tape() as tape:
                data, coords, d2 = evaluate(p, triangle, device, dtype, requires_grad=True)
                loss = wp.zeros(1, dtype=scalar, device=device, requires_grad=True)
                wp.launch(objective, 1, inputs=[coords, d2, loss], device=device)
            tape.backward(loss)
            analytical = data.grad.numpy()
            numerical = np.zeros_like(p)
            h = 2e-3 if dtype == np.float32 else 1e-5
            for index in np.ndindex(p.shape):
                values = []
                for sign in (-1, 1):
                    perturbed = p.copy()
                    perturbed[index] += sign * h
                    _, co, ds = evaluate(perturbed, triangle, device, dtype)
                    values.append(ds.numpy()[0] + np.dot(co.numpy()[0], 0.17 * np.arange(1, 5)))
                numerical[index] = (values[1] - values[0]) / (2 * h)
            np.testing.assert_allclose(analytical, numerical, atol=4e-4 if dtype == np.float32 else 1e-8, rtol=1e-3)


def test_closest_scale_gradients(test, device):
    # Parameters are scale invariant, so their gradients scale inversely with
    # length. Squared-distance gradients scale with length. This supplements
    # finite differences without subtracting tiny/huge objective values.
    for triangle, points in (
        (False, [[0, 0, 0], [2, 0, 0], [0.7, -1, 1], [0.7, 1, 1]]),
        (True, [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0.3, 0.4, 1], [0.5, 0.6, 2]]),
    ):
        for dtype in (np.float32, np.float64):
            scalar = wp.float32 if dtype == np.float32 else wp.float64
            for parameters in (False, True):
                reference = None
                for scale in (1.0, 1e-6, 1e-3, 1e3, 1e6):
                    p = np.array([points], dtype=dtype) * scale
                    with wp.Tape() as tape:
                        data, coords, d2 = evaluate(p, triangle, device, dtype, requires_grad=True)
                        loss = wp.zeros(1, dtype=scalar, device=device, requires_grad=True)
                        wp.launch(scaled_objective, 1, inputs=[coords, d2, loss, parameters], device=device)
                    tape.backward(loss)
                    gradient = data.grad.numpy() * (scale if parameters else 1.0 / scale)
                    test.assertTrue(np.isfinite(gradient).all())
                    if reference is None:
                        reference = gradient
                    else:
                        np.testing.assert_allclose(gradient, reference, atol=2e-6 if dtype == np.float32 else 1e-13)


class TestClosest(unittest.TestCase):
    pass


for fn in (
    test_closest_random,
    test_closest_exact,
    test_closest_features,
    test_closest_gradients,
    test_closest_scale_gradients,
    test_closest_near_parallel,
    test_closest_edge_piercing,
    test_closest_extreme_scales,
    test_closest_mixed_segment_lengths,
    test_closest_thin_triangles,
    test_closest_capsule_witnesses,
):
    add_function_test(TestClosest, fn.__name__, fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
