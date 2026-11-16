# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oriented bounding box fitting."""

import math
import unittest
import unittest.mock

import numpy as np

import warp as wp
import warp.geometry as geo
from warp.tests.geometry import utils as U
from warp.tests.unittest_utils import *

# The Warp module that holds the OBB kernels, which the graph capture tests load before capturing.
# ``warp.geometry`` itself is a different Warp module, holding unrelated kernels.
_OBB_MODULE = geo.oriented_bounding_box.__module__


def _host(result):
    """Read the length-1 device arrays returned by ``oriented_bounding_box`` to host."""
    transform, extents, measure = result
    return transform.numpy()[0], extents.numpy()[0], float(measure.numpy()[0])


def _rotation(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _rod(rng, extents=(2.0, 0.4, 0.15), num_points=400):
    """Build a strongly elongated box, rotated away from the axes.

    The corners are included so the minimal box of the sample is exactly the
    generating box, making the optimal measure known in closed form.
    """
    extents = np.asarray(extents, dtype=np.float64)
    corners = np.array(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij")).reshape(3, -1).T * extents
    interior = rng.uniform(-0.5, 0.5, (num_points, 3)) * extents
    points = np.concatenate([corners, interior], axis=0)
    return np.ascontiguousarray(points @ _rotation((0.3, -0.7, 0.5), 0.9).T, dtype=np.float32), extents


def _to_local(points_np, transform, extents):
    """Map world points into the box frame and return their signed overhang."""
    t = np.array([transform[i] for i in range(7)], dtype=np.float64)
    position, quat = t[:3], t[3:]
    qv, qw = quat[:3], quat[3]

    # Inverse rotation of (p - position) by the unit quaternion.
    d = np.asarray(points_np, dtype=np.float64) - position
    tmp = 2.0 * np.cross(-qv, d)
    local = d + qw * tmp + np.cross(-qv, tmp)

    return np.abs(local) - 0.5 * np.asarray(extents, dtype=np.float64)


def _aabb_measure(points_np, measure_type):
    dims = points_np.max(axis=0) - points_np.min(axis=0)
    if measure_type == geo.OBBMeasureType.VOLUME:
        return float(np.prod(dims))
    if measure_type == geo.OBBMeasureType.SURFACE_AREA:
        return float(2.0 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2]))
    return float(0.5 * np.sum(dims))


def _shapes(rng):
    return {
        "cube": U.unit_cube()[0],
        "icosphere": U.icosphere(2)[0],
        "rod": _rod(rng)[0],
        "cloud": rng.standard_normal((500, 3)).astype(np.float32),
    }


def test_obb_contains_all_points(test, device):
    """Return a box that encloses every input point, for each shape and objective."""
    # The defining property: the returned box must actually bound the input.
    rng = np.random.default_rng(59)
    for name, p_np in _shapes(rng).items():
        points = wp.array(p_np, dtype=wp.vec3, device=device)
        for measure_type in geo.OBBMeasureType:
            with test.subTest(shape=name, measure=measure_type.name):
                transform, extents, _ = _host(
                    geo.oriented_bounding_box(points, measure_type=measure_type, num_samples=256)
                )
                overhang = _to_local(p_np, transform, extents).max()
                scale = float(np.max(np.asarray(extents, dtype=np.float64)))
                test.assertLess(overhang, 1e-5 * max(scale, 1.0))


def test_obb_measure_matches_extents(test, device):
    """Return a measure that matches the volume, surface area or mean width of the returned extents."""
    rng = np.random.default_rng(61)
    points = wp.array(_shapes(rng)["cloud"], dtype=wp.vec3, device=device)

    _, extents, measure = _host(geo.oriented_bounding_box(points, num_samples=128))
    e = np.asarray(extents, dtype=np.float64)
    test.assertAlmostEqual(measure / float(np.prod(e)), 1.0, places=4)

    _, extents, measure = _host(
        geo.oriented_bounding_box(points, measure_type=geo.OBBMeasureType.SURFACE_AREA, num_samples=128)
    )
    e = np.asarray(extents, dtype=np.float64)
    expected = 2.0 * (e[0] * e[1] + e[1] * e[2] + e[0] * e[2])
    test.assertAlmostEqual(measure / expected, 1.0, places=4)

    _, extents, measure = _host(
        geo.oriented_bounding_box(points, measure_type=geo.OBBMeasureType.MEAN_WIDTH, num_samples=128)
    )
    e = np.asarray(extents, dtype=np.float64)
    test.assertAlmostEqual(measure / float(0.5 * np.sum(e)), 1.0, places=4)


def _skewed_cloud(seed):
    """Return a few randomly rotated, anisotropic points whose best boxes differ by objective."""
    rng = np.random.default_rng(seed)
    num_points = int(rng.integers(5, 12))
    cloud = rng.normal(size=(num_points, 3)) * np.array([3.0, 1.0, 0.3]) ** rng.uniform(0.0, 1.5)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    q[:, 0] *= np.linalg.det(q)
    return (cloud @ q.T).astype(np.float32)


def test_obb_mean_width_objective(test, device):
    """Minimize the mean width of the box, which picks a different box than minimizing the volume."""
    # On this cloud the volume-minimizing box has a larger mean width than the mean-width-minimizing
    # box, and the other way around for the volume, so each objective must win on its own measure.
    points = wp.array(_skewed_cloud(14), dtype=wp.vec3, device=device)
    Measure = geo.OBBMeasureType
    kwargs = {"num_samples": 256, "include_pca": False}

    _, volume_extents, _ = _host(geo.oriented_bounding_box(points, measure_type=Measure.VOLUME, **kwargs))
    _, width_extents, width_measure = _host(
        geo.oriented_bounding_box(points, measure_type=Measure.MEAN_WIDTH, **kwargs)
    )
    volume_extents = np.asarray(volume_extents, dtype=np.float64)
    width_extents = np.asarray(width_extents, dtype=np.float64)

    test.assertAlmostEqual(width_measure / float(0.5 * np.sum(width_extents)), 1.0, places=4)
    test.assertLess(0.5 * np.sum(width_extents), 0.98 * 0.5 * np.sum(volume_extents))
    test.assertLess(np.prod(volume_extents), np.prod(width_extents))


def test_obb_never_worse_than_aabb(test, device):
    """Return a box no looser than the axis-aligned box when the identity is a candidate."""
    # include_axis_aligned adds the identity rotation as a candidate, so the
    # result is guaranteed to be at least as tight as the AABB.
    rng = np.random.default_rng(67)
    for name, p_np in _shapes(rng).items():
        points = wp.array(p_np, dtype=wp.vec3, device=device)
        for measure_type in geo.OBBMeasureType:
            with test.subTest(shape=name, measure=measure_type.name):
                _, _, measure = _host(
                    geo.oriented_bounding_box(
                        points, measure_type=measure_type, num_samples=64, include_axis_aligned=True
                    )
                )
                test.assertLessEqual(measure, _aabb_measure(p_np, measure_type) * (1.0 + 1e-5))


def test_obb_recovers_rotated_box(test, device):
    """Recover the size of a rotated box from a dense spiral sample."""
    # With enough samples the fit should be close to the true minimal box.
    rng = np.random.default_rng(71)
    p_np, extents = _rod(rng)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    _, got_extents, measure = _host(geo.oriented_bounding_box(points, num_samples=4096))

    true_volume = float(np.prod(extents))
    test.assertGreaterEqual(measure, true_volume * (1.0 - 1e-4))
    test.assertLess(measure, true_volume * 2.0)

    # The longest side must be recovered accurately even if the two short sides
    # trade off against each other.
    test.assertAlmostEqual(max(np.asarray(got_extents, dtype=np.float64)), max(extents), delta=0.05)


# Keyword arguments that reduce the search to the best sampled orientation: one start, no refinement, no spin.
_SAMPLED_ONLY = {"num_refine_starts": 1, "num_refine_iterations": 0, "num_spin_samples": 0}


def test_obb_pca_helps_elongated_shapes(test, device):
    """Return a tighter box on an elongated shape when the PCA orientation is a candidate."""
    # On a strongly elongated shape the spiral's angular resolution is the limiting factor,
    # and the covariance eigenvectors are a much better guess. Refinement is disabled here
    # (``_SAMPLED_ONLY``) to isolate the initial search: with it on, the local search closes the
    # gap and PCA no longer helps measurably.
    rng = np.random.default_rng(73)
    p_np, extents = _rod(rng)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    _, _, without_pca = _host(geo.oriented_bounding_box(points, num_samples=32, include_pca=False, **_SAMPLED_ONLY))
    _, _, with_pca = _host(geo.oriented_bounding_box(points, num_samples=32, include_pca=True, **_SAMPLED_ONLY))

    test.assertLess(with_pca, without_pca)
    test.assertLess(with_pca, float(np.prod(extents)) * 2.0)


def test_obb_reproducible_without_pca(test, device):
    """Return bitwise identical results across repeated calls without PCA."""
    # The docstring promises a bitwise reproducible result with include_pca=False,
    # since the min/max reduction is exact regardless of atomic ordering.
    rng = np.random.default_rng(79)
    points = wp.array(_shapes(rng)["cloud"], dtype=wp.vec3, device=device)

    results = [_host(geo.oriented_bounding_box(points, num_samples=128, include_pca=False)) for _ in range(3)]
    for transform, extents, measure in results[1:]:
        test.assertEqual(measure, results[0][2])
        test.assertEqual(tuple(extents), tuple(results[0][1]))
        test.assertEqual(tuple(transform), tuple(results[0][0]))


def test_obb_single_point(test, device):
    """Return a zero-size box centered on the point for a single-point cloud."""
    points = wp.array(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    transform, extents, measure = _host(geo.oriented_bounding_box(points, num_samples=8))

    np.testing.assert_allclose(np.asarray(extents, dtype=np.float64), np.zeros(3), atol=1e-5)
    test.assertAlmostEqual(measure, 0.0, places=6)
    np.testing.assert_allclose(np.array([transform[0], transform[1], transform[2]]), [1.0, 2.0, 3.0], rtol=1e-5)


def test_obb_refinement_never_worse(test, device):
    """Return a box no looser with refinement than without it, for each shape and objective."""
    # Refinement is a local search that only commits improvements, so the refined box is
    # never worse than the unrefined one on the same input, for either objective.
    rng = np.random.default_rng(83)
    for name, p_np in _shapes(rng).items():
        points = wp.array(p_np, dtype=wp.vec3, device=device)
        for measure_type in geo.OBBMeasureType:
            with test.subTest(shape=name, measure=measure_type.name):
                _, _, base = _host(
                    geo.oriented_bounding_box(
                        points,
                        measure_type=measure_type,
                        num_samples=128,
                        num_refine_iterations=0,
                        num_spin_samples=0,
                    )
                )
                _, _, refined = _host(
                    geo.oriented_bounding_box(
                        points,
                        measure_type=measure_type,
                        num_samples=128,
                        num_refine_iterations=6,
                        num_spin_samples=0,
                    )
                )
                test.assertLessEqual(refined, base * (1.0 + 1e-5))


def test_obb_refinement_reaches_higher_sample_quality(test, device):
    """Refine a coarse spiral sample to a box close to the minimal one on an elongated shape."""
    # Coarse-to-fine refinement lets a small sample count reach the quality of a much larger
    # one on an elongated shape, where the spiral's angular spacing is the limiting factor.
    rng = np.random.default_rng(89)
    p_np, extents = _rod(rng)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    _, _, coarse = _host(geo.oriented_bounding_box(points, num_samples=64, **_SAMPLED_ONLY))
    _, _, refined = _host(
        geo.oriented_bounding_box(points, num_samples=64, num_refine_iterations=8, num_spin_samples=0)
    )

    test.assertLess(refined, coarse)
    test.assertLess(refined, float(np.prod(extents)) * 1.2)


def _rotated_prism(seed, num_corners=14):
    """Return the corners of an extruded random convex profile under a random rotation."""
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, num_corners))
    radii = rng.uniform(0.6, 1.0, num_corners)
    profile = np.stack([radii * np.cos(angles), 0.55 * radii * np.sin(angles)], axis=1)
    prism = np.concatenate([np.c_[profile, np.full(num_corners, -0.25)], np.c_[profile, np.full(num_corners, 0.25)]])
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    q[:, 0] *= np.linalg.det(q)
    return (prism @ q.T).astype(np.float32)


def test_obb_spin_rounds_settle_face_flush_box(test, device):
    """Spin about the box axes to reach a box that isotropic refinement alone leaves loose."""
    # An extruded profile has a face-flush optimum whose spin about the extrusion axis the
    # isotropic rounds converge on only slowly, so a small budget of them leaves the box loose.
    points = wp.array(_rotated_prism(80), dtype=wp.vec3, device=device)
    kwargs = {"num_samples": 256, "num_refine_starts": 2, "num_refine_iterations": 3, "num_refine_samples": 16}

    _, _, without_spin = _host(geo.oriented_bounding_box(points, num_spin_samples=0, **kwargs))
    _, _, with_spin = _host(geo.oriented_bounding_box(points, num_spin_samples=64, **kwargs))

    test.assertLess(with_spin, 0.9 * without_spin)


def test_obb_spin_rounds_never_worse(test, device):
    """Return a box no looser with spin rounds than without them."""
    rng = np.random.default_rng(131)
    for name, p_np in _shapes(rng).items():
        points = wp.array(p_np, dtype=wp.vec3, device=device)
        for measure_type in geo.OBBMeasureType:
            with test.subTest(shape=name, measure=measure_type.name):
                kwargs = {"measure_type": measure_type, "num_samples": 128, "num_refine_iterations": 2}
                _, _, without_spin = _host(geo.oriented_bounding_box(points, num_spin_samples=0, **kwargs))
                _, _, with_spin = _host(geo.oriented_bounding_box(points, num_spin_samples=32, **kwargs))
                test.assertLessEqual(with_spin, without_spin * (1.0 + 1e-5))


def test_obb_overflowing_measure_returns_valid_box(test, device):
    """Return a finite box when the box volume overflows float32."""
    # At these coordinates every candidate's volume overflows to infinity, so no candidate can
    # win on measure; the fit must still commit to one of them rather than leave its champion
    # uninitialized.
    rng = np.random.default_rng(139)
    points = wp.array((rng.uniform(-1.0, 1.0, (500, 3)) * 1e13).astype(np.float32), dtype=wp.vec3, device=device)

    transform, extents, _ = _host(geo.oriented_bounding_box(points, num_samples=64))

    test.assertTrue(np.all(np.isfinite(transform)))
    test.assertTrue(np.all(np.isfinite(extents)))


def test_obb_multiple_refine_starts_escape_wrong_basin(test, device):
    """Refine several orientations to reach a box that refining only the best sample misses."""
    # On this cloud the best sampled orientation lies in the basin of a worse local optimum, so
    # refining it alone cannot reach the tighter box that another basin offers.
    rng = np.random.default_rng(127)
    p_np = rng.normal(size=(24, 3)).astype(np.float32) * np.array([1.0, 0.8, 0.6], dtype=np.float32)
    points = wp.array(p_np, dtype=wp.vec3, device=device)
    kwargs = {
        "num_samples": 64,
        "include_axis_aligned": False,
        "include_pca": False,
        "num_refine_iterations": 8,
        "num_refine_samples": 16,
        "num_spin_samples": 0,
    }

    _, _, single = _host(geo.oriented_bounding_box(points, num_refine_starts=1, **kwargs))
    _, _, several = _host(geo.oriented_bounding_box(points, num_refine_starts=4, **kwargs))

    test.assertLess(several, 0.9 * single)


def test_obb_refine_starts_exceeding_candidates(test, device):
    """Repeat the best start when more refine starts are requested than there are candidate orientations."""
    # One spiral sample and the identity give two candidates for eight starts, so at least six
    # starts find no candidate left and must repeat the best one instead of staying uninitialized.
    # Uninitialized scratch memory may happen to hold harmless values, so every new float32 array
    # starts at -inf, which wins any comparison, and every new quaternion array starts at NaN:
    # a start left uninitialized would then win the final selection with a NaN rotation.
    p_np = U.unit_cube()[0]
    points = wp.array(p_np, dtype=wp.vec3, device=device)
    real_empty = wp.empty

    def poisoned_empty(*args, **kwargs):
        array = real_empty(*args, **kwargs)
        if array.dtype == wp.float32:
            array.fill_(-math.inf)
        elif array.dtype == wp.quat:
            array.fill_((math.nan,) * 4)
        return array

    with unittest.mock.patch.object(wp, "empty", side_effect=poisoned_empty):
        transform, extents, measure = _host(
            geo.oriented_bounding_box(
                points, num_samples=1, include_axis_aligned=True, include_pca=False, num_refine_starts=8
            )
        )

    test.assertTrue(np.all(np.isfinite(transform)))
    test.assertLess(_to_local(p_np, transform, extents).max(), 1e-5)
    test.assertAlmostEqual(measure, 1.0, places=4)


def _slab_with_outlier(rng, num_points=64):
    """Return a thin square slab whose second point is a far outlier along the slab's diagonal.

    A strided subsample with any stride above one visits only the points at multiples of the
    stride, so it never sees the second point. Without the outlier the slab's own axes give the
    tightest box; with it, a box turned 45 degrees about the slab normal is much tighter.
    """
    p_np = rng.uniform(-0.5, 0.5, (num_points, 3)) * np.array([1.0, 1.0, 0.02])
    p_np[0] = (-0.5, -0.5, 0.0)
    p_np[1] = (3.0, 3.0, 0.0)
    return p_np.astype(np.float32)


def test_obb_default_scores_every_point(test, device):
    """Score candidate orientations over every point unless the caller caps the search."""
    rng = np.random.default_rng(149)
    shapes = [_slab_with_outlier(rng), _slab_with_outlier(rng, num_points=80)]
    kwargs = {"num_samples": 64, "include_pca": False}

    for i, p_np in enumerate(shapes):
        with test.subTest(shape=i):
            points = wp.array(p_np, dtype=wp.vec3, device=device)
            default = _host(geo.oriented_bounding_box(points, **kwargs))
            full = _host(geo.oriented_bounding_box(points, max_search_points=len(p_np), **kwargs))
            subsampled = _host(geo.oriented_bounding_box(points, max_search_points=len(p_np) // 2, **kwargs))

            # The subsample misses the outlier, so it picks a much looser box; the default must not.
            test.assertGreater(subsampled[2], 1.5 * full[2])
            np.testing.assert_array_equal(default[0], full[0])
            np.testing.assert_array_equal(default[1], full[1])
            test.assertEqual(default[2], full[2])

    points, offsets = _pack(shapes, device)
    default = geo.oriented_bounding_boxes(points, offsets, **kwargs)
    full = geo.oriented_bounding_boxes(points, offsets, max_search_points=max(len(p) for p in shapes), **kwargs)
    for got, want in zip(default, full, strict=True):
        np.testing.assert_array_equal(got.numpy(), want.numpy())


def test_obb_subsample_stays_exact(test, device):
    """Return a box that encloses every point when the search scores a subsample."""
    # The winner's box is re-measured over the full cloud, so scoring orientations on a
    # strided subsample still returns a box that encloses every point.
    rng = np.random.default_rng(97)
    p_np, extents = _rod(rng, num_points=20000)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    transform, got_extents, measure = _host(geo.oriented_bounding_box(points, num_samples=256, max_search_points=2000))
    overhang = _to_local(p_np, transform, got_extents).max()
    test.assertLess(overhang, 1e-4 * max(float(np.max(np.asarray(got_extents, dtype=np.float64))), 1.0))
    test.assertLess(measure, float(np.prod(extents)) * 2.0)


def test_obb_subsample_boundary(test, device):
    """Enclose every point when the point count barely exceeds the search cap."""
    # Point counts just above the cap (stride > 1, uneven division): must not read out of
    # bounds and must still enclose every point.
    rng = np.random.default_rng(101)
    for n, cap in ((201, 100), (1000, 999), (1000, 7)):
        with test.subTest(n=n, cap=cap):
            p_np = rng.standard_normal((n, 3)).astype(np.float32)
            points = wp.array(p_np, dtype=wp.vec3, device=device)
            transform, extents, _ = _host(geo.oriented_bounding_box(points, num_samples=32, max_search_points=cap))
            overhang = _to_local(p_np, transform, extents).max()
            test.assertLess(overhang, 1e-4 * max(float(np.max(np.asarray(extents, dtype=np.float64))), 1.0))


def test_obb_refine_tiny_cloud(test, device):
    """Refine a cloud with fewer points than a refinement batch without failing."""
    # Fewer points than a refine batch or a chunk count: must not crash and must enclose all.
    rng = np.random.default_rng(103)
    p_np = rng.standard_normal((3, 3)).astype(np.float32)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    transform, extents, _ = _host(
        geo.oriented_bounding_box(points, num_samples=16, num_refine_iterations=4, num_refine_samples=8)
    )
    overhang = _to_local(p_np, transform, extents).max()
    test.assertLess(overhang, 1e-4 * max(float(np.max(np.asarray(extents, dtype=np.float64))), 1.0))


def test_obb_invalid_arguments(test, device):
    """Reject invalid search parameters and an empty point set with a ValueError."""
    points = wp.array(U.unit_cube()[0], dtype=wp.vec3, device=device)

    with test.assertRaisesRegex(ValueError, "num_samples"):
        geo.oriented_bounding_box(points, num_samples=0)
    with test.assertRaisesRegex(ValueError, "num_samples"):
        geo.oriented_bounding_box(points, num_samples=-5)
    with test.assertRaisesRegex(ValueError, "num_refine_starts"):
        geo.oriented_bounding_box(points, num_refine_starts=0)
    with test.assertRaisesRegex(ValueError, "num_refine_iterations"):
        geo.oriented_bounding_box(points, num_refine_iterations=-1)
    with test.assertRaisesRegex(ValueError, "num_refine_samples"):
        geo.oriented_bounding_box(points, num_refine_samples=0)
    with test.assertRaisesRegex(ValueError, "num_spin_samples"):
        geo.oriented_bounding_box(points, num_spin_samples=-1)
    with test.assertRaisesRegex(ValueError, "max_search_points"):
        geo.oriented_bounding_box(points, max_search_points=0)

    empty = wp.zeros(0, dtype=wp.vec3, device=device)
    with test.assertRaisesRegex(ValueError, "at least one point"):
        geo.oriented_bounding_box(empty, num_samples=8)


def test_obb_rejects_non_integer_parameters(test, device):
    """Reject non-integer search parameters with a ValueError."""
    points = wp.array(U.unit_cube()[0], dtype=wp.vec3, device=device)
    with test.assertRaisesRegex(ValueError, "num_samples.*integer"):
        geo.oriented_bounding_box(points, num_samples=16.0)
    with test.assertRaisesRegex(ValueError, "num_refine_iterations.*integer"):
        geo.oriented_bounding_box(points, num_refine_iterations=1.5)


def test_obb_rejects_mismatched_device(test, device):
    """Reject a ``device`` that differs from the device of the points."""
    other = "cpu" if device.is_cuda else "cuda:0"
    points = wp.array(U.unit_cube()[0], dtype=wp.vec3, device=device)
    with test.assertRaisesRegex(ValueError, "must match the device"):
        geo.oriented_bounding_box(points, num_samples=16, device=other)


def test_obb_graph_capturable(test, device):
    """Capture a fit in a CUDA graph and replay it to the same box."""
    # The whole search -- including the argmin -- runs on the device, so it can be
    # captured into a CUDA graph and replayed without a host synchronization.
    points = wp.array(U.unit_cube()[0], dtype=wp.vec3, device=device)
    wp.load_module(_OBB_MODULE, device=device)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        result = geo.oriented_bounding_box(points, num_samples=16)
    wp.capture_launch(capture.graph)

    # The unit cube is already axis-aligned, so its minimal box is the cube itself.
    _, extents, measure = _host(result)
    np.testing.assert_allclose(np.asarray(extents, dtype=np.float64), np.ones(3), atol=1e-5)
    test.assertAlmostEqual(measure, 1.0, places=5)


def _pack(shapes, device, leading=0):
    """Concatenate point sets and return ``(points, offsets)`` device arrays.

    ``leading`` extra points are placed before the first shape so that ``offsets[0] > 0``.
    """
    pieces = [np.zeros((leading, 3), dtype=np.float32), *shapes]
    sizes = [len(p) for p in shapes]
    offsets = leading + np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    points = wp.array(np.concatenate(pieces, axis=0), dtype=wp.vec3, device=device)
    return points, wp.array(offsets, dtype=wp.int32, device=device)


def _batch_shapes(rng):
    # Sizes straddle ``max_search_points=100`` below so some shapes are subsampled and some are not.
    return [
        U.unit_cube()[0],
        _rod(rng, num_points=30)[0],
        rng.standard_normal((100, 3)).astype(np.float32),
        _rod(rng, extents=(1.0, 0.7, 0.2), num_points=700)[0],
        rng.standard_normal((1, 3)).astype(np.float32),
        U.icosphere(2)[0],
    ]


_SEARCH = {
    "num_samples": 96,
    "max_search_points": 100,
    "num_refine_starts": 3,
    "num_refine_iterations": 2,
    "num_refine_samples": 32,
    "num_spin_samples": 8,
}


def test_obbs_match_single_shape_fits(test, device):
    """Return the same box for each shape of a batch as a separate fit, bit for bit."""
    # Batching must be a pure performance change: each shape gets the same box as a standalone call.
    # PCA is off because its covariance atomics make even repeated identical calls differ in the last bits.
    shapes = _batch_shapes(np.random.default_rng(101))
    points, offsets = _pack(shapes, device)
    for measure_type in geo.OBBMeasureType:
        with test.subTest(measure=measure_type.name):
            transform, extents, measure = geo.oriented_bounding_boxes(
                points, offsets, measure_type=measure_type, include_pca=False, **_SEARCH
            )
            test.assertEqual(transform.shape, (len(shapes),))
            for i, p_np in enumerate(shapes):
                single = _host(
                    geo.oriented_bounding_box(
                        wp.array(p_np, dtype=wp.vec3, device=device),
                        measure_type=measure_type,
                        include_pca=False,
                        **_SEARCH,
                    )
                )
                np.testing.assert_array_equal(transform.numpy()[i], single[0], err_msg=f"shape {i}")
                np.testing.assert_array_equal(extents.numpy()[i], single[1], err_msg=f"shape {i}")
                test.assertEqual(float(measure.numpy()[i]), single[2], msg=f"shape {i}")


def test_obbs_match_single_shape_fits_with_pca(test, device):
    """Return close to the measure of a separate fit for each shape of a batch with PCA."""
    # With PCA the last bits are not reproducible, so compare the achieved measure loosely.
    shapes = _batch_shapes(np.random.default_rng(103))
    points, offsets = _pack(shapes, device)
    _, _, measure = geo.oriented_bounding_boxes(points, offsets, include_pca=True, **_SEARCH)
    for i, p_np in enumerate(shapes):
        _, _, single = _host(
            geo.oriented_bounding_box(wp.array(p_np, dtype=wp.vec3, device=device), include_pca=True, **_SEARCH)
        )
        test.assertAlmostEqual(float(measure.numpy()[i]), single, delta=1e-3 * max(single, 1e-3), msg=f"shape {i}")


def test_obbs_contain_all_points(test, device):
    """Return a box per shape that encloses every point of that shape."""
    shapes = _batch_shapes(np.random.default_rng(107))
    points, offsets = _pack(shapes, device)
    transform, extents, _ = geo.oriented_bounding_boxes(points, offsets, **_SEARCH)
    for i, p_np in enumerate(shapes):
        overhang = _to_local(p_np, transform.numpy()[i], extents.numpy()[i]).max()
        scale = float(np.max(np.asarray(extents.numpy()[i], dtype=np.float64)))
        test.assertLess(overhang, 1e-5 * max(scale, 1.0), msg=f"shape {i}")


def test_obbs_shapes_are_independent(test, device):
    """Return a shape's box independent of the other shapes and its position in the batch."""
    # A shape's box must not depend on which other shapes share the batch or where it sits in it.
    rng = np.random.default_rng(109)
    shapes = _batch_shapes(rng)
    order = [3, 0, 5, 2, 1, 4]
    points_a, offsets_a = _pack(shapes, device)
    points_b, offsets_b = _pack([shapes[j] for j in order], device, leading=17)

    kwargs = {"include_pca": False, **_SEARCH}
    ta, ea, ma = geo.oriented_bounding_boxes(points_a, offsets_a, **kwargs)
    tb, eb, mb = geo.oriented_bounding_boxes(points_b, offsets_b, **kwargs)
    for position, j in enumerate(order):
        np.testing.assert_array_equal(ta.numpy()[j], tb.numpy()[position], err_msg=f"shape {j}")
        np.testing.assert_array_equal(ea.numpy()[j], eb.numpy()[position], err_msg=f"shape {j}")
        test.assertEqual(float(ma.numpy()[j]), float(mb.numpy()[position]), msg=f"shape {j}")


def test_obbs_many_small_shapes(test, device):
    """Fit thousands of small shapes in one batch to the same boxes as separate fits."""
    # Enough shapes that the launch budget forces fewer point chunks per orientation than a single fit uses.
    rng = np.random.default_rng(113)
    num_shapes = 3000
    shapes = [rng.standard_normal((int(rng.integers(1, 12)), 3)).astype(np.float32) for _ in range(num_shapes)]
    points, offsets = _pack(shapes, device)
    kwargs = {"num_samples": 256, "include_pca": False, "num_refine_iterations": 1, "num_refine_samples": 16}
    transform, _, measure = geo.oriented_bounding_boxes(points, offsets, max_points_per_shape=11, **kwargs)
    for i in rng.choice(num_shapes, size=8, replace=False):
        single = _host(geo.oriented_bounding_box(wp.array(shapes[i], dtype=wp.vec3, device=device), **kwargs))
        np.testing.assert_array_equal(transform.numpy()[i], single[0], err_msg=f"shape {i}")
        test.assertEqual(float(measure.numpy()[i]), single[2], msg=f"shape {i}")


def test_obbs_max_points_per_shape_does_not_change_result(test, device):
    """Return the same boxes whether or not a points-per-shape bound is given."""
    shapes = _batch_shapes(np.random.default_rng(127))
    points, offsets = _pack(shapes, device)
    kwargs = {"include_pca": False, **_SEARCH}
    default = geo.oriented_bounding_boxes(points, offsets, **kwargs)
    hinted = geo.oriented_bounding_boxes(points, offsets, max_points_per_shape=max(len(p) for p in shapes), **kwargs)
    for a, b in zip(default, hinted, strict=True):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_obbs_invalid_arguments(test, device):
    """Reject invalid search parameters and malformed offsets."""
    cube = U.unit_cube()[0]
    points, offsets = _pack([cube, cube], device)

    with test.assertRaisesRegex(ValueError, "num_samples"):
        geo.oriented_bounding_boxes(points, offsets, num_samples=0)
    with test.assertRaisesRegex(ValueError, "max_points_per_shape"):
        geo.oriented_bounding_boxes(points, offsets, max_points_per_shape=0)
    with test.assertRaisesRegex(TypeError, "int32"):
        geo.oriented_bounding_boxes(points, wp.array([0, 8, 16], dtype=wp.int64, device=device))
    with test.assertRaisesRegex(ValueError, "at least two offsets"):
        geo.oriented_bounding_boxes(points, wp.array([0], dtype=wp.int32, device=device))

    def bad(values):
        return wp.array(values, dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "point set 1 is empty"):
        geo.oriented_bounding_boxes(points, bad([0, 8, 8, 16]))
    with test.assertRaisesRegex(ValueError, "non-decreasing"):
        geo.oriented_bounding_boxes(points, bad([0, 12, 8, 16]))
    with test.assertRaisesRegex(ValueError, "within `points`"):
        geo.oriented_bounding_boxes(points, bad([0, 8, 17]))
    with test.assertRaisesRegex(ValueError, "within `points`"):
        geo.oriented_bounding_boxes(points, bad([-1, 8, 16]))


def test_obbs_rejects_offsets_on_another_device(test, device):
    """Reject offsets that live on a different device than the points."""
    cube = U.unit_cube()[0]
    points, _ = _pack([cube, cube], device)
    for validate in (True, False):
        with test.assertRaisesRegex(ValueError, "`offsets` is on device"):
            geo.oriented_bounding_boxes(points, wp.array([0, 8, 16], dtype=wp.int32, device="cpu"), validate=validate)


def test_obbs_empty_shape_without_validation_is_not_finite(test, device):
    """Return a non-finite box for an empty shape when validation is skipped."""
    # Skipping validation must not turn an empty point set into a plausible-looking box.
    cube = U.unit_cube()[0]
    points, offsets = _pack([cube, cube[:0], cube], device)
    result = geo.oriented_bounding_boxes(points, offsets, validate=False, **_SEARCH)[1].numpy()
    test.assertFalse(np.all(np.isfinite(result[1])))
    np.testing.assert_allclose(result[[0, 2]], np.ones((2, 3)), atol=1e-5)


def test_obbs_launch_count_does_not_scale_with_shapes(test, device):
    """Launch the same number of kernels for a batch regardless of how many shapes it holds."""
    # The point of batching: the number of kernel launches is independent of the number of shapes,
    # whereas fitting shapes one at a time launches a full search per shape. A launch count is a
    # deterministic proxy for the speedup, which a timing assertion could not give reliably.
    cube = U.unit_cube()[0]
    kwargs = {"num_samples": 8, "num_refine_iterations": 2, "num_refine_samples": 8, "validate": False}

    def count_launches(fit):
        real_launch = wp.launch
        with unittest.mock.patch.object(wp, "launch", side_effect=real_launch) as launch:
            fit()
        return launch.call_count

    counts = {}
    for num_shapes in (2, 64):
        points, offsets = _pack([cube] * num_shapes, device)
        counts[num_shapes] = count_launches(
            lambda points=points, offsets=offsets: geo.oriented_bounding_boxes(points, offsets, **kwargs)
        )
    test.assertEqual(counts[2], counts[64])

    single_points = wp.array(cube, dtype=wp.vec3, device=device)
    single_count = count_launches(
        lambda: geo.oriented_bounding_box(single_points, **{k: v for k, v in kwargs.items() if k != "validate"})
    )
    # A batch costs at most one fit, while 64 single fits cost 64 of them.
    test.assertLessEqual(counts[64], single_count)


def test_obbs_graph_capturable(test, device):
    """Capture a batched fit in a CUDA graph and replay it to the same boxes."""
    cube = U.unit_cube()[0]
    points, offsets = _pack([cube, cube + 3.0], device)
    wp.load_module(_OBB_MODULE, device=device)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        result = geo.oriented_bounding_boxes(points, offsets, num_samples=16, validate=False)
    wp.capture_launch(capture.graph)

    np.testing.assert_allclose(result[1].numpy(), np.ones((2, 3)), atol=1e-5)
    np.testing.assert_allclose(result[2].numpy(), np.ones(2), atol=1e-5)


def test_obbs_graph_capturable_with_subsampling(test, device):
    """Capture a batched fit that subsamples its search and replay it to the uncaptured result."""
    # The subsample's size bookkeeping (a scan over per-shape counts) must also be capturable.
    rng = np.random.default_rng(131)
    shapes = [_rod(rng, num_points=200)[0], _rod(rng, num_points=50)[0]]
    points, offsets = _pack(shapes, device)
    kwargs = {
        "num_samples": 32,
        "max_search_points": 20,
        "include_pca": False,
        "num_refine_iterations": 1,
        "num_refine_samples": 8,
    }
    wp.load_module(_OBB_MODULE, device=device)
    expected = geo.oriented_bounding_boxes(points, offsets, **kwargs)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        result = geo.oriented_bounding_boxes(points, offsets, validate=False, **kwargs)
    wp.capture_launch(capture.graph)

    for got, want in zip(result, expected, strict=True):
        np.testing.assert_array_equal(got.numpy(), want.numpy())


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()
# Graph capture allocates scratch arrays, which needs a CUDA memory pool.
capture_devices = get_selected_cuda_test_devices_with_mempool()


class TestGeometryOBB(unittest.TestCase):
    pass


add_function_test(TestGeometryOBB, "test_obb_contains_all_points", test_obb_contains_all_points, devices=devices)
add_function_test(
    TestGeometryOBB, "test_obb_measure_matches_extents", test_obb_measure_matches_extents, devices=devices
)
add_function_test(TestGeometryOBB, "test_obb_mean_width_objective", test_obb_mean_width_objective, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_never_worse_than_aabb", test_obb_never_worse_than_aabb, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_recovers_rotated_box", test_obb_recovers_rotated_box, devices=devices)
add_function_test(
    TestGeometryOBB, "test_obb_pca_helps_elongated_shapes", test_obb_pca_helps_elongated_shapes, devices=devices
)
add_function_test(
    TestGeometryOBB, "test_obb_reproducible_without_pca", test_obb_reproducible_without_pca, devices=devices
)
add_function_test(TestGeometryOBB, "test_obb_single_point", test_obb_single_point, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_refinement_never_worse", test_obb_refinement_never_worse, devices=devices)
add_function_test(
    TestGeometryOBB,
    "test_obb_refinement_reaches_higher_sample_quality",
    test_obb_refinement_reaches_higher_sample_quality,
    devices=devices,
)
add_function_test(
    TestGeometryOBB, "test_obb_default_scores_every_point", test_obb_default_scores_every_point, devices=devices
)
add_function_test(TestGeometryOBB, "test_obb_subsample_stays_exact", test_obb_subsample_stays_exact, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_subsample_boundary", test_obb_subsample_boundary, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_refine_tiny_cloud", test_obb_refine_tiny_cloud, devices=devices)
add_function_test(
    TestGeometryOBB,
    "test_obb_spin_rounds_settle_face_flush_box",
    test_obb_spin_rounds_settle_face_flush_box,
    devices=devices,
)
add_function_test(
    TestGeometryOBB, "test_obb_spin_rounds_never_worse", test_obb_spin_rounds_never_worse, devices=devices
)
add_function_test(
    TestGeometryOBB,
    "test_obb_overflowing_measure_returns_valid_box",
    test_obb_overflowing_measure_returns_valid_box,
    devices=devices,
)
add_function_test(
    TestGeometryOBB,
    "test_obb_multiple_refine_starts_escape_wrong_basin",
    test_obb_multiple_refine_starts_escape_wrong_basin,
    devices=devices,
)
add_function_test(
    TestGeometryOBB,
    "test_obb_refine_starts_exceeding_candidates",
    test_obb_refine_starts_exceeding_candidates,
    devices=devices,
)
add_function_test(TestGeometryOBB, "test_obb_invalid_arguments", test_obb_invalid_arguments, devices=devices)
add_function_test(
    TestGeometryOBB, "test_obb_rejects_non_integer_parameters", test_obb_rejects_non_integer_parameters, devices=devices
)
add_function_test(
    TestGeometryOBB, "test_obb_rejects_mismatched_device", test_obb_rejects_mismatched_device, devices=cuda_devices
)
add_function_test(TestGeometryOBB, "test_obb_graph_capturable", test_obb_graph_capturable, devices=capture_devices)

add_function_test(
    TestGeometryOBB, "test_obbs_match_single_shape_fits", test_obbs_match_single_shape_fits, devices=devices
)
add_function_test(
    TestGeometryOBB,
    "test_obbs_match_single_shape_fits_with_pca",
    test_obbs_match_single_shape_fits_with_pca,
    devices=devices,
)
add_function_test(TestGeometryOBB, "test_obbs_contain_all_points", test_obbs_contain_all_points, devices=devices)
add_function_test(
    TestGeometryOBB, "test_obbs_shapes_are_independent", test_obbs_shapes_are_independent, devices=devices
)
add_function_test(TestGeometryOBB, "test_obbs_many_small_shapes", test_obbs_many_small_shapes, devices=devices)
add_function_test(
    TestGeometryOBB,
    "test_obbs_max_points_per_shape_does_not_change_result",
    test_obbs_max_points_per_shape_does_not_change_result,
    devices=devices,
)
add_function_test(TestGeometryOBB, "test_obbs_invalid_arguments", test_obbs_invalid_arguments, devices=devices)
add_function_test(
    TestGeometryOBB,
    "test_obbs_rejects_offsets_on_another_device",
    test_obbs_rejects_offsets_on_another_device,
    devices=cuda_devices,
)
add_function_test(
    TestGeometryOBB,
    "test_obbs_empty_shape_without_validation_is_not_finite",
    test_obbs_empty_shape_without_validation_is_not_finite,
    devices=devices,
)
add_function_test(
    TestGeometryOBB,
    "test_obbs_launch_count_does_not_scale_with_shapes",
    test_obbs_launch_count_does_not_scale_with_shapes,
    devices=devices,
)
add_function_test(TestGeometryOBB, "test_obbs_graph_capturable", test_obbs_graph_capturable, devices=capture_devices)
add_function_test(
    TestGeometryOBB,
    "test_obbs_graph_capturable_with_subsampling",
    test_obbs_graph_capturable_with_subsampling,
    devices=capture_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
