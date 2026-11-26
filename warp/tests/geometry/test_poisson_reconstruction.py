# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from functools import lru_cache
from types import SimpleNamespace
from unittest import mock

import numpy as np

import warp as wp
import warp.geometry as geometry
from warp._src.geometry.poisson import field as field_module
from warp._src.geometry.poisson import meshing, meshing_bounds
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

wp.set_module_options({"fuse_fp": False})


def oriented_points(device, dtype=wp.vec3, *, requires_grad=False):
    """Generate clustered points away from adaptive assignment boundaries."""
    rng = np.random.default_rng(71)
    centers = np.array(
        [[0.1, 0.2, 0.3], [0.8, 1.5, 0.2], [-0.7, -0.6, -0.2], [-0.3, 0.7, 0.7], [0.3, -1.7, 0], [0.5, 0.8, -0.8]]
    )
    points = np.repeat(centers, 3, axis=0) + rng.normal(size=(18, 3)) * 0.01
    normals = rng.normal(size=points.shape)
    return (
        wp.array(points, dtype=dtype, device=device, requires_grad=requires_grad),
        wp.array(normals, dtype=dtype, device=device, requires_grad=requires_grad),
    )


def sphere_points(device, count=256):
    """Generate an oriented unit sphere without external assets."""
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    angle = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(1.0 - z * z)
    points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    return wp.array(points, dtype=wp.vec3, device=device), wp.array(points, dtype=wp.vec3, device=device)


@lru_cache(None)
def query_kernel(dtype):
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    field_type = geometry.PoissonField64 if dtype == wp.float64 else geometry.PoissonField

    @wp.func
    def composed_query(field: field_type, point: vector):
        return geometry.poisson_evaluate(field, point)

    @wp.kernel
    def evaluate_kernel(
        field: field_type,
        points: wp.array[vector],
        values: wp.array[dtype],
        gradients: wp.array[vector],
        valid: wp.array[int],
    ):
        index = wp.tid()
        value, gradient, inside = composed_query(field, points[index])
        values[index] = value
        gradients[index] = gradient
        valid[index] = inside

    return evaluate_kernel


def query_device(handle, queries, *, requires_grad=False):
    dtype = handle.coefficients.dtype
    values = wp.empty(len(queries), dtype=dtype, device=queries.device, requires_grad=requires_grad)
    gradients = wp.empty_like(queries, requires_grad=requires_grad)
    valid = wp.empty(len(queries), dtype=int, device=queries.device)
    wp.launch(
        query_kernel(dtype),
        len(queries),
        inputs=[handle, queries],
        outputs=[values, gradients, valid],
        device=queries.device,
    )
    return values, gradients, valid


def adaptive_field(device, dtype):
    """Construct a small field with a refined corner and known coefficients."""
    keys = []
    for depth in range(3):
        for x in range((1 << depth) + 1):
            for y in range((1 << depth) + 1):
                for z in range((1 << depth) + 1):
                    if depth < 2 or (x <= 2 and y <= 2 and z <= 2):
                        keys.append((depth << 48) | (x << 32) | (y << 16) | z)
    keys = wp.array(keys, dtype=wp.int64, device=device)
    coefficients = wp.array(
        np.random.default_rng(271).normal(size=len(keys)), dtype=dtype, device=device, requires_grad=True
    )
    cumulative = field_module.cumulative_coefficients(keys, coefficients, 2)
    cumulative.requires_grad = True
    return field_module.LevelField(
        keys, coefficients, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 2, 0.0, cumulative
    )


def test_device_queries(test, device, dtype):
    """Match fast batched evaluation through an ordinary composed device function."""
    field = adaptive_field(device, dtype)
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    points = np.concatenate(
        (
            np.random.default_rng(2026).uniform(0, 1, (17, 3)),
            [
                [0, 0.3, 0.4],
                [1, 0.3, 0.4],
                [0.5, 0.3, 0.4],
                [0.25, 0.3, 0.4],
                [0, 0, 0],
                [1, 1, 1],
                [1.01, 0.3, 0.4],
                [np.nan, 0.3, 0.4],
                [np.inf, 0.3, 0.4],
                [-np.inf, 0.3, 0.4],
            ],
        )
    )
    queries = wp.array(points, dtype=vector, device=device)
    handle = field.device_field()
    actual = query_device(handle, queries)
    with mock.patch.object(
        field_module, "reference_evaluation_kernel", wraps=field_module.reference_evaluation_kernel
    ) as factory:
        expected = field.evaluate(queries)
    factory.assert_called_once_with(dtype)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    np.testing.assert_array_equal(actual[2].numpy()[-4:], 0)
    test.assertTrue(np.isnan(actual[0].numpy()[-4:]).all())
    test.assertEqual(handle.coefficients.ptr, field.coefficients.ptr)
    test.assertEqual(handle.cumulative.ptr, field.cumulative.ptr)
    test.assertEqual(len(query_device(handle, wp.empty(0, dtype=vector, device=device))[0]), 0)


def test_device_query_gradient(test, device, dtype):
    """Differentiate a mixed value and spatial-gradient loss through a device query."""
    field = adaptive_field(device, dtype)
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    queries = wp.array([[0.13, 0.21, 0.17], [0.61, 0.71, 0.81]], dtype=vector, device=device, requires_grad=True)
    value_seed = np.array([0.7, -0.2])
    gradient_seed = np.array([[0.03, -0.02, 0.01], [-0.04, 0.07, -0.02]])
    with wp.Tape() as tape:
        values, gradients, _ = query_device(field.device_field(), queries, requires_grad=True)
    tape.backward(
        grads={
            values: wp.array(value_seed, dtype=dtype, device=device),
            gradients: wp.array(gradient_seed, dtype=vector, device=device),
        }
    )
    rng = np.random.default_rng(537)
    for array in (queries, field.coefficients, field.cumulative):
        original, adjoint = array.numpy().copy(), array.grad.numpy().copy()
        direction = rng.normal(size=original.shape).astype(original.dtype)
        epsilon = 1e-4 if dtype == wp.float32 else 1e-6
        losses = []
        for sign in (1, -1):
            array.assign(original + sign * epsilon * direction)
            value, spatial, _ = field.evaluate(queries)
            losses.append(
                value.numpy().astype(float) @ value_seed + np.sum(spatial.numpy().astype(float) * gradient_seed)
            )
        array.assign(original)
        np.testing.assert_allclose(
            np.sum(adjoint.astype(float) * direction),
            (losses[0] - losses[1]) / (2 * epsilon),
            rtol=5e-3 if dtype == wp.float32 else 1e-7,
            atol=2e-3 if dtype == wp.float32 else 1e-8,
        )


def test_reconstruction(test, device, dtype, weight):
    """Preserve the forward field when reconstruction derivatives are enabled."""
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    points, normals = oriented_points(device, vector, requires_grad=True)
    options = {"depth": 3, "full_depth": 1, "iterations": 2, "samples_per_node": 0.05, "point_weight": weight}
    result = geometry.poisson_reconstruct(points, normals, **options)
    tape = wp.Tape()
    tracked = geometry.poisson_reconstruct(points, normals, tape=tape, **options)
    queries = wp.array(
        [[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.4, 0.7, 0.3], [0.05, 0.3, -0.2]], dtype=vector, device=device
    )
    expected = result.evaluate(queries, centered=True)
    with tape:
        actual = tracked.evaluate(queries, centered=True)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    test.assertEqual(result.iso_value, tracked.iso_value)
    test.assertEqual(result.iso_value_array.numpy()[0], result.iso_value)
    test.assertFalse(result.iso_value_array.requires_grad)
    test.assertTrue(tracked.iso_value_array.requires_grad)
    np.testing.assert_array_equal(actual[2].numpy(), 1)
    raw = query_device(result.device_field(), queries)
    np.testing.assert_array_equal(raw[0].numpy() - result.iso_value, expected[0].numpy())
    np.testing.assert_array_equal(raw[1].numpy(), expected[1].numpy())


def test_reconstruction_gradient(test, device, dtype, weight):
    """Check point and normal derivatives of the finite multilevel reconstruction."""
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    points, normals = oriented_points(device, vector, requires_grad=True)
    options = {
        "depth": 3,
        "full_depth": 1 if weight else 0,
        "iterations": 2,
        "samples_per_node": 0.05,
        "point_weight": weight,
    }
    tape = wp.Tape()
    result = geometry.poisson_reconstruct(points, normals, tape=tape, **options)
    queries = wp.array(
        [[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.4, 0.7, 0.3], [0.05, 0.3, -0.2]], dtype=vector, device=device
    )
    with tape:
        value, _, _ = result.evaluate(queries, centered=True)
    seed = np.array([0.7, -0.5, 0.9, 0.3])
    tape.backward(grads={value: wp.array(seed, dtype=dtype, device=device)})
    rng = np.random.default_rng(283)
    original_keys = [level.keys.numpy() for level in result._result.diagnostics["tree"].basis]
    for array in (points, normals):
        original, adjoint = array.numpy().copy(), array.grad.numpy().copy()
        direction = rng.normal(size=original.shape).astype(original.dtype)
        epsilon = 1e-4 if dtype == wp.float32 else 1e-6
        losses = []
        for sign in (1, -1):
            array.assign(original + sign * epsilon * direction)
            candidate = geometry.poisson_reconstruct(points, normals, **options)
            for keys, level in zip(original_keys, candidate._result.diagnostics["tree"].basis, strict=True):
                np.testing.assert_array_equal(keys, level.keys.numpy())
            values, _, _ = candidate.evaluate(queries, centered=True)
            losses.append(values.numpy().astype(float) @ seed)
        array.assign(original)
        test.assertTrue(np.isfinite(adjoint).all())
        np.testing.assert_allclose(
            np.sum(adjoint.astype(float) * direction),
            (losses[0] - losses[1]) / (2 * epsilon),
            rtol=2e-2 if dtype == wp.float32 else 3e-5,
            atol=2e-3 if dtype == wp.float32 else 2e-7,
        )


def test_validation(test, device):
    """Reject unsupported inputs and modes before launching reconstruction work."""
    points, normals = oriented_points(device)
    with test.assertRaises(TypeError):
        geometry.poisson_reconstruct(points.numpy(), normals)
    with test.assertRaises(TypeError):
        geometry.poisson_reconstruct(points, normals, exact_interpolation=True)
    for options, error in (
        ({"depth": True}, TypeError),
        ({"iterations": 1.5}, TypeError),
        ({"depth": 0}, ValueError),
        ({"full_depth": 7}, ValueError),
        ({"depth": 12, "full_depth": 1}, ValueError),
        ({"scale": 1.0}, ValueError),
        ({"point_weight": -1}, ValueError),
        ({"cg_accuracy": float("nan")}, ValueError),
        ({"tape": 1}, TypeError),
    ):
        with test.subTest(options=options), test.assertRaises(error):
            geometry.poisson_reconstruct(points, normals, **options)
    with test.assertRaises(ValueError):
        geometry.poisson_reconstruct(points[:2], normals)
    with test.assertRaises(ValueError):
        geometry.poisson_reconstruct(points[::2], normals[::2])
    with test.assertRaises(TypeError):
        geometry.poisson_reconstruct(points, wp.array(normals.numpy(), dtype=wp.vec3d, device=device))
    with wp.Tape(), test.assertRaisesRegex(ValueError, "outside"):
        geometry.poisson_reconstruct(points, normals)
    with wp.ScopedCapture(device=device), test.assertRaisesRegex(ValueError, "capture"):
        geometry.poisson_reconstruct(points, normals)
    with test.assertRaises(ValueError):
        geometry.poisson_reconstruct(
            wp.empty(0, dtype=wp.vec3, device=device), wp.empty(0, dtype=wp.vec3, device=device)
        )
    bad = points.numpy()
    bad[0, 0] = np.nan
    with test.assertRaises(ValueError):
        geometry.poisson_reconstruct(wp.array(bad, dtype=wp.vec3, device=device), normals, depth=3, full_depth=1)
    with test.assertRaises(ValueError):
        geometry.poisson_reconstruct(points, normals, tape=wp.Tape(), depth=3, full_depth=1)
    result = geometry.poisson_reconstruct(points, normals, depth=3, full_depth=1)
    with test.assertRaises(TypeError):
        result.evaluate(wp.empty(0, dtype=wp.vec3d, device=device))
    with wp.Tape(), test.assertRaisesRegex(ValueError, "tape"):
        result.evaluate(points)
    with test.assertRaises(TypeError):
        geometry.poisson_extract_mesh(result, resolution=16, bounds=((-1.0,) * 3, (1.0,) * 3), method=1)
    with test.assertRaises(ValueError):
        geometry.poisson_extract_mesh(result, resolution=1, bounds=((-1.0,) * 3, (1.0,) * 3))
    for method in ("unknown", "lipschitz"):
        with test.subTest(method=method), test.assertRaises(ValueError):
            geometry.poisson_extract_mesh(result, resolution=16, bounds=((-1.0,) * 3, (1.0,) * 3), method=method)


def test_composed_reconstruction_gradient(test, device):
    """Match batch derivatives through a composed device query across CUDA streams."""
    points, normals = oriented_points(device, wp.vec3d, requires_grad=True)
    queries = wp.array([[0.13, 0.21, 0.17], [-0.1, -0.1, -0.1]], dtype=wp.vec3d, device=device, requires_grad=True)
    forward_stream, query_stream, backward_stream = wp.Stream(device), wp.Stream(device), wp.Stream(device)
    inputs_ready = wp.get_stream(device).record_event()
    tape = wp.Tape()
    with wp.ScopedStream(forward_stream):
        forward_stream.wait_event(inputs_ready)
        result = geometry.poisson_reconstruct(
            points, normals, depth=3, full_depth=1, iterations=2, samples_per_node=0.05, tape=tape
        )
    with wp.ScopedStream(query_stream):
        handle = result.device_field()
        with tape:
            actual = query_device(handle, queries, requires_grad=True)
            expected = result.evaluate(queries)
        value_seed = wp.array([0.7, -0.2], dtype=wp.float64, device=device)
        spatial_seed = wp.array([[0.03, -0.02, 0.01], [-0.04, 0.07, -0.02]], dtype=wp.vec3d, device=device)
        query_ready = query_stream.record_event()
    with wp.ScopedStream(backward_stream):
        backward_stream.wait_event(query_ready)
        tape.backward(grads={actual[0]: value_seed, actual[1]: spatial_seed})
        adjoints = [array.grad.numpy().copy() for array in (points, normals, queries)]
        tape.zero()
        tape.backward(grads={expected[0]: value_seed, expected[1]: spatial_seed})
        for array, adjoint in zip((points, normals, queries), adjoints, strict=True):
            np.testing.assert_allclose(array.grad.numpy(), adjoint, rtol=2e-10, atol=2e-11)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    with wp.ScopedCapture(device=device) as capture:
        captured = query_device(handle, queries)
    wp.capture_launch(capture.graph)
    for left, right in zip(captured, expected, strict=True):
        np.testing.assert_array_equal(left.numpy(), right.numpy())


def canonical_faces(indices):
    """Sort indexed triangles while preserving their winding."""
    faces = indices.reshape(-1, 3)
    faces = np.array([np.roll(face, -np.argmin(face)) for face in faces])
    return faces[np.lexsort(faces.T[::-1])]


def test_mesh(test, device):
    """Extract the same sphere geometry with sparse and dense marching cubes."""
    points, normals = sphere_points(device)
    result = geometry.poisson_reconstruct(points, normals, depth=4, full_depth=2)
    options = {"resolution": 24, "bounds": ((-1.02,) * 3, (1.02,) * 3)}
    with mock.patch.object(
        meshing, "sparse_marching_cubes_from_cells", wraps=meshing.sparse_marching_cubes_from_cells
    ) as extract:
        sparse = geometry.poisson_extract_mesh(result, **options)
    extract.assert_called_once()
    test.assertIsInstance(sparse, tuple)
    vertices, indices = sparse
    dense_vertices_array, dense_indices = geometry.poisson_extract_mesh(result, method="dense", **options)
    test.assertGreater(len(vertices), 0)
    test.assertEqual(len(indices) % 3, 0)
    test.assertEqual(len(indices), len(dense_indices))
    query_mesh = wp.Mesh(points=vertices, indices=indices)
    test.assertEqual(query_mesh.points.ptr, vertices.ptr)
    test.assertEqual(query_mesh.indices.ptr, indices.ptr)
    dense_vertices, sparse_vertices = dense_vertices_array.numpy(), vertices.numpy()
    test.assertEqual(dense_vertices.shape, sparse_vertices.shape)
    mapping = np.empty(len(sparse_vertices), dtype=np.int32)
    for begin in range(0, len(mapping), 128):
        vertex_batch = sparse_vertices[begin : begin + 128]
        distances = np.sum((vertex_batch[:, None, :] - dense_vertices[None, :, :]) ** 2, axis=2)
        mapping[begin : begin + len(vertex_batch)] = np.argmin(distances, axis=1)
    np.testing.assert_allclose(sparse_vertices, dense_vertices[mapping], rtol=0, atol=2e-6)
    test.assertEqual(len(np.unique(mapping)), len(mapping))
    np.testing.assert_array_equal(canonical_faces(dense_indices.numpy()), canonical_faces(mapping[indices.numpy()]))
    np.testing.assert_allclose(np.linalg.norm(vertices.numpy(), axis=1), 1.0, atol=0.1)
    empty_vertices, empty_indices = geometry.poisson_extract_mesh(result, iso_value=1e10, **options)
    test.assertEqual(len(empty_vertices), 0)
    test.assertEqual(len(empty_indices), 0)
    double = geometry.poisson_reconstruct(
        wp.array(points.numpy(), dtype=wp.vec3d, device=device),
        wp.array(normals.numpy(), dtype=wp.vec3d, device=device),
        depth=3,
        full_depth=1,
    )
    with test.assertRaises(ValueError):
        geometry.poisson_extract_mesh(double, **options)


def test_mesh_gradient(test, device, method, native=False):
    """Propagate mesh vertex losses through extraction and reconstruction."""
    points, normals = oriented_points(device, requires_grad=True)
    options = {"depth": 3, "full_depth": 1, "iterations": 2, "samples_per_node": 0.05}
    mesh_options = {"resolution": 10, "bounds": ((-0.6,) * 3, (0.6,) * 3), "method": method}
    if native:
        mesh_options = {"method": method}
    tape = wp.Tape()
    result = geometry.poisson_reconstruct(points, normals, tape=tape, **options)
    with tape:
        vertices, indices = geometry.poisson_extract_mesh(result, **mesh_options)
    test.assertGreater(len(vertices), 0)
    seed = np.random.default_rng(371).normal(size=(len(vertices), 3)).astype(np.float32)
    tape.backward(grads={vertices: wp.array(seed, dtype=wp.vec3, device=device)})
    original = normals.numpy().copy()
    adjoint = normals.grad.numpy().copy()
    direction = np.random.default_rng(457).normal(size=original.shape).astype(np.float32)
    losses = []
    epsilon = 1e-4
    for sign in (1, -1):
        normals.assign(original + sign * epsilon * direction)
        candidate_vertices, candidate_indices = geometry.poisson_extract_mesh(
            geometry.poisson_reconstruct(points, normals, **options), **mesh_options
        )
        test.assertEqual(candidate_vertices.shape, vertices.shape)
        np.testing.assert_array_equal(candidate_indices.numpy(), indices.numpy())
        losses.append(np.sum(candidate_vertices.numpy().astype(float) * seed))
    normals.assign(original)
    np.testing.assert_allclose(
        np.sum(adjoint.astype(float) * direction), (losses[0] - losses[1]) / (2 * epsilon), rtol=0.03, atol=0.01
    )

    if native:
        # Translation changes only the native model frame, not the cube samples.
        # This catches accidental use of the detached host normalization matrix.
        np.testing.assert_allclose(points.grad.numpy().sum(axis=0), seed.sum(axis=0), rtol=2e-4, atol=2e-4)


def assert_mesh_equal(test, sparse, dense):
    """Compare vertex coordinates and oriented faces independent of numbering."""
    sparse_vertices, sparse_indices = sparse[0].numpy(), sparse[1].numpy()
    dense_vertices, dense_indices = dense[0].numpy(), dense[1].numpy()
    test.assertEqual(sparse_vertices.shape, dense_vertices.shape)
    mapping = np.empty(len(sparse_vertices), dtype=np.int32)
    reverse = np.empty(len(dense_vertices), dtype=np.int32)
    for begin in range(0, len(mapping), 128):
        batch = sparse_vertices[begin : begin + 128]
        distances = np.sum((batch[:, None, :] - dense_vertices[None, :, :]) ** 2, axis=2)
        mapping[begin : begin + len(batch)] = np.argmin(distances, axis=1)
        batch = dense_vertices[begin : begin + 128]
        distances = np.sum((batch[:, None, :] - sparse_vertices[None, :, :]) ** 2, axis=2)
        reverse[begin : begin + len(batch)] = np.argmin(distances, axis=1)
    np.testing.assert_allclose(sparse_vertices, dense_vertices[mapping], rtol=0, atol=2e-6)
    np.testing.assert_allclose(dense_vertices, sparse_vertices[reverse], rtol=0, atol=2e-6)
    # Exact threshold knots can put different MC edges at the same vertex.
    # Compare their geometric representative without requiring an edge ordering.
    np.testing.assert_array_equal(
        canonical_faces(mapping[sparse_indices]), canonical_faces(mapping[reverse][dense_indices])
    )


def test_native_mesh(test, device):
    """Match dense extraction on the native grid with a translated, scaled model frame."""
    points, normals = sphere_points(device)
    points = wp.array(points.numpy() * 1.7 + [3.1, -2.3, 0.7], dtype=wp.vec3, device=device)
    result = geometry.poisson_reconstruct(points, normals, depth=4, full_depth=2)
    with mock.patch.object(meshing, "extract_isosurface", side_effect=AssertionError("resampling path used")):
        for resolution in (None, 9, (33, 33, 33)):
            sparse = geometry.poisson_extract_mesh(result, resolution=resolution)
            dense = geometry.poisson_extract_mesh(result, resolution=resolution, method="dense")
            assert_mesh_equal(test, sparse, dense)
            test.assertGreater(len(sparse[0]), 0)
    for resolution in (True, 10, (17, 9, 17), 65537):
        with test.assertRaises(ValueError):
            geometry.poisson_extract_mesh(result, resolution=resolution)
    with test.assertRaisesRegex(ValueError, "explicit resolution"):
        geometry.poisson_extract_mesh(result, bounds=((-1,) * 3, (1,) * 3))
    for method in ("sparse", "dense"):
        vertices, indices = geometry.poisson_extract_mesh(result, method=method, iso_value=1e10)
        test.assertEqual(len(vertices), 0)
        test.assertEqual(len(indices), 0)


def test_mesh_certificates(test, device):
    """Keep an interior fine-level crossing despite same-sign coarse corners and prune empty space."""
    keys = []
    coefficients = []
    for depth in range(4):
        nodes = range((1 << depth) + 1) if depth < 3 else range(3, 6)
        for x in nodes:
            for y in nodes:
                for z in nodes:
                    keys.append((depth << 48) | (x << 32) | (y << 16) | z)
                    coefficients.append(1.0 if depth == 0 else (-3.0 if (depth, x, y, z) == (3, 4, 4, 4) else 0.0))
    keys = wp.array(keys, dtype=wp.int64, device=device)
    coefficients = wp.array(coefficients, dtype=float, device=device)
    cumulative = field_module.cumulative_coefficients(keys, coefficients, 3)
    field = field_module.LevelField(keys, coefficients, tuple(np.eye(3).flat), (0.0, 0.0, 0.0), 3, 0.0, cumulative)
    raw = SimpleNamespace(field=field, metadata={"cube_to_model": np.eye(4).tolist()})
    iso = wp.zeros(1, dtype=float, device=device)
    root_corners = wp.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=wp.vec3, device=device)
    test.assertTrue(np.all(field.evaluate(root_corners)[0].numpy() > 0))
    test.assertLess(field.evaluate(wp.array([[0.5] * 3], dtype=wp.vec3, device=device))[0].numpy()[0], 0)
    statistics = {}
    sparse = meshing.extract_native_mesh(
        raw, resolution=65, backend="sparse", iso=iso, differentiable=False, statistics=statistics
    )
    dense = meshing.extract_native_mesh(raw, resolution=65, backend="dense", iso=iso, differentiable=False)
    test.assertGreater(len(sparse[0]), 0)
    assert_mesh_equal(test, sparse, dense)
    test.assertLess(statistics["candidates"], 64**3 // 4)
    test.assertLess(sum(statistics["visited"]), 64**3 // 4)
    test.assertGreater(statistics["crossing_cells"], 0)
    sparse = meshing.extract_native_mesh(
        raw, resolution=None, backend="sparse", iso=iso, differentiable=False, statistics=statistics
    )
    dense = meshing.extract_native_mesh(raw, resolution=None, backend="dense", iso=iso, differentiable=False)
    assert_mesh_equal(test, sparse, dense)
    test.assertGreater(len(sparse[0]), 0)
    test.assertEqual(statistics["bound_depth"], field.depth - 1)

    # Adaptive knots, mixed signs, exact equality, and both boundary nudges.
    random_field = adaptive_field(device, wp.float32)
    raw.field = random_field
    for threshold in (
        0.0,
        float(random_field.evaluate(wp.array([[0, 0, 0]], dtype=wp.vec3, device=device))[0].numpy()[0]),
    ):
        iso.assign([threshold])
        sparse = meshing.extract_native_mesh(raw, resolution=17, backend="sparse", iso=iso, differentiable=False)
        dense = meshing.extract_native_mesh(raw, resolution=17, backend="dense", iso=iso, differentiable=False)
        assert_mesh_equal(test, sparse, dense)

    random_field.coefficients.fill_(float("nan"))
    with test.assertRaisesRegex(ValueError, "finite"):
        meshing_bounds.candidate_cells(random_field, 4, iso)
    random_field.coefficients.zero_()
    random_field.cumulative.zero_()
    iso.zero_()
    statistics = {}
    test.assertEqual(len(meshing_bounds.candidate_cells(random_field, 15, iso, statistics=statistics)), 0)
    test.assertEqual(statistics["visited"], [1])


def test_reference_values(test, device):
    """Match fixed CPU reference values for screened and unscreened sphere reconstruction."""
    points = np.random.default_rng(2026).normal(size=(512, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    positions = wp.array(points, dtype=wp.vec3d, device=device)
    normals = wp.array(points, dtype=wp.vec3d, device=device)
    queries = wp.array(
        [
            [-0.3681430464414248, -0.221912265760416, -0.21708198260425227],
            [-0.07605888379530645, -0.07587018443735682, -0.07103990128119309],
            [0.2160252788508117, 0.8003823035009978, 0.36708634268798424],
            [0.654151522819989, 0.6543402221779386, 0.6591705053341024],
        ],
        dtype=wp.vec3d,
        device=device,
    )
    # Degree-1 Neumann CPU reference, with approximate screening and eight sweeps.
    references = [
        (
            0.0,
            -0.019816519182881364,
            [
                [0.49826140185438006, 0.020846675727542056, 0.030404614280776527, 0.05653988635179756],
                [0.5050854966827951, -0.001868266607815889, -0.0007590913290465814, 0.02282260541933435],
                [0.2462568638721857, -0.6900435857193463, -1.215102400215903, -0.9016267223437425],
                [-0.4149859598366291, -1.4196961598414644, -0.7453861009257762, -0.9945257634265876],
            ],
        ),
        (
            2.0,
            0.49815548193350817,
            [
                [1.0125086429667003, 0.01904103846607812, 0.03391451340472562, 0.05424336307283319],
                [1.018179675950853, -0.0064570359018198, 0.0016769920051081361, 0.019581541407856454],
                [0.7528124373409262, -0.6990760559548729, -1.2294703819233366, -0.883739604608302],
                [0.09545638210392472, -1.403648994761041, -0.7681443855881952, -0.9964931750476225],
            ],
        ),
    ]
    for weight, iso_value, expected in references:
        with test.subTest(point_weight=weight):
            result = geometry.poisson_reconstruct(positions, normals, depth=4, full_depth=2, point_weight=weight)
            values, gradients, valid = result.evaluate(queries)
            np.testing.assert_array_equal(valid.numpy(), 1)
            np.testing.assert_allclose(result.iso_value, iso_value, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(
                np.column_stack((values.numpy(), gradients.numpy())), expected, rtol=1e-6, atol=1e-8
            )


class TestPoissonReconstruction(unittest.TestCase):
    pass


for precision in (wp.float32, wp.float64):
    suffix = precision.__name__
    add_function_test(
        TestPoissonReconstruction,
        f"test_device_queries_{suffix}",
        test_device_queries,
        devices=get_test_devices(),
        dtype=precision,
    )
    add_function_test(
        TestPoissonReconstruction,
        f"test_device_query_gradient_{suffix}",
        test_device_query_gradient,
        devices=get_test_devices(),
        dtype=precision,
    )
    for screening in (0.0, 2.0):
        name = "screened" if screening else "unscreened"
        add_function_test(
            TestPoissonReconstruction,
            f"test_reconstruction_{name}_{suffix}",
            test_reconstruction,
            devices=get_cuda_test_devices(),
            dtype=precision,
            weight=screening,
        )
        add_function_test(
            TestPoissonReconstruction,
            f"test_reconstruction_gradient_{name}_{suffix}",
            test_reconstruction_gradient,
            devices=get_cuda_test_devices(),
            dtype=precision,
            weight=screening,
        )
add_function_test(TestPoissonReconstruction, "test_validation", test_validation, devices=get_cuda_test_devices())
add_function_test(
    TestPoissonReconstruction,
    "test_composed_reconstruction_gradient",
    test_composed_reconstruction_gradient,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestPoissonReconstruction, "test_reference_values", test_reference_values, devices=get_cuda_test_devices()
)
add_function_test(TestPoissonReconstruction, "test_mesh", test_mesh, devices=get_cuda_test_devices())
add_function_test(TestPoissonReconstruction, "test_native_mesh", test_native_mesh, devices=get_cuda_test_devices())
add_function_test(
    TestPoissonReconstruction, "test_mesh_certificates", test_mesh_certificates, devices=get_cuda_test_devices()
)
for mesh_method in ("sparse", "dense"):
    add_function_test(
        TestPoissonReconstruction,
        f"test_native_mesh_gradient_{mesh_method}",
        test_mesh_gradient,
        devices=get_cuda_test_devices(),
        method=mesh_method,
        native=True,
    )
    add_function_test(
        TestPoissonReconstruction,
        f"test_mesh_gradient_{mesh_method}",
        test_mesh_gradient,
        devices=get_cuda_test_devices(),
        method=mesh_method,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
