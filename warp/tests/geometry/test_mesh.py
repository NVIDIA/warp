# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gc
import itertools
import unittest
import weakref
from unittest import mock

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# fmt: off

POINT_POSITIONS = (
    ( 0.5, -0.5,  0.5),
    (-0.5, -0.5,  0.5),
    ( 0.5,  0.5,  0.5),
    (-0.5,  0.5,  0.5),
    (-0.5, -0.5, -0.5),
    ( 0.5, -0.5, -0.5),
    (-0.5,  0.5, -0.5),
    ( 0.5,  0.5, -0.5),
)

# Right-hand winding order. This corresponds to USD's (and others).
RIGHT_HANDED_FACE_VERTEX_INDICES = (
    0, 3, 1,
    0, 2, 3,
    4, 7, 5,
    4, 6, 7,
    6, 2, 7,
    6, 3, 2,
    5, 1, 4,
    5, 0, 1,
    5, 2, 0,
    5, 7, 2,
    1, 6, 4,
    1, 3, 6,
)


# Left-hand winding order. This corresponds to Houdini's (and others).
LEFT_HANDED_FACE_VERTEX_INDICES = (
    0, 1, 3,
    0, 3, 2,
    4, 5, 7,
    4, 7, 6,
    6, 7, 2,
    6, 2, 3,
    5, 4, 1,
    5, 1, 0,
    5, 0, 2,
    5, 2, 7,
    1, 4, 6,
    1, 6, 3,
)

# fmt: on

POINT_COUNT = 8
VERTEX_COUNT = 36
FACE_COUNT = 12


@wp.kernel(enable_backward=False)
def read_points_kernel(
    mesh_id: wp.uint64,
    out_points: wp.array[wp.vec3],
):
    tid = wp.tid()
    mesh = wp.mesh_get(mesh_id)
    out_points[tid] = mesh.points[tid]


@wp.kernel(enable_backward=False)
def read_indices_kernel(
    mesh_id: wp.uint64,
    out_indices: wp.array[int],
):
    tid = wp.tid()
    mesh = wp.mesh_get(mesh_id)
    out_indices[tid * 3 + 0] = mesh.indices[tid * 3 + 0]
    out_indices[tid * 3 + 1] = mesh.indices[tid * 3 + 1]
    out_indices[tid * 3 + 2] = mesh.indices[tid * 3 + 2]


def test_mesh_read_properties(test, device):
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    for constructor in constructors:
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=constructor)

        assert mesh.points.size == POINT_COUNT
        assert mesh.indices.size == VERTEX_COUNT
        assert int(mesh.indices.size / 3) == FACE_COUNT

        out_points = wp.empty(POINT_COUNT, dtype=wp.vec3, device=device)
        wp.launch(read_points_kernel, dim=POINT_COUNT, inputs=[mesh.id], outputs=[out_points], device=device)
        assert_np_equal(out_points.numpy(), np.array(POINT_POSITIONS))

        out_indices = wp.empty(VERTEX_COUNT, dtype=int, device=device)
        wp.launch(read_indices_kernel, dim=FACE_COUNT, inputs=[mesh.id], outputs=[out_indices], device=device)
        assert_np_equal(out_indices.numpy(), np.array(RIGHT_HANDED_FACE_VERTEX_INDICES))


@wp.kernel(enable_backward=False)
def query_point_kernel(
    mesh_id: wp.uint64,
    expected_sign: float,
):
    point = wp.vec3(0.1, 0.2, 0.3)
    expected_pos = wp.vec3(0.1, 0.2, 0.5)

    sign = float(0.0)
    face = int(0)
    bary_u = float(0.0)
    bary_v = float(0.0)

    wp.mesh_query_point(mesh_id, point, 1e6, sign, face, bary_u, bary_v)
    pos = wp.mesh_eval_position(mesh_id, face, bary_u, bary_v)

    wp.expect_eq(wp.sign(sign), expected_sign)
    wp.expect_eq(face, 1)
    wp.expect_near(wp.length(pos - expected_pos), 0.0)


def test_mesh_query_point(test, device):
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    for constructor in constructors:
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)

        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=constructor)
        expected_sign = -1.0
        wp.launch(query_point_kernel, dim=1, inputs=[mesh.id, expected_sign], device=device)

        indices = wp.array(LEFT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        mesh = wp.Mesh(points=points, indices=indices)
        expected_sign = 1.0
        wp.launch(query_point_kernel, dim=1, inputs=[mesh.id, expected_sign], device=device)


@wp.kernel(enable_backward=False)
def query_ray_kernel(
    mesh_id: wp.uint64,
    expected_sign: float,
):
    start = wp.vec3(0.1, 0.2, 0.3)
    dir = wp.normalize(wp.vec3(-1.2, 2.3, -3.4))
    expected_t = 0.557828
    expected_pos = wp.vec3(-0.0565217, 0.5, -0.143478)

    t = float(0.0)
    bary_u = float(0.0)
    bary_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)

    wp.mesh_query_ray(
        mesh_id,
        start,
        dir,
        1e6,
        t,
        bary_u,
        bary_v,
        sign,
        normal,
        face,
    )
    pos = wp.mesh_eval_position(mesh_id, face, bary_u, bary_v)

    wp.expect_near(t, expected_t)
    wp.expect_near(t, wp.length(pos - start), 1e-6)
    wp.expect_eq(wp.sign(sign), expected_sign)
    wp.expect_eq(face, 4)
    wp.expect_near(wp.length(pos - expected_pos), 0.0, 1e-6)


@wp.kernel(enable_backward=False)
def query_ray_group_kernel(
    mesh_id: wp.uint64,
):
    start = wp.vec3(0.1, 0.2, 0.3)
    dir = wp.normalize(wp.vec3(-1.2, 2.3, -3.4))
    expected_t = 0.557828

    t = float(0.0)
    bary_u = float(0.0)
    bary_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)
    root = -1

    hit = wp.mesh_query_ray(
        mesh_id,
        start,
        dir,
        1e6,
        t,
        bary_u,
        bary_v,
        sign,
        normal,
        face,
        root,
    )

    wp.expect_eq(hit, True)
    wp.expect_near(t, expected_t)


def test_mesh_query_ray(test, device):
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    if wp.is_cubql_available():
        constructors.append("cubql")

    leaf_sizes = [1, 2, 4]

    for leaf_size, constructor in itertools.product(leaf_sizes, constructors):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)

        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=constructor, bvh_leaf_size=leaf_size)
        expected_sign = -1.0
        wp.launch(
            query_ray_kernel,
            dim=1,
            inputs=[
                mesh.id,
                expected_sign,
            ],
            device=device,
        )

        indices = wp.array(LEFT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        mesh = wp.Mesh(points=points, indices=indices)
        expected_sign = 1.0
        wp.launch(
            query_ray_kernel,
            dim=1,
            inputs=[
                mesh.id,
                expected_sign,
            ],
            device=device,
        )


def test_grouped_mesh_query_ray(test, device):
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    leaf_sizes = [1, 2, 4]

    points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)
    indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
    same_group = wp.zeros(FACE_COUNT, dtype=int, device=device)
    different_group = np.ones(FACE_COUNT)
    different_group[: FACE_COUNT // 2] = 0
    different_group = wp.array(different_group, dtype=int, device=device)

    # Test that group construction maintains the same behavior as non-grouped construction
    for leaf_size, constructor in itertools.product(leaf_sizes, constructors):
        mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=constructor, bvh_leaf_size=leaf_size)
        wp.launch(query_ray_group_kernel, dim=1, inputs=[mesh.id], device=device)

        mesh = wp.Mesh(
            points=points, indices=indices, groups=same_group, bvh_constructor=constructor, bvh_leaf_size=leaf_size
        )
        wp.launch(query_ray_group_kernel, dim=1, inputs=[mesh.id], device=device)

        mesh = wp.Mesh(
            points=points, indices=indices, groups=different_group, bvh_constructor=constructor, bvh_leaf_size=leaf_size
        )
        wp.launch(query_ray_group_kernel, dim=1, inputs=[mesh.id], device=device)

        wp.synchronize_device(device)


@wp.kernel(enable_backward=False)
def query_ray_hit_kernel(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    direction: wp.vec3,
    hit_result: wp.array[wp.int32],
):
    t = float(0.0)
    bary_u = float(0.0)
    bary_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)

    hit = wp.mesh_query_ray(mesh_id, origin, direction, 1e6, t, bary_u, bary_v, sign, normal, face)
    if hit:
        hit_result[0] = 1
    else:
        hit_result[0] = 0


def test_mesh_refit(test, device):
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    if wp.is_cubql_available():
        constructors.append("cubql")

    # Ray aimed at the origin — hits the unit cube centered there
    origin = wp.vec3(0.0, 5.0, 0.0)
    direction = wp.vec3(0.0, -1.0, 0.0)
    offset = wp.vec3(10.0, 0.0, 0.0)

    for constructor in constructors:
        points_np = np.array(POINT_POSITIONS, dtype=np.float32)
        points = wp.array(points_np, dtype=wp.vec3, device=device)
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=constructor)

        hit_result = wp.zeros(1, dtype=wp.int32, device=device)

        # Ray should hit the original mesh at the origin
        wp.launch(query_ray_hit_kernel, dim=1, inputs=[mesh.id, origin, direction, hit_result], device=device)
        test.assertEqual(hit_result.numpy()[0], 1, f"Expected hit at origin ({constructor})")

        # Move the mesh, refit, and shoot at the new location
        moved_origin = wp.vec3(origin.x + offset.x, origin.y + offset.y, origin.z + offset.z)
        points_np += np.array([offset.x, offset.y, offset.z], dtype=np.float32)
        wp.copy(points, wp.array(points_np, dtype=wp.vec3, device=device))
        mesh.refit()

        # Ray at the new location should hit
        wp.launch(query_ray_hit_kernel, dim=1, inputs=[mesh.id, moved_origin, direction, hit_result], device=device)
        test.assertEqual(hit_result.numpy()[0], 1, f"Expected hit at moved location after refit ({constructor})")

        # Ray at the old location should miss
        wp.launch(query_ray_hit_kernel, dim=1, inputs=[mesh.id, origin, direction, hit_result], device=device)
        test.assertEqual(hit_result.numpy()[0], 0, f"Expected miss at origin after move ({constructor})")


def test_mesh_refit_graph(test, device):
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    for constructor in constructors:
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)

        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=constructor)

        wp.capture_begin(device, force_module_load=False)
        try:
            mesh.refit()
        finally:
            graph = wp.capture_end(device)

        # replay
        num_iters = 10
        for _ in range(num_iters):
            wp.capture_launch(graph)

        wp.synchronize_device(device)


@wp.kernel(enable_backward=False)
def count_aabb_hits_kernel(
    mesh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    hits: wp.array[wp.int32],
    slot: int,
):
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face = int(0)
    while wp.mesh_query_aabb_next(query, face):
        wp.atomic_add(hits, slot, 1)


@wp.kernel(enable_backward=False)
def eval_velocity_kernel(
    mesh_id: wp.uint64,
    out: wp.array[wp.vec3],
    slot: int,
):
    out[slot] = wp.mesh_eval_velocity(mesh_id, 0, 0.25, 0.25)


def _triangle_at(x):
    """Return the points of a unit right triangle in the z=0 plane, offset by ``x`` along the x axis."""
    return np.array([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]], dtype=np.float32)


def _launch_count_aabb_hits(mesh, x, hits, slot, device):
    """Count the triangles overlapping a box around the triangle placed by ``_triangle_at(x)``."""
    wp.launch(
        count_aabb_hits_kernel,
        dim=1,
        inputs=[mesh.id, wp.vec3(x - 0.5, -0.5, -0.5), wp.vec3(x + 1.5, 1.5, 0.5), hits, slot],
        device=device,
    )


def test_mesh_refit_capture_replay(test, device):
    """Re-run a captured refit on every replay, against the points as they are at replay time."""
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    wp.load_module(device=device)

    for constructor in constructors:
        for support_winding_number in (False, True):
            with test.subTest(constructor=constructor, support_winding_number=support_winding_number):
                points = wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device)
                src = wp.array(_triangle_at(10.0), dtype=wp.vec3, device=device)
                indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
                hits = wp.zeros(2, dtype=wp.int32, device=device)
                mesh = wp.Mesh(
                    points=points,
                    indices=indices,
                    support_winding_number=support_winding_number,
                    bvh_constructor=constructor,
                )

                with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                    hits.zero_()
                    wp.copy(points, src)
                    mesh.refit()
                    _launch_count_aabb_hits(mesh, 0.0, hits, 0, device)
                    _launch_count_aabb_hits(mesh, 10.0, hits, 1, device)

                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(hits.numpy(), np.array([0, 1], dtype=np.int32))

                src.assign(_triangle_at(0.0))
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(hits.numpy(), np.array([1, 0], dtype=np.int32))


def test_mesh_points_setter_capture_replay(test, device):
    """Apply a captured points assignment on every replay, in capture order."""
    points = wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device)
    new_points = wp.array(_triangle_at(-50.0), dtype=wp.vec3, device=device)
    src = wp.array(_triangle_at(10.0), dtype=wp.vec3, device=device)
    indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    hits = wp.zeros(3, dtype=wp.int32, device=device)
    read_points = wp.zeros(3, dtype=wp.vec3, device=device)
    mesh = wp.Mesh(points=points, indices=indices)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        hits.zero_()
        _launch_count_aabb_hits(mesh, 0.0, hits, 0, device)
        wp.copy(new_points, src)
        mesh.points = new_points
        _launch_count_aabb_hits(mesh, 10.0, hits, 1, device)
        _launch_count_aabb_hits(mesh, 20.0, hits, 2, device)

    test.assertIs(mesh.points, new_points)

    # Nothing runs before the first replay: the mesh still reads its original points.
    wp.launch(read_points_kernel, dim=3, inputs=[mesh.id, read_points], device=device)
    np.testing.assert_array_equal(read_points.numpy(), _triangle_at(0.0))

    # The query recorded before the assignment sees the original points; the one after sees the copied ones.
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(hits.numpy(), np.array([1, 1, 0], dtype=np.int32))
    wp.launch(read_points_kernel, dim=3, inputs=[mesh.id, read_points], device=device)
    np.testing.assert_array_equal(read_points.numpy(), _triangle_at(10.0))

    # The second replay refits against the replay-time contents of the new points.
    src.assign(_triangle_at(20.0))
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(hits.numpy(), np.array([0, 0, 1], dtype=np.int32))


def test_mesh_velocities_setter_capture_replay(test, device):
    """Apply a captured velocities assignment on every replay, in capture order."""
    points = wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device)
    indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    velocities = wp.full(3, wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3, device=device)
    new_velocities = wp.full(3, wp.vec3(0.0, 2.0, 0.0), dtype=wp.vec3, device=device)
    src = wp.full(3, wp.vec3(0.0, 0.0, 3.0), dtype=wp.vec3, device=device)
    out = wp.zeros(2, dtype=wp.vec3, device=device)
    mesh = wp.Mesh(points=points, indices=indices, velocities=velocities)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.launch(eval_velocity_kernel, dim=1, inputs=[mesh.id, out, 0], device=device)
        wp.copy(new_velocities, src)
        mesh.velocities = new_velocities
        wp.launch(eval_velocity_kernel, dim=1, inputs=[mesh.id, out, 1], device=device)

    test.assertIs(mesh.velocities, new_velocities)

    # Nothing runs before the first replay: the mesh still reads its original velocities.
    wp.launch(eval_velocity_kernel, dim=1, inputs=[mesh.id, out, 0], device=device)
    np.testing.assert_array_equal(out.numpy()[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))

    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(out.numpy()[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(out.numpy()[1], np.array([0.0, 0.0, 3.0], dtype=np.float32))


def test_mesh_points_setter_capture_if_replay(test, device):
    """Apply a points assignment recorded in a conditional branch only on the replays that take the branch.

    The assigned points are a view at a nonzero offset into a larger array, which the replay must resolve.
    """
    points = wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device)
    storage = wp.array(np.concatenate((_triangle_at(-50.0), _triangle_at(10.0))), dtype=wp.vec3, device=device)
    new_points = storage[3:]
    indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    condition = wp.zeros(1, dtype=wp.int32, device=device)
    hits = wp.zeros(1, dtype=wp.int32, device=device)
    read_points = wp.zeros(3, dtype=wp.vec3, device=device)
    mesh = wp.Mesh(points=points, indices=indices)

    def assign_points():
        mesh.points = new_points

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        hits.zero_()
        wp.capture_if(condition, on_true=assign_points)
        wp.launch(read_points_kernel, dim=3, inputs=[mesh.id, read_points], device=device)
        _launch_count_aabb_hits(mesh, 10.0, hits, 0, device)

    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(read_points.numpy(), _triangle_at(0.0))
    np.testing.assert_array_equal(hits.numpy(), np.array([0], dtype=np.int32))

    condition.fill_(1)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(read_points.numpy(), _triangle_at(10.0))
    np.testing.assert_array_equal(hits.numpy(), np.array([1], dtype=np.int32))


@wp.kernel
def write_points_kernel(mesh_id: wp.uint64, values: wp.array[wp.vec3]):
    tid = wp.tid()
    mesh = wp.mesh_get(mesh_id)
    mesh.points[tid] = values[tid]


@wp.kernel
def write_velocities_kernel(mesh_id: wp.uint64, values: wp.array[wp.vec3]):
    tid = wp.tid()
    mesh = wp.mesh_get(mesh_id)
    mesh.velocities[tid] = values[tid]


def test_mesh_points_setter_capture_keeps_retain_grad(test, device):
    """Replay a captured points assignment with the assigned array's ``retain_grad`` flag.

    The backward pass of a kernel that writes the mesh points clears their gradient unless the flag is set.
    """
    indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    kept_mesh = wp.Mesh(points=wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device), indices=indices)
    cleared_mesh = wp.Mesh(points=wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device), indices=indices)
    kept_points = wp.array(_triangle_at(10.0), dtype=wp.vec3, device=device, requires_grad=True)
    kept_points.retain_grad = True
    cleared_points = wp.array(_triangle_at(10.0), dtype=wp.vec3, device=device, requires_grad=True)
    kept_values = wp.zeros(3, dtype=wp.vec3, device=device, requires_grad=True)
    cleared_values = wp.zeros(3, dtype=wp.vec3, device=device, requires_grad=True)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        kept_mesh.points = kept_points
        cleared_mesh.points = cleared_points
    wp.capture_launch(capture.graph)

    tape = wp.Tape()
    with tape:
        wp.launch(write_points_kernel, dim=3, inputs=[kept_mesh.id, kept_values], device=device)
        wp.launch(write_points_kernel, dim=3, inputs=[cleared_mesh.id, cleared_values], device=device)
    ones = np.ones((3, 3), dtype=np.float32)
    tape.backward(
        grads={
            kept_points: wp.array(ones, dtype=wp.vec3, device=device),
            cleared_points: wp.array(ones, dtype=wp.vec3, device=device),
        }
    )

    # The adjoints read each gradient through the replayed mesh descriptor...
    np.testing.assert_array_equal(kept_values.grad.numpy(), ones)
    np.testing.assert_array_equal(cleared_values.grad.numpy(), ones)
    # ...and cleared only the one without retain_grad.
    np.testing.assert_array_equal(kept_points.grad.numpy(), ones)
    np.testing.assert_array_equal(cleared_points.grad.numpy(), np.zeros((3, 3), dtype=np.float32))


def test_mesh_velocities_setter_capture_keeps_retain_grad(test, device):
    """Replay a captured velocities assignment with the assigned array's ``retain_grad`` flag.

    The backward pass of a kernel that writes the mesh velocities clears their gradient unless the flag is set.
    """
    indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    kept_mesh = wp.Mesh(
        points=wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device),
        indices=indices,
        velocities=wp.zeros(3, dtype=wp.vec3, device=device),
    )
    cleared_mesh = wp.Mesh(
        points=wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device),
        indices=indices,
        velocities=wp.zeros(3, dtype=wp.vec3, device=device),
    )
    kept_velocities = wp.zeros(3, dtype=wp.vec3, device=device, requires_grad=True)
    kept_velocities.retain_grad = True
    cleared_velocities = wp.zeros(3, dtype=wp.vec3, device=device, requires_grad=True)
    kept_values = wp.zeros(3, dtype=wp.vec3, device=device, requires_grad=True)
    cleared_values = wp.zeros(3, dtype=wp.vec3, device=device, requires_grad=True)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        kept_mesh.velocities = kept_velocities
        cleared_mesh.velocities = cleared_velocities
    wp.capture_launch(capture.graph)

    tape = wp.Tape()
    with tape:
        wp.launch(write_velocities_kernel, dim=3, inputs=[kept_mesh.id, kept_values], device=device)
        wp.launch(write_velocities_kernel, dim=3, inputs=[cleared_mesh.id, cleared_values], device=device)
    ones = np.ones((3, 3), dtype=np.float32)
    tape.backward(
        grads={
            kept_velocities: wp.array(ones, dtype=wp.vec3, device=device),
            cleared_velocities: wp.array(ones, dtype=wp.vec3, device=device),
        }
    )

    # The adjoints read each gradient through the replayed mesh descriptor...
    np.testing.assert_array_equal(kept_values.grad.numpy(), ones)
    np.testing.assert_array_equal(cleared_values.grad.numpy(), ones)
    # ...and cleared only the one without retain_grad.
    np.testing.assert_array_equal(kept_velocities.grad.numpy(), ones)
    np.testing.assert_array_equal(cleared_velocities.grad.numpy(), np.zeros((3, 3), dtype=np.float32))


def test_mesh_cpu_construction_during_cuda_capture(test, device):
    """Build a CPU mesh eagerly while a CUDA graph capture is active: the CPU-capture check is device-scoped."""
    points = wp.array(_triangle_at(0.0), dtype=wp.vec3, device="cpu")
    indices = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")

    with wp.ScopedCapture(device=device, force_module_load=False):
        mesh = wp.Mesh(points=points, indices=indices)

    test.assertNotEqual(mesh.id, 0)


def test_mesh_exceptions(test, device):
    """Reject mesh points and indices located on different devices."""
    with test.assertRaises(RuntimeError):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device="cpu")
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        wp.Mesh(points=points, indices=indices)

    # points must be vec3
    with test.assertRaises(RuntimeError):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3d, device=device)
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        wp.Mesh(points=points, indices=indices)

    # velocities must be vec3
    with test.assertRaises(RuntimeError):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)
        velocities = wp.zeros(points.shape, dtype=wp.vec3d, device=device)
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        wp.Mesh(points=points, indices=indices, velocities=velocities)

    # indices must be int32
    with test.assertRaises(RuntimeError):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=wp.int64, device=device)
        wp.Mesh(points=points, indices=indices)

    # indices must be 1d
    with test.assertRaises(RuntimeError):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        indices = indices.reshape((3, -1))
        wp.Mesh(points=points, indices=indices)

    # grouped queries are not supported with the cuBQL constructor
    with test.assertRaises(RuntimeError):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        groups = wp.zeros(FACE_COUNT, dtype=int, device=device)
        wp.Mesh(points=points, indices=indices, groups=groups, bvh_constructor="cubql")

    # winding number support is not available with the cuBQL constructor
    with test.assertRaises(RuntimeError):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        wp.Mesh(points=points, indices=indices, support_winding_number=True, bvh_constructor="cubql")

    # unknown bvh_constructor string raises ValueError
    with test.assertRaises(ValueError):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device=device)
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=device)
        wp.Mesh(points=points, indices=indices, bvh_constructor="cuqbl")


devices = get_test_devices()
cuda_devices_with_mempool = get_selected_cuda_test_devices_with_mempool()


class TestMesh(unittest.TestCase):
    def test_mesh_new_del(self):
        """Delete a mesh that was allocated without initialization."""
        instance = wp.Mesh.__new__(wp.Mesh)
        instance.__del__()

    def test_mesh_create_raises_on_native_failure(self):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device="cpu")
        indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device="cpu")
        runtime = wp._src.context.runtime

        with (
            mock.patch.object(runtime.core, "wp_mesh_create_host", return_value=0),
            mock.patch.object(runtime, "get_error_string", return_value="native failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to create mesh: native failure"):
                wp.Mesh(points=points, indices=indices)

    def test_mesh_refit_raises_on_native_device_failure(self):
        mesh = wp.Mesh.__new__(wp.Mesh)
        mesh.id = 123
        mesh.device = mock.Mock(is_cpu=False)
        mesh.runtime = mock.Mock()
        mesh.runtime.core.wp_mesh_refit_device.return_value = 0
        mesh.runtime.get_error_string.return_value = "native refit failure"

        with self.assertRaisesRegex(RuntimeError, "Failed to refit mesh: native refit failure"):
            mesh.refit()

    def test_mesh_points_setter_raises_on_native_device_failure(self):
        points = wp.array(POINT_POSITIONS, dtype=wp.vec3, device="cpu")
        points_new = wp.array(POINT_POSITIONS, dtype=wp.vec3, device="cpu")

        mesh = wp.Mesh.__new__(wp.Mesh)
        mesh.id = 123
        mesh.device = mock.Mock(is_cpu=False)
        mesh._points = points
        mesh.runtime = mock.Mock()
        mesh.runtime.core.wp_mesh_set_points_device.return_value = 0
        mesh.runtime.get_error_string.return_value = "native refit failure"

        with self.assertRaisesRegex(RuntimeError, "Failed to set mesh points: native refit failure"):
            mesh.points = points_new

    def test_mesh_points_setter_capture_keeps_replaced_points_alive(self):
        """Keep the points a captured assignment replaces alive while the native mesh can still read them.

        The assignment takes effect only at replay, so until then the mesh keeps reading the replaced points, even if
        the graph is destroyed without being launched.
        """
        device = "cpu"
        indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        mesh = wp.Mesh(points=wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device), indices=indices)
        replaced_points = weakref.ref(mesh.points)
        read_points = wp.zeros(3, dtype=wp.vec3, device=device)

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            mesh.points = wp.array(_triangle_at(10.0), dtype=wp.vec3, device=device)

        gc.collect()
        self.assertIsNotNone(replaced_points())

        del capture
        gc.collect()
        self.assertIsNotNone(replaced_points())
        wp.launch(read_points_kernel, dim=3, inputs=[mesh.id, read_points], device=device)
        np.testing.assert_array_equal(read_points.numpy(), _triangle_at(0.0))

    def test_mesh_points_setter_capture_keeps_replayed_points_alive(self):
        """Keep the points a replay assigns alive while the native mesh reads them, even after the graph is destroyed.

        An eager assignment made between capture and replay doesn't stop the replay from repointing the mesh.
        """
        device = "cpu"
        points = wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device)
        indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        mesh = wp.Mesh(points=points, indices=indices)
        replayed_points = wp.array(_triangle_at(10.0), dtype=wp.vec3, device=device)
        replayed_points_ref = weakref.ref(replayed_points)
        read_points = wp.zeros(3, dtype=wp.vec3, device=device)

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            mesh.points = replayed_points
        del replayed_points

        mesh.points = points
        wp.capture_launch(capture.graph)
        del capture
        gc.collect()

        self.assertIsNotNone(replayed_points_ref())
        wp.launch(read_points_kernel, dim=3, inputs=[mesh.id, read_points], device=device)
        np.testing.assert_array_equal(read_points.numpy(), _triangle_at(10.0))

        # An eager assignment repoints the mesh, so nothing can read the replayed points anymore.
        mesh.points = points
        gc.collect()
        self.assertIsNone(replayed_points_ref())

    def test_mesh_points_setter_recapture_releases_unread_points(self):
        """Release the points of destroyed graphs once the native mesh stops reading them, instead of accumulating them."""
        device = "cpu"
        indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        points = wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device)
        mesh = wp.Mesh(points=points, indices=indices)
        assigned_points_refs = []

        wp.load_module(device=device)
        for i in range(4):
            assigned_points = wp.array(_triangle_at(10.0 * (i + 1)), dtype=wp.vec3, device=device)
            assigned_points_refs.append(weakref.ref(assigned_points))
            with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                mesh.points = assigned_points
            wp.capture_launch(capture.graph)
            del assigned_points, capture
            gc.collect()

        # The last two capture rounds may still be read through the native mesh; the earlier ones may not.
        self.assertIsNone(assigned_points_refs[0]())
        self.assertIsNone(assigned_points_refs[1]())

        mesh.points = points
        gc.collect()
        self.assertIsNone(assigned_points_refs[2]())
        self.assertIsNone(assigned_points_refs[3]())

    def test_mesh_velocities_setter_capture_keeps_replaced_velocities_alive(self):
        """Keep the velocities a captured assignment replaces alive while the native mesh can still read them."""
        device = "cpu"
        points = wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device)
        indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        mesh = wp.Mesh(
            points=points,
            indices=indices,
            velocities=wp.full(3, wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3, device=device),
        )
        replaced_velocities = weakref.ref(mesh.velocities)
        out = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            mesh.velocities = wp.full(3, wp.vec3(0.0, 2.0, 0.0), dtype=wp.vec3, device=device)

        gc.collect()
        self.assertIsNotNone(replaced_velocities())

        del capture
        gc.collect()
        self.assertIsNotNone(replaced_velocities())
        wp.launch(eval_velocity_kernel, dim=1, inputs=[mesh.id, out, 0], device=device)
        np.testing.assert_array_equal(out.numpy()[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def test_mesh_construction_cpu_capture_raises(self):
        """Reject building a mesh during CPU graph capture, where the build would never re-run on replay."""
        device = "cpu"
        points = wp.array(_triangle_at(0.0), dtype=wp.vec3, device=device)
        indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)

        with self.assertRaisesRegex(NotImplementedError, r"wp\.Mesh\(\) cannot be recorded during CPU graph capture"):
            with wp.ScopedCapture(device=device, force_module_load=False):
                wp.Mesh(points=points, indices=indices)

        # The capture ended cleanly, so eager construction works again.
        mesh = wp.Mesh(points=points, indices=indices)
        self.assertNotEqual(mesh.id, 0)


add_function_test(TestMesh, "test_mesh_read_properties", test_mesh_read_properties, devices=devices)
add_function_test(TestMesh, "test_mesh_query_point", test_mesh_query_point, devices=devices)
add_function_test(TestMesh, "test_mesh_query_ray", test_mesh_query_ray, devices=devices)
add_function_test(TestMesh, "test_grouped_mesh_query_ray", test_grouped_mesh_query_ray, devices=devices)
add_function_test(TestMesh, "test_mesh_refit", test_mesh_refit, devices=devices)
add_function_test(TestMesh, "test_mesh_refit_graph", test_mesh_refit_graph, devices=cuda_devices_with_mempool)
add_function_test(
    TestMesh,
    "test_mesh_refit_capture_replay",
    test_mesh_refit_capture_replay,
    devices=get_test_devices_with_graph_capture_allocation(),
)
# CUDA is excluded until the CUDA setters stop capturing a copy from a stack descriptor (mesh.cu).
add_function_test(
    TestMesh,
    "test_mesh_points_setter_capture_replay",
    test_mesh_points_setter_capture_replay,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestMesh,
    "test_mesh_velocities_setter_capture_replay",
    test_mesh_velocities_setter_capture_replay,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestMesh,
    "test_mesh_points_setter_capture_if_replay",
    test_mesh_points_setter_capture_if_replay,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestMesh,
    "test_mesh_points_setter_capture_keeps_retain_grad",
    test_mesh_points_setter_capture_keeps_retain_grad,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestMesh,
    "test_mesh_velocities_setter_capture_keeps_retain_grad",
    test_mesh_velocities_setter_capture_keeps_retain_grad,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestMesh,
    "test_mesh_cpu_construction_during_cuda_capture",
    test_mesh_cpu_construction_during_cuda_capture,
    devices=cuda_devices_with_mempool,
)
add_function_test(TestMesh, "test_mesh_exceptions", test_mesh_exceptions, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
