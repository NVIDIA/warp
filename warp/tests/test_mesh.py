# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.test_base import *


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


wp.init()


@wp.kernel(enable_backward=False)
def read_points_kernel(
    mesh_id: wp.uint64,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    mesh = wp.mesh_get(mesh_id)
    out_points[tid] = mesh.points[tid]


@wp.kernel(enable_backward=False)
def read_indices_kernel(
    mesh_id: wp.uint64,
    out_indices: wp.array(dtype=int),
):
    tid = wp.tid()
    mesh = wp.mesh_get(mesh_id)
    out_indices[tid * 3 + 0] = mesh.indices[tid * 3 + 0]
    out_indices[tid * 3 + 1] = mesh.indices[tid * 3 + 1]
    out_indices[tid * 3 + 2] = mesh.indices[tid * 3 + 2]


def test_mesh_read_properties(test, device):
    points = wp.array(POINT_POSITIONS, dtype=wp.vec3)
    indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int)
    mesh = wp.Mesh(points=points, indices=indices)

    assert mesh.points.size == POINT_COUNT
    assert mesh.indices.size == VERTEX_COUNT
    assert int(mesh.indices.size / 3) == FACE_COUNT

    out_points = wp.empty(POINT_COUNT, dtype=wp.vec3)
    wp.launch(
        read_points_kernel,
        dim=POINT_COUNT,
        inputs=[
            mesh.id,
        ],
        outputs=[
            out_points,
        ],
    )
    assert_np_equal(out_points.numpy(), np.array(POINT_POSITIONS))

    out_indices = wp.empty(VERTEX_COUNT, dtype=int)
    wp.launch(
        read_indices_kernel,
        dim=FACE_COUNT,
        inputs=[
            mesh.id,
        ],
        outputs=[
            out_indices,
        ],
    )
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

    wp.mesh_query_point(
        mesh_id,
        point,
        1e6,
        sign,
        face,
        bary_u,
        bary_v,
    )
    pos = wp.mesh_eval_position(mesh_id, face, bary_u, bary_v)

    wp.expect_eq(wp.sign(sign), expected_sign)
    wp.expect_eq(face, 1)
    wp.expect_near(wp.length(pos - expected_pos), 0.0)


def test_mesh_query_point(test, device):
    points = wp.array(POINT_POSITIONS, dtype=wp.vec3)

    indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int)
    mesh = wp.Mesh(points=points, indices=indices)
    expected_sign = -1.0
    wp.launch(
        query_point_kernel,
        dim=1,
        inputs=[
            mesh.id,
            expected_sign,
        ],
    )

    indices = wp.array(LEFT_HANDED_FACE_VERTEX_INDICES, dtype=int)
    mesh = wp.Mesh(points=points, indices=indices)
    expected_sign = 1.0
    wp.launch(
        query_point_kernel,
        dim=1,
        inputs=[
            mesh.id,
            expected_sign,
        ],
    )


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


def test_mesh_query_ray(test, device):
    points = wp.array(POINT_POSITIONS, dtype=wp.vec3)

    indices = wp.array(RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int)
    mesh = wp.Mesh(points=points, indices=indices)
    expected_sign = -1.0
    wp.launch(
        query_ray_kernel,
        dim=1,
        inputs=[
            mesh.id,
            expected_sign,
        ],
    )

    indices = wp.array(LEFT_HANDED_FACE_VERTEX_INDICES, dtype=int)
    mesh = wp.Mesh(points=points, indices=indices)
    expected_sign = 1.0
    wp.launch(
        query_ray_kernel,
        dim=1,
        inputs=[
            mesh.id,
            expected_sign,
        ],
    )


def register(parent):
    devices = get_test_devices()

    class TestMesh(parent):
        pass

    add_function_test(TestMesh, "test_mesh_read_properties", test_mesh_read_properties, devices=devices)
    add_function_test(TestMesh, "test_mesh_query_point", test_mesh_query_point, devices=devices)
    add_function_test(TestMesh, "test_mesh_query_ray", test_mesh_query_ray, devices=devices)
    return TestMesh


if __name__ == "__main__":
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
