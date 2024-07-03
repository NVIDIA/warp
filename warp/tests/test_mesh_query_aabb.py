# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.func
def min_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def max_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.kernel
def compute_bounds(
    indices: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = indices[tid * 3 + 0]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]

    x0 = positions[i]  # point zero
    x1 = positions[j]  # point one
    x2 = positions[k]  # point two

    lower = min_vec3(min_vec3(x0, x1), x2)
    upper = max_vec3(max_vec3(x0, x1), x2)

    lowers[tid] = lower
    uppers[tid] = upper


@wp.kernel
def compute_num_contacts(
    lowers: wp.array(dtype=wp.vec3), uppers: wp.array(dtype=wp.vec3), mesh_id: wp.uint64, counts: wp.array(dtype=int)
):
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    tid = wp.tid()

    upper = uppers[tid]
    lower = lowers[tid]

    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    count = int(0)

    # index = int(-1)
    # while wp.mesh_query_aabb_next(query, index):

    for _index in query:
        count = count + 1

    counts[tid] = count


def test_compute_bounds(test, device):
    # create two touching triangles.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, -1, 1]])
    indices = np.array([0, 1, 2, 1, 2, 3])
    m = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=int, device=device),
    )

    num_tris = int(len(indices) / 3)

    # First compute bounds of each of the triangles.
    lowers = wp.empty(n=num_tris, dtype=wp.vec3, device=device)
    uppers = wp.empty_like(lowers)
    wp.launch(
        kernel=compute_bounds,
        dim=num_tris,
        inputs=[m.indices, m.points],
        outputs=[lowers, uppers],
        device=device,
    )

    lower_view = lowers.numpy()
    upper_view = uppers.numpy()

    # Confirm the bounds of each triangle are correct.
    test.assertTrue(lower_view[0][0] == 0)
    test.assertTrue(lower_view[0][1] == 0)
    test.assertTrue(lower_view[0][2] == 0)

    test.assertTrue(upper_view[0][0] == 1)
    test.assertTrue(upper_view[0][1] == 1)
    test.assertTrue(upper_view[0][2] == 0)

    test.assertTrue(lower_view[1][0] == -1)
    test.assertTrue(lower_view[1][1] == -1)
    test.assertTrue(lower_view[1][2] == 0)

    test.assertTrue(upper_view[1][0] == 1)
    test.assertTrue(upper_view[1][1] == 1)
    test.assertTrue(upper_view[1][2] == 1)


def test_mesh_query_aabb_count_overlap(test, device):
    # create two touching triangles.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, -1, 1]])
    indices = np.array([0, 1, 2, 1, 2, 3])
    m = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=int, device=device),
    )

    num_tris = int(len(indices) / 3)

    # Compute AABB of each of the triangles.
    lowers = wp.empty(n=num_tris, dtype=wp.vec3, device=device)
    uppers = wp.empty_like(lowers)
    wp.launch(
        kernel=compute_bounds,
        dim=num_tris,
        inputs=[m.indices, m.points],
        outputs=[lowers, uppers],
        device=device,
    )

    counts = wp.empty(n=num_tris, dtype=int, device=device)

    wp.launch(
        kernel=compute_num_contacts,
        dim=num_tris,
        inputs=[lowers, uppers, m.id],
        outputs=[counts],
        device=device,
    )

    view = counts.numpy()

    # 2 triangles that share a vertex having overlapping AABBs.
    for c in view:
        test.assertTrue(c == 2)


def test_mesh_query_aabb_count_nonoverlap(test, device):
    # create two separate triangles.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [10, 1, 0], [10, 0, 1]])
    indices = np.array([0, 1, 2, 3, 4, 5])
    m = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=int, device=device),
    )

    num_tris = int(len(indices) / 3)

    lowers = wp.empty(n=num_tris, dtype=wp.vec3, device=device)
    uppers = wp.empty_like(lowers)
    wp.launch(
        kernel=compute_bounds,
        dim=num_tris,
        inputs=[m.indices, m.points],
        outputs=[lowers, uppers],
        device=device,
    )

    counts = wp.empty(n=num_tris, dtype=int, device=device)

    wp.launch(
        kernel=compute_num_contacts,
        dim=num_tris,
        inputs=[lowers, uppers, m.id],
        outputs=[counts],
        device=device,
    )

    view = counts.numpy()

    # AABB query only returns one triangle at a time, the triangles are not close enough to overlap.
    for c in view:
        test.assertTrue(c == 1)


devices = get_test_devices()


class TestMeshQueryAABBMethods(unittest.TestCase):
    def test_mesh_query_aabb_codegen_adjoints_with_select(self):
        def kernel_fn(
            mesh: wp.uint64,
        ):
            v = wp.vec3(0.0, 0.0, 0.0)

            if True:
                query = wp.mesh_query_aabb(mesh, v, v)
            else:
                query = wp.mesh_query_aabb(mesh, v, v)

        wp.Kernel(func=kernel_fn)


add_function_test(TestMeshQueryAABBMethods, "test_compute_bounds", test_compute_bounds, devices=devices)
add_function_test(
    TestMeshQueryAABBMethods, "test_mesh_query_aabb_count_overlap", test_mesh_query_aabb_count_overlap, devices=devices
)
add_function_test(
    TestMeshQueryAABBMethods,
    "test_mesh_query_aabb_count_nonoverlap",
    test_mesh_query_aabb_count_nonoverlap,
    devices=devices,
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
