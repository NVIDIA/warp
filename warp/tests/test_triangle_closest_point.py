# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from warp.sim.collide import triangle_closest_point_barycentric
from warp.tests.unittest_utils import *


# a-b is the edge where the closest point is located at
@wp.func
def check_edge_feasible_region(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, eps: float):
    ap = p - a
    bp = p - b
    ab = b - a

    if wp.dot(ap, ab) < -eps:
        return False

    if wp.dot(bp, ab) > eps:
        return False

    ab_sqr_norm = wp.dot(ab, ab)
    if ab_sqr_norm < eps:
        return False

    t = wp.dot(ab, c - a) / ab_sqr_norm

    perpendicular_foot = a + t * ab

    if wp.dot(c - perpendicular_foot, p - perpendicular_foot) > eps:
        return False

    return True


# closest point is a
@wp.func
def check_vertex_feasible_region(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, eps: float):
    ap = p - a
    ba = a - b
    ca = a - c

    if wp.dot(ap, ba) < -eps:
        return False

    if wp.dot(p, ca) < -eps:
        return False

    return True


@wp.kernel
def test_triangle_closest_point_kernel(tri: wp.array(dtype=wp.vec3), passed: wp.array(dtype=wp.bool)):
    state = wp.uint32(wp.rand_init(wp.int32(123), wp.int32(0)))
    eps = 1e-5

    a = tri[0]
    b = tri[1]
    c = tri[2]

    for _i in range(1000):
        l = wp.float32(0.0)
        while l < eps:
            p = wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))
            l = wp.length(p)

        # project to a sphere with r=2
        p = 2.0 * p / l

        bary = triangle_closest_point_barycentric(tri[0], tri[1], tri[2], p)

        for dim in range(3):
            v1_index = (dim + 1) % 3
            v2_index = (dim + 2) % 3
            v1 = tri[v1_index]
            v2 = tri[v2_index]
            v3 = tri[dim]

            # on edge
            if bary[dim] == 0.0 and bary[v1_index] != 0.0 and bary[v2_index] != 0.0:
                if not check_edge_feasible_region(p, v1, v2, v3, eps):
                    passed[0] = False
                    return

                # p-closest_p must be perpendicular to v1-v2
                closest_p = a * bary[0] + b * bary[1] + c * bary[2]
                e = v1 - v2
                err = wp.dot(e, closest_p - p)
                if wp.abs(err) > eps:
                    passed[0] = False
                    return

            if bary[v1_index] == 0.0 and bary[v2_index] == 0.0:
                if not check_vertex_feasible_region(p, v3, v1, v2, eps):
                    passed[0] = False
                    return

            if bary[dim] != 0.0 and bary[v1_index] != 0.0 and bary[v2_index] != 0.0:
                closest_p = a * bary[0] + b * bary[1] + c * bary[2]
                e1 = v1 - v2
                e2 = v1 - v3
                if wp.abs(wp.dot(e1, closest_p - p)) > eps or wp.abs(wp.dot(e2, closest_p - p)) > eps:
                    passed[0] = False
                    return


def test_triangle_closest_point(test, device):
    passed = wp.array([True], dtype=wp.bool, device=device)

    a = wp.vec3(1.0, 0.0, 0.0)
    b = wp.vec3(0.0, 0.0, 0.0)
    c = wp.vec3(0.0, 1.0, 0.0)

    tri = wp.array([a, b, c], dtype=wp.vec3, device=device)
    wp.launch(test_triangle_closest_point_kernel, dim=1, inputs=[tri, passed], device=device)
    passed = passed.numpy()

    test.assertTrue(passed.all())


devices = get_test_devices()


class TestTriangleClosestPoint(unittest.TestCase):
    pass


add_function_test(TestTriangleClosestPoint, "test_triangle_closest_point", test_triangle_closest_point, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
