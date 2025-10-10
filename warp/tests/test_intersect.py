# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def intersect_tri(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    u0: wp.vec3,
    u1: wp.vec3,
    u2: wp.vec3,
    result_single_precision: wp.array(dtype=int),
    result_double_precision: wp.array(dtype=int),
):
    tid = wp.tid()

    result_single_precision[0] = wp.intersect_tri_tri(v0, v1, v2, u0, u1, u2)

    u0_d = wp.vec3d(u0)
    u1_d = wp.vec3d(u1)
    u2_d = wp.vec3d(u2)

    v0_d = wp.vec3d(v0)
    v1_d = wp.vec3d(v1)
    v2_d = wp.vec3d(v2)

    result_double_precision[0] = wp.intersect_tri_tri(v0_d, v1_d, v2_d, u0_d, u1_d, u2_d)


def test_intersect_tri(test, device):
    points_intersect = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.5, -0.5, 0.0),
        wp.vec3(0.5, -0.5, 1.0),
        wp.vec3(0.5, 0.5, 0.0),
    ]

    points_separated = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(-0.5, -0.5, 0.0),
        wp.vec3(-0.5, -0.5, 1.0),
        wp.vec3(-0.5, 0.5, 0.0),
    ]

    result_single_precision = wp.zeros(1, dtype=int, device=device)
    result_double_precision = wp.zeros(1, dtype=int, device=device)

    wp.launch(
        intersect_tri,
        dim=1,
        inputs=[*points_intersect, result_single_precision, result_double_precision],
        device=device,
    )
    assert_np_equal(result_single_precision.numpy(), np.array([1]))
    assert_np_equal(result_double_precision.numpy(), np.array([1]))

    wp.launch(
        intersect_tri,
        dim=1,
        inputs=[*points_separated, result_single_precision, result_double_precision],
        device=device,
    )
    assert_np_equal(result_single_precision.numpy(), np.array([0]))
    assert_np_equal(result_double_precision.numpy(), np.array([0]))


devices = get_test_devices()


class TestIntersect(unittest.TestCase):
    pass


add_function_test(TestIntersect, "test_intersect_tri", test_intersect_tri, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
