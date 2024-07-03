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


@wp.kernel
def intersect_tri(
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, u0: wp.vec3, u1: wp.vec3, u2: wp.vec3, result: wp.array(dtype=int)
):
    tid = wp.tid()

    result[0] = wp.intersect_tri_tri(v0, v1, v2, u0, u1, u2)


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

    result = wp.zeros(1, dtype=int, device=device)

    wp.launch(intersect_tri, dim=1, inputs=[*points_intersect, result], device=device)
    assert_np_equal(result.numpy(), np.array([1]))

    wp.launch(intersect_tri, dim=1, inputs=[*points_separated, result], device=device)
    assert_np_equal(result.numpy(), np.array([0]))


devices = get_test_devices()


class TestIntersect(unittest.TestCase):
    pass


add_function_test(TestIntersect, "test_intersect_tri", test_intersect_tri, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
