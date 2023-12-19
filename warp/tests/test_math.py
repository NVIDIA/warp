# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
from typing import NamedTuple

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()


class ScalarFloatValues(NamedTuple):
    degrees: wp.float32 = None
    radians: wp.float32 = None


@wp.kernel
def scalar_float_kernel(
    i: int,
    x: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    if i == 0:
        out[0] = wp.degrees(x[0])
    elif i == 1:
        out[0] = wp.radians(x[0])


def test_scalar_math(test, device):
    float_values = ScalarFloatValues(
        degrees=(0.123,),
        radians=(123.0,),
    )
    float_results_expected = ScalarFloatValues(
        degrees=7.047381,
        radians=2.146755,
    )
    adj_float_results_expected = ScalarFloatValues(
        degrees=57.29578,
        radians=0.017453,
    )
    for i, values in enumerate(float_values):
        x = wp.array(
            [values[0]],
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        )
        out = wp.array(
            [0.0],
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        )

        tape = wp.Tape()
        with tape:
            wp.launch(
                scalar_float_kernel,
                dim=1,
                inputs=[i, x, out],
                device=device,
            )

        assert_np_equal(
            out.numpy(),
            np.array([float_results_expected[i]]),
            tol=1e-6,
        )

        tape.backward(out)

        assert_np_equal(
            tape.gradients[x].numpy(),
            np.array([adj_float_results_expected[i]]),
            tol=1e-6,
        )


def test_vec_type(test, device):
    vec5 = wp.vec(length=5, dtype=float)
    v = vec5()
    w = vec5()
    a = vec5(1.0)
    b = vec5(0.0, 0.0, 0.0, 0.0, 0.0)
    c = vec5(0.0)

    v[0] = 1.0
    v.x = 0.0
    v[1:] = [1.0, 1.0, 1.0, 1.0]

    w[0] = 1.0
    w[1:] = [0.0, 0.0, 0.0, 0.0]

    if v[0] != w[1] or v.x != w.y:
        raise ValueError("vec setter error")

    for x in v[1:]:
        if x != 1.0:
            raise ValueError("vec slicing error")

    if b != c:
        raise ValueError("vec equality error")

    if str(v) != "[0.0, 1.0, 1.0, 1.0, 1.0]":
        raise ValueError("vec to string error")


def test_mat_type(test, device):
    mat55 = wp.mat(shape=(5, 5), dtype=float)
    m1 = mat55()
    m2 = mat55()

    for i in range(5):
        for j in range(5):
            if i == j:
                m1[i, j] = 1.0
            else:
                m1[i, j] = 0.0

    for i in range(5):
        m2[i] = [1.0, 1.0, 1.0, 1.0, 1.0]

    a = mat55(1.0)
    b = mat55(
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    if m1 != b:
        raise ValueError("mat element setting error")

    if m2 != a:
        raise ValueError("mat row setting error")

    if m1[0, 0] != 1.0:
        raise ValueError("mat element getting error")

    if m2[0] != [1.0, 1.0, 1.0, 1.0, 1.0]:
        raise ValueError("mat row getting error")

    if (
        str(b)
        != "[[1.0, 0.0, 0.0, 0.0, 0.0],\n [0.0, 1.0, 0.0, 0.0, 0.0],\n [0.0, 0.0, 1.0, 0.0, 0.0],\n [0.0, 0.0, 0.0, 1.0, 0.0],\n [0.0, 0.0, 0.0, 0.0, 1.0]]"
    ):
        raise ValueError("mat to string error")


devices = get_test_devices()


class TestMath(unittest.TestCase):
    pass


add_function_test(TestMath, "test_scalar_math", test_scalar_math, devices=devices)
add_function_test(TestMath, "test_vec_type", test_vec_type, devices=devices)
add_function_test(TestMath, "test_mat_type", test_mat_type, devices=devices)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
