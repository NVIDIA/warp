# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import NamedTuple
import unittest

import numpy as np

import warp as wp
from warp.tests.test_base import *

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

def register(parent):
    devices = get_test_devices()

    class TestMath(parent):
        pass

    add_function_test(TestMath, "test_scalar_math", test_scalar_math, devices=devices)
    return TestMath

if __name__ == "__main__":
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
