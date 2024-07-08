# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@dataclass
class TestData:
    a: Any
    b: Any
    t: float
    expected: Any
    expected_adj_a: Any = None
    expected_adj_b: Any = None
    expected_adj_t: float = None

    def check_backwards(self):
        return self.expected_adj_a is not None and self.expected_adj_b is not None and self.expected_adj_t is not None


TEST_DATA = {
    wp.float32: (
        TestData(
            a=1.0,
            b=2.0,
            t=1.5,
            expected=0.5,
            expected_adj_a=-0.75,
            expected_adj_b=-0.75,
            expected_adj_t=1.5,
        ),
        TestData(
            a=-1.0,
            b=2.0,
            t=-0.25,
            expected=0.15625,
            expected_adj_a=-0.28125,
            expected_adj_b=-0.09375,
            expected_adj_t=0.375,
        ),
        TestData(
            a=0.0,
            b=1.0,
            t=9.9,
            expected=1.0,
            expected_adj_a=0.0,
            expected_adj_b=0.0,
            expected_adj_t=0.0,
        ),
        TestData(
            a=0.0,
            b=1.0,
            t=-9.9,
            expected=0.0,
            expected_adj_a=0.0,
            expected_adj_b=0.0,
            expected_adj_t=0.0,
        ),
    ),
}


def test_smoothstep(test, device):
    def make_kernel_fn(data_type):
        def fn(
            a: wp.array(dtype=data_type),
            b: wp.array(dtype=data_type),
            t: wp.array(dtype=float),
            out: wp.array(dtype=data_type),
        ):
            out[0] = wp.smoothstep(a[0], b[0], t[0])

        return fn

    for data_type in TEST_DATA:
        kernel_fn = make_kernel_fn(data_type)
        kernel = wp.Kernel(
            func=kernel_fn,
            key=f"test_smoothstep{data_type.__name__}_kernel",
        )

        for test_data in TEST_DATA[data_type]:
            a = wp.array(
                [test_data.a],
                dtype=data_type,
                device=device,
                requires_grad=True,
            )
            b = wp.array(
                [test_data.b],
                dtype=data_type,
                device=device,
                requires_grad=True,
            )
            t = wp.array(
                [test_data.t],
                dtype=float,
                device=device,
                requires_grad=True,
            )
            out = wp.array(
                [0] * wp.types.type_length(data_type),
                dtype=data_type,
                device=device,
                requires_grad=True,
            )

            tape = wp.Tape()
            with tape:
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[a, b, t, out],
                    device=device,
                )

            assert_np_equal(
                out.numpy(),
                np.array([test_data.expected]),
                tol=1e-6,
            )

            if test_data.check_backwards():
                tape.backward(out)

                assert_np_equal(
                    tape.gradients[a].numpy(),
                    np.array([test_data.expected_adj_a]),
                    tol=1e-6,
                )
                assert_np_equal(
                    tape.gradients[b].numpy(),
                    np.array([test_data.expected_adj_b]),
                    tol=1e-6,
                )
                assert_np_equal(
                    tape.gradients[t].numpy(),
                    np.array([test_data.expected_adj_t]),
                    tol=1e-6,
                )


devices = get_test_devices()


class TestSmoothstep(unittest.TestCase):
    pass


add_function_test(TestSmoothstep, "test_smoothstep", test_smoothstep, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
