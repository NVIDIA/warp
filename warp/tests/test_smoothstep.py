# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        TestData(a=1.0, b=2.0, t=1.5, expected=0.5, expected_adj_a=-0.75, expected_adj_b=-0.75, expected_adj_t=1.5),
        TestData(
            a=-1.0,
            b=2.0,
            t=-0.25,
            expected=0.15625,
            expected_adj_a=-0.28125,
            expected_adj_b=-0.09375,
            expected_adj_t=0.375,
        ),
        TestData(a=0.0, b=1.0, t=9.9, expected=1.0, expected_adj_a=0.0, expected_adj_b=0.0, expected_adj_t=0.0),
        TestData(a=0.0, b=1.0, t=-9.9, expected=0.0, expected_adj_a=0.0, expected_adj_b=0.0, expected_adj_t=0.0),
    ),
}


def test_smoothstep(test, device):
    def make_kernel_fn(data_type):
        def fn(
            a: wp.array[data_type],
            b: wp.array[data_type],
            t: wp.array[float],
            out: wp.array[data_type],
        ):
            out[0] = wp.smoothstep(a[0], b[0], t[0])

        return fn

    for data_type, test_data_set in TEST_DATA.items():
        kernel_fn = make_kernel_fn(data_type)
        kernel = wp.Kernel(
            func=kernel_fn,
            key=f"test_smoothstep{data_type.__name__}_kernel",
        )

        for test_data in test_data_set:
            a = wp.array([test_data.a], dtype=data_type, device=device, requires_grad=True)
            b = wp.array([test_data.b], dtype=data_type, device=device, requires_grad=True)
            t = wp.array([test_data.t], dtype=float, device=device, requires_grad=True)
            out = wp.array([0] * wp.types.type_size(data_type), dtype=data_type, device=device, requires_grad=True)

            with wp.Tape() as tape:
                wp.launch(kernel, dim=1, inputs=[a, b, t, out], device=device)

            assert_np_equal(out.numpy(), np.array([test_data.expected]), tol=1e-6)

            if test_data.check_backwards():
                tape.backward(out)

                assert_np_equal(tape.gradients[a].numpy(), np.array([test_data.expected_adj_a]), tol=1e-6)
                assert_np_equal(tape.gradients[b].numpy(), np.array([test_data.expected_adj_b]), tol=1e-6)
                assert_np_equal(tape.gradients[t].numpy(), np.array([test_data.expected_adj_t]), tol=1e-6)


def test_smoothstep_scaled_gradient(test, device):
    for data_type, numpy_type, scale, rtol in (
        (wp.float16, np.float16, 0.01, 2.0e-3),
        (wp.float32, np.float32, 1.0e-12, 2.0e-6),
        (wp.float64, np.float64, 1.0e-90, 2.0e-14),
    ):

        def fn(
            a: wp.array[data_type],
            b: wp.array[data_type],
            x: wp.array[data_type],
            out: wp.array[data_type],
        ):
            i = wp.tid()
            out[i] = wp.smoothstep(a[i], b[i], x[i]) + wp.smoothstep(a[i], b[i], x[i])

        kernel = wp.Kernel(func=fn, key=f"smoothstep_scaled_gradient_{data_type.__name__}")
        cases = []
        for width in (scale, 1.0, 1.0 / scale):
            cases.extend(
                (
                    (0.0, width, width * 0.5),
                    (width, 0.0, width * 0.5),
                    (0.0, width, -width),
                    (0.0, width, 0.0),
                    (0.0, width, width),
                    (0.0, width, width * 2.0),
                )
            )
        values = np.asarray(cases, dtype=numpy_type)
        a = wp.array(values[:, 0], dtype=data_type, device=device, requires_grad=True)
        b = wp.array(values[:, 1], dtype=data_type, device=device, requires_grad=True)
        x = wp.array(values[:, 2], dtype=data_type, device=device, requires_grad=True)
        out = wp.zeros(len(cases), dtype=data_type, device=device, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(kernel, dim=len(cases), inputs=[a, b, x], outputs=[out], device=device)
        tape.backward(grads={out: wp.ones(len(cases), dtype=data_type, device=device)})

        expected_value = np.tile([1.0, 1.0, 0.0, 0.0, 2.0, 2.0], 3)
        expected_gradients = np.zeros((len(cases), 3), dtype=np.float64)
        for i in range(len(cases)):
            if i % 6 < 2:
                width = float(values[i, 1]) - float(values[i, 0])
                expected_gradients[i] = (-1.5 / width, -1.5 / width, 3.0 / width)
        np.testing.assert_array_equal(out.numpy(), expected_value)
        for array, expected in zip((a, b, x), expected_gradients.T, strict=True):
            np.testing.assert_allclose(array.grad.numpy(), expected, rtol=rtol, atol=0.0)


devices = get_test_devices()


class TestSmoothstep(unittest.TestCase):
    pass


add_function_test(TestSmoothstep, "test_smoothstep", test_smoothstep, devices=devices)
add_function_test(TestSmoothstep, "test_smoothstep_scaled_gradient", test_smoothstep_scaled_gradient, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
