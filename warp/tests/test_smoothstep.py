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
            a: wp.array(dtype=data_type),
            b: wp.array(dtype=data_type),
            t: wp.array(dtype=float),
            out: wp.array(dtype=data_type),
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
            out = wp.array([0] * wp.types.type_length(data_type), dtype=data_type, device=device, requires_grad=True)

            with wp.Tape() as tape:
                wp.launch(kernel, dim=1, inputs=[a, b, t, out], device=device)

            assert_np_equal(out.numpy(), np.array([test_data.expected]), tol=1e-6)

            if test_data.check_backwards():
                tape.backward(out)

                assert_np_equal(tape.gradients[a].numpy(), np.array([test_data.expected_adj_a]), tol=1e-6)
                assert_np_equal(tape.gradients[b].numpy(), np.array([test_data.expected_adj_b]), tol=1e-6)
                assert_np_equal(tape.gradients[t].numpy(), np.array([test_data.expected_adj_t]), tol=1e-6)


devices = get_test_devices()


class TestSmoothstep(unittest.TestCase):
    pass


add_function_test(TestSmoothstep, "test_smoothstep", test_smoothstep, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
