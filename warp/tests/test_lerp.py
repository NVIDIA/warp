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
        TestData(a=1.0, b=5.0, t=0.75, expected=4.0, expected_adj_a=0.25, expected_adj_b=0.75, expected_adj_t=4.0),
        TestData(a=-2.0, b=5.0, t=0.25, expected=-0.25, expected_adj_a=0.75, expected_adj_b=0.25, expected_adj_t=7.0),
        TestData(a=1.23, b=2.34, t=0.5, expected=1.785, expected_adj_a=0.5, expected_adj_b=0.5, expected_adj_t=1.11),
    ),
    wp.vec2: (TestData(a=[1, 2], b=[3, 4], t=0.5, expected=[2, 3]),),
    wp.vec3: (TestData(a=[1, 2, 3], b=[3, 4, 5], t=0.5, expected=[2, 3, 4]),),
    wp.vec4: (TestData(a=[1, 2, 3, 4], b=[3, 4, 5, 6], t=0.5, expected=[2, 3, 4, 5]),),
    wp.mat22: (TestData(a=[[1, 2], [2, 1]], b=[[3, 4], [4, 3]], t=0.5, expected=[[2, 3], [3, 2]]),),
    wp.mat33: (
        TestData(
            a=[[1, 2, 3], [3, 1, 2], [2, 3, 1]],
            b=[[3, 4, 5], [5, 3, 4], [4, 5, 3]],
            t=0.5,
            expected=[[2, 3, 4], [4, 2, 3], [3, 4, 2]],
        ),
    ),
    wp.mat44: (
        TestData(
            a=[[1, 2, 3, 4], [4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1]],
            b=[[3, 4, 5, 6], [6, 3, 4, 5], [5, 6, 3, 4], [4, 5, 6, 3]],
            t=0.5,
            expected=[[2, 3, 4, 5], [5, 2, 3, 4], [4, 5, 2, 3], [3, 4, 5, 2]],
        ),
    ),
    wp.quat: (TestData(a=[1, 2, 3, 4], b=[3, 4, 5, 6], t=0.5, expected=[2, 3, 4, 5]),),
    wp.transform: (TestData(a=[1, 2, 3, 4, 5, 6, 7], b=[3, 4, 5, 6, 7, 8, 9], t=0.5, expected=[2, 3, 4, 5, 6, 7, 8]),),
    wp.spatial_vector: (TestData(a=[1, 2, 3, 4, 5, 6], b=[3, 4, 5, 6, 7, 8], t=0.5, expected=[2, 3, 4, 5, 6, 7]),),
    wp.spatial_matrix: (
        TestData(
            a=[
                [1, 2, 3, 4, 5, 6],
                [6, 1, 2, 3, 4, 5],
                [5, 6, 1, 2, 3, 4],
                [4, 5, 6, 1, 2, 3],
                [3, 4, 5, 6, 1, 2],
                [2, 3, 4, 5, 6, 1],
            ],
            b=[
                [3, 4, 5, 6, 7, 8],
                [8, 3, 4, 5, 6, 7],
                [7, 8, 3, 4, 5, 6],
                [6, 7, 8, 3, 4, 5],
                [5, 6, 7, 8, 3, 4],
                [4, 5, 6, 7, 8, 3],
            ],
            t=0.5,
            expected=[
                [2, 3, 4, 5, 6, 7],
                [7, 2, 3, 4, 5, 6],
                [6, 7, 2, 3, 4, 5],
                [5, 6, 7, 2, 3, 4],
                [4, 5, 6, 7, 2, 3],
                [3, 4, 5, 6, 7, 2],
            ],
        ),
    ),
}


def test_lerp(test, device):
    def make_kernel_fn(data_type):
        def fn(
            a: wp.array(dtype=data_type),
            b: wp.array(dtype=data_type),
            t: wp.array(dtype=float),
            out: wp.array(dtype=data_type),
        ):
            out[0] = wp.lerp(a[0], b[0], t[0])

        return fn

    for data_type, test_data_set in TEST_DATA.items():
        kernel_fn = make_kernel_fn(data_type)
        kernel = wp.Kernel(func=kernel_fn, key=f"test_lerp_{data_type.__name__}_kernel")

        with test.subTest(data_type=data_type):
            for test_data in test_data_set:
                a = wp.array([test_data.a], dtype=data_type, device=device, requires_grad=True)
                b = wp.array([test_data.b], dtype=data_type, device=device, requires_grad=True)
                t = wp.array([test_data.t], dtype=float, device=device, requires_grad=True)
                out = wp.array(
                    [0] * wp.types.type_length(data_type), dtype=data_type, device=device, requires_grad=True
                )

                with wp.Tape() as tape:
                    wp.launch(kernel, dim=1, inputs=[a, b, t, out], device=device)

                assert_np_equal(out.numpy(), np.array([test_data.expected]), tol=1e-6)

                if test_data.check_backwards():
                    tape.backward(out)

                    assert_np_equal(tape.gradients[a].numpy(), np.array([test_data.expected_adj_a]), tol=1e-6)
                    assert_np_equal(tape.gradients[b].numpy(), np.array([test_data.expected_adj_b]), tol=1e-6)
                    assert_np_equal(tape.gradients[t].numpy(), np.array([test_data.expected_adj_t]), tol=1e-6)


devices = get_test_devices()


class TestLerp(unittest.TestCase):
    pass


add_function_test(TestLerp, "test_lerp", test_lerp, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
