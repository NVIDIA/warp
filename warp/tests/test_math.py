# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, NamedTuple

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


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
    float_values = ScalarFloatValues(degrees=(0.123,), radians=(123.0,))
    float_results_expected = ScalarFloatValues(degrees=7.047381, radians=2.146755)
    adj_float_results_expected = ScalarFloatValues(degrees=57.29578, radians=0.017453)
    for i, values in enumerate(float_values):
        x = wp.array([values[0]], dtype=wp.float32, requires_grad=True, device=device)
        out = wp.array([0.0], dtype=wp.float32, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(scalar_float_kernel, dim=1, inputs=[i, x, out], device=device)

        assert_np_equal(out.numpy(), np.array([float_results_expected[i]]), tol=1e-6)

        tape.backward(out)

        assert_np_equal(tape.gradients[x].numpy(), np.array([adj_float_results_expected[i]]), tol=1e-6)


@wp.kernel
def test_vec_norm_kernel(vs: wp.array(dtype=Any), out: wp.array(dtype=float, ndim=2)):
    tid = wp.tid()
    out[tid, 0] = wp.norm_l1(vs[tid])
    out[tid, 1] = wp.norm_l2(vs[tid])
    out[tid, 2] = wp.norm_huber(vs[tid])
    out[tid, 3] = wp.norm_pseudo_huber(vs[tid])


def test_vec_norm(test, device):
    # ground-truth implementations from SciPy
    def huber(delta, x):
        if x <= delta:
            return 0.5 * x**2
        else:
            return delta * (x - 0.5 * delta)

    def pseudo_huber(delta, x):
        return delta**2 * (np.sqrt(1 + (x / delta) ** 2) - 1)

    v0 = wp.vec3(-2.0, -1.0, -3.0)
    v1 = wp.vec3(2.0, 1.0, 3.0)
    v2 = wp.vec3(0.0, 0.0, 0.0)

    xs = wp.array([v0, v1, v2], dtype=wp.vec3, requires_grad=True, device=device)
    out = wp.empty((len(xs), 4), dtype=wp.float32, requires_grad=True, device=device)

    wp.launch(test_vec_norm_kernel, dim=len(xs), inputs=[xs], outputs=[out], device=device)

    for i, x in enumerate([v0, v1, v2]):
        assert_np_equal(
            out.numpy()[i],
            np.array(
                [
                    np.linalg.norm(x, ord=1),
                    np.linalg.norm(x, ord=2),
                    huber(1.0, wp.length(x)),
                    # note SciPy defines the Pseudo-Huber loss slightly differently
                    pseudo_huber(1.0, wp.length(x)) + 1.0,
                ]
            ),
            tol=1e-6,
        )


devices = get_test_devices()


class TestMath(unittest.TestCase):
    def test_vec_type(self):
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

        self.assertEqual(v[0], w[1], "vec setter error")
        self.assertEqual(v.x, w.y, "vec setter error")

        for x in v[1:]:
            self.assertEqual(x, 1.0, "vec slicing error")

        self.assertEqual(b, c, "vec equality error")

        self.assertEqual(str(v), "[0.0, 1.0, 1.0, 1.0, 1.0]", "vec to string error")

    def test_mat_type(self):
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
        # fmt: off
        b = mat55(
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0,
        )
        # fmt: on

        self.assertEqual(m1, b, "mat element setting error")
        self.assertEqual(m2, a, "mat row setting error")
        self.assertEqual(m1[0, 0], 1.0, "mat element getting error")
        self.assertEqual(m2[0], [1.0, 1.0, 1.0, 1.0, 1.0], "mat row getting error")
        self.assertEqual(
            str(b),
            "[[1.0, 0.0, 0.0, 0.0, 0.0],\n [0.0, 1.0, 0.0, 0.0, 0.0],\n [0.0, 0.0, 1.0, 0.0, 0.0],\n [0.0, 0.0, 0.0, 1.0, 0.0],\n [0.0, 0.0, 0.0, 0.0, 1.0]]",
            "mat to string error",
        )


add_function_test(TestMath, "test_scalar_math", test_scalar_math, devices=devices)
add_function_test(TestMath, "test_vec_norm", test_vec_norm, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
