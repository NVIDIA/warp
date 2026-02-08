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
from typing import Any

import warp as wp
from warp.autograd import (
    gradcheck,
    gradcheck_tape,
    jacobian,
    jacobian_fd,
)
from warp.tests.unittest_utils import *


@wp.kernel
def kernel_3d(
    a: wp.array3d(dtype=Any),
    b: wp.array3d(dtype=Any),
    c: wp.array3d(dtype=Any),
    out1: wp.array3d(dtype=Any),
    out2: wp.array3d(dtype=Any),
):
    i, j, k = wp.tid()
    out1[i, j, k] = a[i, j, k] * b[i, j, k] + c[i, j, k]
    out2[i, j, k] = -a[i, j, k] * b[i, j, k] - c[i, j, k]


wp.overload(
    kernel_3d,
    [
        wp.array3d(dtype=wp.float32),
        wp.array3d(dtype=wp.float32),
        wp.array3d(dtype=wp.float32),
        wp.array3d(dtype=wp.float32),
        wp.array3d(dtype=wp.float32),
    ],
)

wp.overload(
    kernel_3d,
    [
        wp.array3d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
    ],
)


@wp.kernel
def kernel_mixed(
    a: wp.array(dtype=float),
    b: wp.array(dtype=wp.vec3),
    out1: wp.array(dtype=wp.vec2),
    out2: wp.array(dtype=wp.quat),
):
    tid = wp.tid()
    ai, bi = a[tid], b[tid]
    out1[tid] = wp.vec2(ai * wp.length(bi), -ai * wp.dot(bi, wp.vec3(0.1, 1.0, -0.1)))
    out2[tid] = wp.normalize(wp.quat(ai, bi[0], bi[1], bi[2]))


@wp.kernel
def vec_length_kernel(a: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    v = a[tid]
    # instead of wp.length(v), we use a trivial implementation that
    # fails when a division by zero occurs in the backward pass of sqrt
    out[tid] = wp.sqrt(v[0] ** 2.0 + v[1] ** 2.0 + v[2] ** 2.0)


@wp.func
def wrong_grad_func(x: float):
    return x * x


@wp.func_grad(wrong_grad_func)
def adj_wrong_grad_func(x: float, adj: float):
    wp.adjoint[x] -= 2.0 * x * adj


@wp.kernel
def wrong_grad_kernel(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wrong_grad_func(a[tid])


@wp.kernel
def transform_point_kernel(
    transforms: wp.array(dtype=wp.transform),
    points: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out[tid] = wp.transform_point(transforms[tid], points[tid])


def test_gradcheck_3d(test, device, dtype):
    # Adjust tolerances based on dtype precision
    if dtype == wp.float64:
        eps = 1e-5
        atol, rtol = 1e-7, 1e-7
    else:
        eps = 1e-4
        atol, rtol = 1e-2, 1e-2

    a_3d = wp.array([((2.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    b_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    c_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    out1_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    out2_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    jacs_ad = jacobian(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_outputs_per_var=4,
        input_output_mask=[("a", "out1"), ("b", "out2")],
    )

    test.assertEqual(sorted(jacs_ad.keys()), [(0, 0), (1, 1)])
    test.assertEqual(jacs_ad[(0, 0)].shape, (6, 6))
    test.assertEqual(jacs_ad[(1, 1)].shape, (6, 6))
    # all entries beyond the max_outputs_per_var are NaN
    test.assertTrue(np.all(np.isnan(jacs_ad[(0, 0)].numpy()[4:])))
    test.assertTrue(np.all(np.isnan(jacs_ad[(1, 1)].numpy()[4:])))

    jacs_fd = jacobian_fd(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_inputs_per_var=4,
        # use integer indices instead of variable names
        input_output_mask=[(0, 0), (1, 1)],
        eps=eps,
    )

    test.assertEqual(sorted(jacs_fd.keys()), [(0, 0), (1, 1)])
    test.assertEqual(jacs_fd[(0, 0)].shape, (6, 6))
    test.assertEqual(jacs_fd[(1, 1)].shape, (6, 6))
    # all entries beyond the max_inputs_per_var are NaN
    test.assertTrue(np.all(np.isnan(jacs_fd[(0, 0)].numpy()[:, 4:])))
    test.assertTrue(np.all(np.isnan(jacs_fd[(1, 1)].numpy()[:, 4:])))

    # manual gradcheck
    np.testing.assert_allclose(jacs_ad[(0, 0)].numpy()[:4, :4], jacs_fd[(0, 0)].numpy()[:4, :4], atol=atol, rtol=rtol)
    np.testing.assert_allclose(jacs_ad[(1, 1)].numpy()[:4, :4], jacs_fd[(1, 1)].numpy()[:4, :4], atol=atol, rtol=rtol)

    passed = gradcheck(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_inputs_per_var=4,
        max_outputs_per_var=4,
        input_output_mask=[("a", "out1"), ("b", "out2")],
        show_summary=False,
    )
    test.assertTrue(
        passed,
        f"gradcheck failed for kernel_3d (dtype={dtype.__name__}, eps={eps}, atol={atol}, rtol={rtol})",
    )


def test_gradcheck_mixed(test, device):
    a = wp.array([2.0, -1.0], dtype=wp.float32, requires_grad=True, device=device)
    b = wp.array([wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device=device)
    out2 = wp.zeros(2, dtype=wp.quat, requires_grad=True, device=device)

    jacs_ad = jacobian(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2])
    jacs_fd = jacobian_fd(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], eps=1e-4)

    # manual gradcheck
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(jacs_ad[(i, j)].numpy(), jacs_fd[(i, j)].numpy(), atol=1e-2, rtol=1e-2)

    passed = gradcheck(
        kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], raise_exception=False, show_summary=False
    )

    test.assertTrue(passed, "gradcheck failed for kernel_mixed")


def test_gradcheck_nan(test, device):
    a = wp.array([wp.vec3(1.0, 2.0, 3.0), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out = wp.array([0.0, 0.0], dtype=float, requires_grad=True, device=device)

    with test.assertRaises(ValueError):
        gradcheck(vec_length_kernel, dim=a.shape, inputs=[a], outputs=[out], raise_exception=True, show_summary=False)


def test_gradcheck_incorrect(test, device):
    a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device=device)
    out = wp.zeros_like(a)

    with test.assertRaises(ValueError):
        gradcheck(wrong_grad_kernel, dim=a.shape, inputs=[a], outputs=[out], raise_exception=True, show_summary=False)


def test_gradcheck_tape_basic(test, device, dtype):
    a_3d = wp.array([((2.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    b_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    c_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    out1_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    out2_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(kernel_3d, dim=a_3d.shape, inputs=[a_3d, b_3d, c_3d], outputs=[out1_3d, out2_3d], device=device)

    passed = gradcheck_tape(tape, raise_exception=False, show_summary=False)

    test.assertTrue(passed, f"gradcheck_tape failed for kernel_3d (dtype={dtype.__name__})")


def test_gradcheck_tape_mixed(test, device):
    a = wp.array([2.0, -1.0], dtype=wp.float32, requires_grad=True, device=device)
    b = wp.array([wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device=device)
    out2 = wp.zeros(2, dtype=wp.quat, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], device=device)

    passed = gradcheck_tape(tape, raise_exception=False, show_summary=False)

    test.assertTrue(passed, "gradcheck_tape failed for kernel_mixed")


devices = get_test_devices()


class TestGradDebug(unittest.TestCase):
    pass


for dtype in [wp.float32, wp.float64]:
    add_function_test(
        TestGradDebug, f"test_gradcheck_3d_{dtype.__name__}", test_gradcheck_3d, devices=devices, dtype=dtype
    )
    add_function_test(
        TestGradDebug,
        f"test_gradcheck_tape_basic_{dtype.__name__}",
        test_gradcheck_tape_basic,
        devices=devices,
        dtype=dtype,
    )
add_function_test(TestGradDebug, "test_gradcheck_mixed", test_gradcheck_mixed, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_nan", test_gradcheck_nan, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_incorrect", test_gradcheck_incorrect, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_tape_mixed", test_gradcheck_tape_mixed, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
