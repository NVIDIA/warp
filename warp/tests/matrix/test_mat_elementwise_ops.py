# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from warp.tests.matrix.utils import (
    get_select_kernel,
    getkernel,
    np_float_types,
    np_scalar_types,
    randvals,
)
from warp.tests.unittest_utils import *

kernel_cache = {}


def test_scalar_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_scalar_mul(
        s: wp.array(dtype=wptype),
        m2: wp.array(dtype=mat22),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
        outcomponents_rightmul: wp.array(dtype=wptype),
    ):
        m2result = s[0] * m2[0]
        m4result = s[0] * m4[0]

        m2resultright = m2[0] * s[0]
        m4resultright = m4[0] * s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * m2result[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m2resultright[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * m4result[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m4resultright[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_scalar_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 4 * 4, dtype=wptype, requires_grad=True, device=device)
    outcomponents_rightmul = wp.zeros(2 * 2 + 4 * 4, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[s, m2, m4], outputs=[outcomponents, outcomponents_rightmul], device=device)

    sval = s.numpy()[0]
    assert_np_equal(outcomponents.numpy()[:4], 2 * sval * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:20], 2 * sval * m4.numpy().reshape(-1), tol=10 * tol)

    assert_np_equal(outcomponents_rightmul.numpy()[:4], 2 * sval * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents_rightmul.numpy()[4:20], 2 * sval * m4.numpy().reshape(-1), tol=10 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (4, m4)]:
            for i in range(dim):
                for j in range(dim):
                    # test left mul gradient:
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[s, m2, m4],
                            outputs=[outcomponents, outcomponents_rightmul],
                            device=device,
                        )
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim, dim), dtype=dtype)
                    expectedresult[i, j] = 2 * sval
                    assert_np_equal(tape.gradients[input].numpy()[0], expectedresult, tol=10 * tol)
                    assert_np_equal(tape.gradients[s].numpy()[0], 2 * input.numpy()[0, i, j], tol=10 * tol)
                    tape.zero()

                    # test right mul gradient:
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[s, m2, m4],
                            outputs=[outcomponents, outcomponents_rightmul],
                            device=device,
                        )
                        wp.launch(
                            output_select_kernel,
                            dim=1,
                            inputs=[outcomponents_rightmul, idx],
                            outputs=[out],
                            device=device,
                        )
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim, dim), dtype=dtype)
                    expectedresult[i, j] = 2 * sval
                    assert_np_equal(tape.gradients[input].numpy()[0], expectedresult, tol=10 * tol)
                    assert_np_equal(tape.gradients[s].numpy()[0], 2 * input.numpy()[0, i, j], tol=10 * tol)
                    tape.zero()

                    idx = idx + 1


def test_addition(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-2,
        np.float32: 5.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_add(
        s2: wp.array(dtype=mat22),
        s4: wp.array(dtype=mat44),
        v2: wp.array(dtype=mat22),
        v4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = v2[0] + s2[0]
        v4result = v4[0] + s4[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * v2result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * v4result[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_add, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 4 * 4, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[s2, s4, v2, v4],
        outputs=[outcomponents],
        device=device,
    )

    assert_np_equal(outcomponents.numpy()[:4], 2 * (v2.numpy() + s2.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:20], 2 * (v4.numpy() + s4.numpy()).reshape(-1), tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, in1, in2 in [(2, s2, v2), (4, s4, v4)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[s2, s4, v2, v4],
                            outputs=[outcomponents],
                            device=device,
                        )
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim, dim), dtype=dtype)
                    expectedresult[i, j] = 2
                    assert_np_equal(tape.gradients[in2].numpy()[0], expectedresult, tol=10 * tol)
                    expectedresult[i, j] = 2
                    assert_np_equal(tape.gradients[in1].numpy()[0], expectedresult, tol=10 * tol)
                    tape.zero()

                    idx = idx + 1


def test_scalar_division(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_scalar_div(
        s: wp.array(dtype=wptype),
        m2: wp.array(dtype=mat22),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2result = m2[0] / s[0]
        m4result = m4[0] / s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * m4result[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_scalar_div, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 4 * 4, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[s, m2, m4], outputs=[outcomponents], device=device)

    sval = s.numpy()[0]
    if dtype in np_float_types:
        assert_np_equal(outcomponents.numpy()[:4], 2 * m2.numpy().reshape(-1) / sval, tol=tol)
        assert_np_equal(outcomponents.numpy()[4:20], 2 * m4.numpy().reshape(-1) / sval, tol=10 * tol)
    else:
        assert_np_equal(outcomponents.numpy()[:4], 2 * (m2.numpy().reshape(-1) // sval), tol=tol)
        assert_np_equal(outcomponents.numpy()[4:20], 2 * (m4.numpy().reshape(-1) // sval), tol=10 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (4, m4)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[s, m2, m4], outputs=[outcomponents], device=device)
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim, dim), dtype=dtype)
                    expectedresult[i, j] = 2.0 / sval
                    assert_np_equal(tape.gradients[input].numpy()[0], expectedresult, tol=10 * tol)
                    assert_np_equal(
                        tape.gradients[s].numpy()[0], -2 * input.numpy()[0, i, j] / (sval * sval), tol=10 * tol
                    )
                    tape.zero()

                    idx = idx + 1


def test_cw_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_cw_mul(
        s2: wp.array(dtype=mat22),
        s4: wp.array(dtype=mat44),
        v2: wp.array(dtype=mat22),
        v4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = wptype(2) * wp.cw_mul(v2[0], s2[0])
        v4result = wptype(2) * wp.cw_mul(v4[0], s4[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = v2result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = v4result[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_cw_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 4 * 4, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[s2, s4, v2, v4],
        outputs=[outcomponents],
        device=device,
    )

    outcomponents_np = outcomponents.numpy()

    assert_np_equal(outcomponents_np[:4], 2 * (v2.numpy() * s2.numpy()).reshape(-1), tol=50 * tol)
    assert_np_equal(outcomponents_np[4:20], 2 * (v4.numpy() * s4.numpy()).reshape(-1), tol=50 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, in1, in2 in [(2, s2, v2), (4, s4, v4)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[s2, s4, v2, v4],
                            outputs=[outcomponents],
                            device=device,
                        )
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim, dim), dtype=dtype)
                    expectedresult[i, j] = 2 * in1.numpy()[0][i, j]
                    assert_np_equal(tape.gradients[in2].numpy()[0], expectedresult, tol=5 * tol)
                    expectedresult[i, j] = 2 * in2.numpy()[0][i, j]
                    assert_np_equal(tape.gradients[in1].numpy()[0], expectedresult, tol=5 * tol)
                    tape.zero()

                    idx = idx + 1


def test_cw_division(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_cw_div(
        s2: wp.array(dtype=mat22),
        s4: wp.array(dtype=mat44),
        v2: wp.array(dtype=mat22),
        v4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = wptype(2) * wp.cw_div(v2[0], s2[0])
        v4result = wptype(2) * wp.cw_div(v4[0], s4[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = v2result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = v4result[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_cw_div, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = randvals(rng, [1, 2, 2], dtype)
    s4 = randvals(rng, [1, 4, 4], dtype)

    # set denominators to 1 if their magnitudes are small
    # to prevent divide by zero, or overflows if we're testing
    # float16:
    s2[np.abs(s2) < 1.0e-2] = 1
    s4[np.abs(s4) < 1.0e-2] = 1

    s2 = wp.array(s2, dtype=mat22, requires_grad=True, device=device)
    s4 = wp.array(s4, dtype=mat44, requires_grad=True, device=device)

    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 4 * 4, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[s2, s4, v2, v4],
        outputs=[outcomponents],
        device=device,
    )

    if dtype in np_float_types:
        assert_np_equal(outcomponents.numpy()[:4], 2 * (v2.numpy() / s2.numpy()).reshape(-1), tol=50 * tol)
        assert_np_equal(outcomponents.numpy()[4:20], 2 * (v4.numpy() / s4.numpy()).reshape(-1), tol=50 * tol)
    else:
        assert_np_equal(outcomponents.numpy()[:4], 2 * (v2.numpy() // s2.numpy()).reshape(-1), tol=50 * tol)
        assert_np_equal(outcomponents.numpy()[4:20], 2 * (v4.numpy() // s4.numpy()).reshape(-1), tol=50 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, s, v in [(2, s2, v2), (4, s4, v4)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[s2, s4, v2, v4],
                            outputs=[outcomponents],
                            device=device,
                        )
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)

                    # y = v/s
                    # dy/dv = 1.0/s
                    # dy/ds = -v/s^2

                    expectedresult = np.zeros((dim, dim), dtype=dtype)
                    expectedresult[i, j] = 2.0 / (s.numpy()[0, i, j])
                    assert_np_equal(tape.gradients[v].numpy()[0], expectedresult, tol=50 * tol)
                    expectedresult[i, j] = -2.0 * v.numpy()[0, i, j] / (s.numpy()[0, i, j] ** 2)
                    assert_np_equal(
                        tape.gradients[s].numpy()[0], expectedresult, tol=abs(outcomponents.numpy()[idx]) * 50 * tol
                    )
                    tape.zero()

                    idx = idx + 1


devices = get_test_devices()


class TestMatElementwiseOps(unittest.TestCase):
    pass


for dtype in np_scalar_types:
    add_function_test_register_kernel(
        TestMatElementwiseOps,
        f"test_scalar_multiplication_{dtype.__name__}",
        test_scalar_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatElementwiseOps, f"test_addition_{dtype.__name__}", test_addition, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatElementwiseOps,
        f"test_scalar_division_{dtype.__name__}",
        test_scalar_division,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatElementwiseOps,
        f"test_cw_multiplication_{dtype.__name__}",
        test_cw_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatElementwiseOps, f"test_cw_division_{dtype.__name__}", test_cw_division, devices=devices, dtype=dtype
    )

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
