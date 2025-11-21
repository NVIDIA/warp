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


def test_matvec_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-2,
        np.float32: 5.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat32 = wp._src.types.matrix(shape=(3, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    vec2 = wp._src.types.vector(length=2, dtype=wptype)
    vec4 = wp._src.types.vector(length=4, dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_vec_mul(
        v2: wp.array(dtype=vec2),
        v4: wp.array(dtype=vec4),
        v32: wp.array(dtype=vec2),
        m2: wp.array(dtype=mat22),
        m4: wp.array(dtype=mat44),
        m32: wp.array(dtype=mat32),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = m2[0] * v2[0]
        v4result = m4[0] * v4[0]
        v32result = m32[0] * v32[0]
        v2result_2 = m2[0] @ v2[0]
        v4result_2 = m4[0] @ v4[0]
        v32result_2 = m32[0] @ v32[0]

        idx = 0

        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(2):
            outcomponents[idx] = wptype(2) * v2result[i]
            idx = idx + 1

        for i in range(4):
            outcomponents[idx] = wptype(2) * v4result[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v32result[i]
            idx = idx + 1

        for i in range(2):
            outcomponents[idx] = wptype(2) * v2result_2[i]
            idx = idx + 1

        for i in range(4):
            outcomponents[idx] = wptype(2) * v4result_2[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v32result_2[i]
            idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_vec_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4], dtype), dtype=vec4, requires_grad=True, device=device)
    v32 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * (2 + 4 + 3), dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v2, v4, v32, m2, m4, m32], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:2], 2 * np.matmul(m2.numpy()[0], v2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[2:6], 2 * np.matmul(m4.numpy()[0], v4.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[6:9], 2 * np.matmul(m32.numpy()[0], v32.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[9:11], 2 * np.matmul(m2.numpy()[0], v2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[11:15], 2 * np.matmul(m4.numpy()[0], v4.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[15:18], 2 * np.matmul(m32.numpy()[0], v32.numpy()[0]), tol=5 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, invec, inmat in [(2, v2, m2), (4, v4, m4), (3, v32, m32)]:
            for i in range(dim):
                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[v2, v4, v32, m2, m4, m32], outputs=[outcomponents], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                tape.backward(loss=out)

                assert_np_equal(tape.gradients[invec].numpy()[0], 2 * inmat.numpy()[0, i, :], tol=2 * tol)
                expectedresult = np.zeros(inmat.dtype._shape_, dtype=dtype)
                expectedresult[i, :] = 2 * invec.numpy()[0]
                assert_np_equal(tape.gradients[inmat].numpy()[0], expectedresult, tol=2 * tol)

                tape.zero()

                idx = idx + 1


def test_vecmat_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-2,
        np.float32: 5.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat23 = wp._src.types.matrix(shape=(2, 3), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    vec2 = wp._src.types.vector(length=2, dtype=wptype)
    vec4 = wp._src.types.vector(length=4, dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_vec_mat_mul(
        v2: wp.array(dtype=vec2),
        v4: wp.array(dtype=vec4),
        v32: wp.array(dtype=vec2),
        m2: wp.array(dtype=mat22),
        m4: wp.array(dtype=mat44),
        m23: wp.array(dtype=mat23),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = v2[0] * m2[0]
        v4result = v4[0] * m4[0]
        v32result = v32[0] * m23[0]
        v2result_2 = v2[0] @ m2[0]
        v4result_2 = v4[0] @ m4[0]
        v32result_2 = v32[0] @ m23[0]

        idx = 0

        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(2):
            outcomponents[idx] = wptype(2) * v2result[i]
            idx = idx + 1

        for i in range(4):
            outcomponents[idx] = wptype(2) * v4result[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v32result[i]
            idx = idx + 1

        for i in range(2):
            outcomponents[idx] = wptype(2) * v2result_2[i]
            idx = idx + 1

        for i in range(4):
            outcomponents[idx] = wptype(2) * v4result_2[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v32result_2[i]
            idx = idx + 1

    kernel = getkernel(kernel_cache, check_vec_mat_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4], dtype), dtype=vec4, requires_grad=True, device=device)
    v32 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m23 = wp.array(randvals(rng, [1, 2, 3], dtype), dtype=mat23, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * (2 + 4 + 3), dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v2, v4, v32, m2, m4, m23], outputs=[outcomponents], device=device)

    outcomponents_np = outcomponents.numpy()

    assert_np_equal(outcomponents_np[:2], 2 * np.matmul(v2.numpy()[0], m2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[2:6], 2 * np.matmul(v4.numpy()[0], m4.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[6:9], 2 * np.matmul(v32.numpy()[0], m23.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[9:11], 2 * np.matmul(v2.numpy()[0], m2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[11:15], 2 * np.matmul(v4.numpy()[0], m4.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[15:18], 2 * np.matmul(v32.numpy()[0], m23.numpy()[0]), tol=5 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, inmat, invec in [(2, m2, v2), (4, m4, v4), (3, m23, v32)]:
            for i in range(dim):
                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[v2, v4, v32, m2, m4, m23], outputs=[outcomponents], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                tape.backward(loss=out)

                assert_np_equal(tape.gradients[invec].numpy()[0], 2 * inmat.numpy()[0, :, i], tol=2 * tol)
                expectedresult = np.zeros(inmat.dtype._shape_, dtype=dtype)
                expectedresult[:, i] = 2 * invec.numpy()[0]
                assert_np_equal(tape.gradients[inmat].numpy()[0], expectedresult, tol=2 * tol)

                tape.zero()

                idx = idx + 1


def test_matmat_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-2,
        np.float32: 5.0e-6,
        np.float64: 5.0e-7,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat32 = wp._src.types.matrix(shape=(3, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_mat_mul(
        a2: wp.array(dtype=mat22),
        a4: wp.array(dtype=mat44),
        a32: wp.array(dtype=mat32),
        b2: wp.array(dtype=mat22),
        b4: wp.array(dtype=mat44),
        b32: wp.array(dtype=mat32),
        outcomponents: wp.array(dtype=wptype),
    ):
        c2result = b2[0] * a2[0]
        c4result = b4[0] * a4[0]
        c32result = b32[0] * a2[0]
        c2result_2 = b2[0] @ a2[0]
        c4result_2 = b4[0] @ a4[0]
        c32result_2 = b32[0] @ a2[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c2result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * c4result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c32result[i, j]
                idx = idx + 1

        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c2result_2[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * c4result_2[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c32result_2[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_mat_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    a2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    a4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    a32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=True, device=device)
    b2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    b4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    b32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * (2 * 2 + 4 * 4 + 3 * 2), dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[a2, a4, a32, b2, b4, b32], outputs=[outcomponents], device=device)

    outcomponents_np = outcomponents.numpy()

    assert_np_equal(outcomponents_np[:4].reshape((2, 2)), 2 * np.matmul(b2.numpy()[0], a2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[4:20].reshape((4, 4)), 2 * np.matmul(b4.numpy()[0], a4.numpy()[0]), tol=2 * tol)
    assert_np_equal(outcomponents_np[20:26].reshape((3, 2)), 2 * np.matmul(b32.numpy()[0], a2.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[26:30].reshape((2, 2)), 2 * np.matmul(b2.numpy()[0], a2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[30:46].reshape((4, 4)), 2 * np.matmul(b4.numpy()[0], a4.numpy()[0]), tol=2 * tol)
    assert_np_equal(outcomponents_np[46:52].reshape((3, 2)), 2 * np.matmul(b32.numpy()[0], a2.numpy()[0]), tol=5 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for a, b in [(a2, b2), (a4, b4), (a2, b32)]:
            rows, cols = b.dtype._shape_[0], a.dtype._shape_[1]
            for i in range(rows):
                for j in range(cols):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel, dim=1, inputs=[a2, a4, a32, b2, b4, b32], outputs=[outcomponents], device=device
                        )
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)

                    expected = np.zeros(a.dtype._shape_, dtype=dtype)
                    expected[:, j] = 2 * b.numpy()[0, i, :]
                    assert_np_equal(tape.gradients[a].numpy()[0], expected, tol=10 * tol)

                    expected = np.zeros(b.dtype._shape_, dtype=dtype)
                    expected[i, :] = 2 * a.numpy()[0, :, j]
                    assert_np_equal(tape.gradients[b].numpy()[0], expected, tol=10 * tol)

                    tape.zero()
                    idx = idx + 1


def test_outer_product(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp._src.types.vector(length=2, dtype=wptype)
    vec4 = wp._src.types.vector(length=4, dtype=wptype)
    vec5 = wp._src.types.vector(length=5, dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_outer_product(
        s2: wp.array(dtype=vec2),
        s4: wp.array(dtype=vec4),
        v2: wp.array(dtype=vec2),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        outcomponents: wp.array(dtype=wptype),
    ):
        m22result = wptype(2) * wp.outer(s2[0], v2[0])
        m44result = wptype(2) * wp.outer(s4[0], v4[0])
        m25result = wptype(2) * wp.outer(s2[0], v5[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m22result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m44result[i, j]
                idx = idx + 1

        for i in range(2):
            for j in range(5):
                outcomponents[idx] = m25result[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_outer_product, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4], dtype), dtype=vec4, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4], dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5], dtype), dtype=vec5, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 4 * 4 + 2 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[s2, s4, v2, v4, v5], outputs=[outcomponents], device=device)

    outcomponents_np = outcomponents.numpy()

    assert_np_equal(outcomponents_np[:4].reshape((2, 2)), 2 * s2.numpy()[0, :, None] * v2.numpy()[0, None, :], tol=tol)
    assert_np_equal(
        outcomponents_np[4:20].reshape((4, 4)), 2 * s4.numpy()[0, :, None] * v4.numpy()[0, None, :], tol=10 * tol
    )
    assert_np_equal(
        outcomponents_np[20:].reshape(2, 5), 2 * s2.numpy()[0, :, None] * v5.numpy()[0, None, :], tol=10 * tol
    )

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for s, v in [(s2, v2), (s4, v4), (s2, v5)]:
            rows = s.dtype._length_
            cols = v.dtype._length_
            for i in range(rows):
                for j in range(cols):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[s2, s4, v2, v4, v5], outputs=[outcomponents], device=device)
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)

                    # this component's gonna be s_i * v_j, so its s gradient is gonna be nozero
                    # at the ith component and its v gradient will be nonzero at the jth component:

                    expectedresult = np.zeros((rows), dtype=dtype)
                    expectedresult[i] = 2 * v.numpy()[0, j]
                    assert_np_equal(tape.gradients[s].numpy()[0], expectedresult, tol=10 * tol)

                    expectedresult = np.zeros((cols), dtype=dtype)
                    expectedresult[j] = 2 * s.numpy()[0, i]
                    assert_np_equal(tape.gradients[v].numpy()[0], expectedresult, tol=10 * tol)
                    tape.zero()

                    idx = idx + 1


def test_transpose(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat32 = wp._src.types.matrix(shape=(3, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_transpose(
        m2: wp.array(dtype=mat22),
        m4: wp.array(dtype=mat44),
        m32: wp.array(dtype=mat32),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        mat2 = wptype(2) * wp.transpose(m2[0])
        mat4 = wptype(2) * wp.transpose(m4[0])
        mat32 = wptype(2) * wp.transpose(m32[0])

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = mat2[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = mat4[i, j]
                idx = idx + 1

        for i in range(2):
            for j in range(3):
                outcomponents[idx] = mat32[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_transpose, suffix=dtype.__name__)

    if register_kernels:
        return

    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 4 * 4 + 2 * 3, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m2, m4, m32], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], 2 * m2.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:20], 2 * m4.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[20:], 2 * m32.numpy()[0].T.reshape(-1), tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for input in [m2, m4, m32]:
            rows, cols = input.dtype._shape_
            # Iterate through transposed output shape
            for i in range(cols):
                for j in range(rows):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[m2, m4, m32], outputs=[outcomponents], device=device)
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    # Gradient at transposed position [i, j] flows back to input position [j, i]
                    expectedresult = np.zeros((rows, cols), dtype=dtype)
                    expectedresult[j, i] = 2
                    assert_np_equal(tape.gradients[input].numpy()[0], expectedresult)
                    tape.zero()
                    idx = idx + 1


def test_ddot(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    def check_mat_dot(
        s2: wp.array(dtype=mat22),
        s4: wp.array(dtype=mat44),
        v2: wp.array(dtype=mat22),
        v4: wp.array(dtype=mat44),
        dot2: wp.array(dtype=wptype),
        dot4: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        dot2[0] = wptype(2) * wp.ddot(v2[0], s2[0])
        dot4[0] = wptype(2) * wp.ddot(v4[0], s4[0])

    kernel = getkernel(kernel_cache, check_mat_dot, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    dot2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[s2, s4, v2, v4],
            outputs=[dot2, dot4],
            device=device,
        )

    assert_np_equal(dot2.numpy()[0], 2 * (v2.numpy() * s2.numpy()).sum(), tol=10 * tol)
    assert_np_equal(dot4.numpy()[0], 2 * (v4.numpy() * s4.numpy()).sum(), tol=50 * tol)

    if dtype in np_float_types:
        tape.backward(loss=dot2)
        sgrads = tape.gradients[s2].numpy()[0]
        expected_grads = 2.0 * v2.numpy()[0]
        assert_np_equal(sgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v2].numpy()[0]
        expected_grads = 2.0 * s2.numpy()[0]
        assert_np_equal(vgrads, expected_grads, tol=10 * tol)

        tape.zero()

        tape.backward(loss=dot4)
        sgrads = tape.gradients[s4].numpy()[0]
        expected_grads = 2.0 * v4.numpy()[0]
        assert_np_equal(sgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v4].numpy()[0]
        expected_grads = 2.0 * s4.numpy()[0]
        assert_np_equal(vgrads, expected_grads, tol=10 * tol)

        tape.zero()


def test_trace(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    def check_mat_trace(
        v2: wp.array(dtype=mat22),
        v4: wp.array(dtype=mat44),
        tr2: wp.array(dtype=wptype),
        tr4: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        tr2[0] = wptype(2) * wp.trace(v2[0])
        tr4[0] = wptype(2) * wp.trace(v4[0])

    kernel = getkernel(kernel_cache, check_mat_trace, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    tr2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tr4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[v2, v4], outputs=[tr2, tr4], device=device)

    assert_np_equal(tr2.numpy()[0], 2 * np.trace(v2.numpy()[0]), tol=10 * tol)
    assert_np_equal(tr4.numpy()[0], 2 * np.trace(v4.numpy()[0]), tol=200 * tol)

    if dtype in np_float_types:
        tape.backward(loss=tr2)
        vgrads = tape.gradients[v2].numpy()[0]
        assert_np_equal(vgrads, 2.0 * np.eye(2), tol=10 * tol)
        tape.zero()

        tape.backward(loss=tr4)
        vgrads = tape.gradients[v4].numpy()[0]
        assert_np_equal(vgrads, 2.0 * np.eye(4), tol=10 * tol)
        tape.zero()


devices = get_test_devices()


class TestMatLinalg(unittest.TestCase):
    pass


for dtype in np_scalar_types:
    add_function_test_register_kernel(
        TestMatLinalg,
        f"test_matvec_multiplication_{dtype.__name__}",
        test_matvec_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatLinalg,
        f"test_vecmat_multiplication_{dtype.__name__}",
        test_vecmat_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatLinalg,
        f"test_matmat_multiplication_{dtype.__name__}",
        test_matmat_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatLinalg, f"test_outer_product_{dtype.__name__}", test_outer_product, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatLinalg, f"test_transpose_{dtype.__name__}", test_transpose, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatLinalg, f"test_ddot_{dtype.__name__}", test_ddot, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatLinalg, f"test_trace_{dtype.__name__}", test_trace, devices=devices, dtype=dtype
    )

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
