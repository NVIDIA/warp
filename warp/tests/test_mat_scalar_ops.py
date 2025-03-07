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

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

np_signed_int_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.byte,
]

np_unsigned_int_types = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ubyte,
]

np_int_types = np_signed_int_types + np_unsigned_int_types

np_float_types = [np.float16, np.float32, np.float64]

np_scalar_types = np_int_types + np_float_types


def randvals(rng, shape, dtype):
    if dtype in np_float_types:
        return rng.standard_normal(size=shape).astype(dtype)
    elif dtype in [np.int8, np.uint8, np.byte, np.ubyte]:
        return rng.integers(1, high=3, size=shape, dtype=dtype)
    return rng.integers(1, high=5, size=shape, dtype=dtype)


kernel_cache = {}


def getkernel(func, suffix=""):
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key)
    return kernel_cache[key]


def get_select_kernel(dtype):
    def output_select_kernel_fn(
        input: wp.array(dtype=dtype),
        index: int,
        out: wp.array(dtype=dtype),
    ):
        out[0] = input[index]

    return getkernel(output_select_kernel_fn, suffix=dtype.__name__)


def test_arrays(test, device, dtype):
    rng = np.random.default_rng(123)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)
    mat32 = wp.types.matrix(shape=(3, 2), dtype=wptype)

    v2_np = randvals(rng, [10, 2, 2], dtype)
    v3_np = randvals(rng, [10, 3, 3], dtype)
    v4_np = randvals(rng, [10, 4, 4], dtype)
    v5_np = randvals(rng, [10, 5, 5], dtype)
    v32_np = randvals(rng, [10, 3, 2], dtype)

    v2 = wp.array(v2_np, dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(v3_np, dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(v4_np, dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(v5_np, dtype=mat55, requires_grad=True, device=device)
    v32 = wp.array(v32_np, dtype=mat32, requires_grad=True, device=device)

    assert_np_equal(v2.numpy(), v2_np, tol=1.0e-6)
    assert_np_equal(v3.numpy(), v3_np, tol=1.0e-6)
    assert_np_equal(v4.numpy(), v4_np, tol=1.0e-6)
    assert_np_equal(v5.numpy(), v5_np, tol=1.0e-6)
    assert_np_equal(v32.numpy(), v32_np, tol=1.0e-6)

    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)

    v2 = wp.array(v2_np, dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(v3_np, dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(v4_np, dtype=mat44, requires_grad=True, device=device)

    assert_np_equal(v2.numpy(), v2_np, tol=1.0e-6)
    assert_np_equal(v3.numpy(), v3_np, tol=1.0e-6)
    assert_np_equal(v4.numpy(), v4_np, tol=1.0e-6)


def test_components(test, device, dtype):
    # test accessing matrix components from Python - this is especially important
    # for float16, which requires special handling internally

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat23 = wp.types.matrix(shape=(2, 3), dtype=wptype)

    m = mat23(1, 2, 3, 4, 5, 6)

    # test __getitem__ for row vectors
    r0 = m[0]
    r1 = m[1]
    test.assertEqual(r0[0], 1)
    test.assertEqual(r0[1], 2)
    test.assertEqual(r0[2], 3)
    test.assertEqual(r1[0], 4)
    test.assertEqual(r1[1], 5)
    test.assertEqual(r1[2], 6)

    # test __getitem__ for individual components
    test.assertEqual(m[0, 0], 1)
    test.assertEqual(m[0, 1], 2)
    test.assertEqual(m[0, 2], 3)
    test.assertEqual(m[1, 0], 4)
    test.assertEqual(m[1, 1], 5)
    test.assertEqual(m[1, 2], 6)

    # test __setitem__ for row vectors
    m[0] = [7, 8, 9]
    m[1] = [10, 11, 12]
    test.assertEqual(m[0, 0], 7)
    test.assertEqual(m[0, 1], 8)
    test.assertEqual(m[0, 2], 9)
    test.assertEqual(m[1, 0], 10)
    test.assertEqual(m[1, 1], 11)
    test.assertEqual(m[1, 2], 12)

    # test __setitem__ for individual components
    m[0, 0] = 13
    m[0, 1] = 14
    m[0, 2] = 15
    m[1, 0] = 16
    m[1, 1] = 17
    m[1, 2] = 18
    test.assertEqual(m[0, 0], 13)
    test.assertEqual(m[0, 1], 14)
    test.assertEqual(m[0, 2], 15)
    test.assertEqual(m[1, 0], 16)
    test.assertEqual(m[1, 1], 17)
    test.assertEqual(m[1, 2], 18)


def test_constants(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)
    mat32 = wp.types.matrix(shape=(3, 2), dtype=wptype)

    cm22 = wp.constant(mat22(22))
    cm33 = wp.constant(mat33(33))
    cm44 = wp.constant(mat44(44))
    cm55 = wp.constant(mat55(55))
    cm32 = wp.constant(mat32(32))

    def check_matrix_constants():
        wp.expect_eq(cm22, mat22(wptype(22)))
        wp.expect_eq(cm33, mat33(wptype(33)))
        wp.expect_eq(cm44, mat44(wptype(44)))
        wp.expect_eq(cm55, mat55(wptype(55)))
        wp.expect_eq(cm32, mat32(wptype(32)))

    kernel = getkernel(check_matrix_constants, suffix=dtype.__name__)

    if register_kernels:
        return


def test_constructors(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_scalar_mat_constructor(
        input: wp.array(dtype=wptype),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        m2result = wptype(2) * mat22(input[0])
        m3result = wptype(2) * mat33(input[0])
        m4result = wptype(2) * mat44(input[0])
        m5result = wptype(2) * mat55(input[0])

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5result[i, j]
                idx = idx + 1

    def check_component_mat_constructor(
        input: wp.array(dtype=wptype),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        m2result = wptype(2) * mat22(input[0], input[1], input[2], input[3])
        m3result = wptype(2) * mat33(
            input[4],
            input[5],
            input[6],
            input[7],
            input[8],
            input[9],
            input[10],
            input[11],
            input[12],
        )
        m4result = wptype(2) * mat44(
            input[13],
            input[14],
            input[15],
            input[16],
            input[17],
            input[18],
            input[19],
            input[20],
            input[21],
            input[22],
            input[23],
            input[24],
            input[25],
            input[26],
            input[27],
            input[28],
        )
        m5result = wptype(2) * mat55(
            input[29],
            input[30],
            input[31],
            input[32],
            input[33],
            input[34],
            input[35],
            input[36],
            input[37],
            input[38],
            input[39],
            input[40],
            input[41],
            input[42],
            input[43],
            input[44],
            input[45],
            input[46],
            input[47],
            input[48],
            input[49],
            input[50],
            input[51],
            input[52],
            input[53],
        )

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5result[i, j]
                idx = idx + 1

    def check_vector_mat_constructor(
        input: wp.array(dtype=wptype),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        m2result = wptype(2) * wp.matrix_from_cols(vec2(input[0], input[2]), vec2(input[1], input[3]))
        m3result = wptype(2) * wp.matrix_from_cols(
            vec3(input[4], input[7], input[10]),
            vec3(input[5], input[8], input[11]),
            vec3(input[6], input[9], input[12]),
        )
        m4result = wptype(2) * wp.matrix_from_cols(
            vec4(input[13], input[17], input[21], input[25]),
            vec4(input[14], input[18], input[22], input[26]),
            vec4(input[15], input[19], input[23], input[27]),
            vec4(input[16], input[20], input[24], input[28]),
        )
        m5result = wptype(2) * wp.matrix_from_cols(
            vec5(input[29], input[34], input[39], input[44], input[49]),
            vec5(input[30], input[35], input[40], input[45], input[50]),
            vec5(input[31], input[36], input[41], input[46], input[51]),
            vec5(input[32], input[37], input[42], input[47], input[52]),
            vec5(input[33], input[38], input[43], input[48], input[53]),
        )

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5result[i, j]
                idx = idx + 1

    kernel = getkernel(check_scalar_mat_constructor, suffix=dtype.__name__)
    compkernel = getkernel(check_component_mat_constructor, suffix=dtype.__name__)
    veckernel = getkernel(check_vector_mat_constructor, suffix=dtype.__name__)

    if register_kernels:
        return

    input = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    val = input.numpy()[0]
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[input], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], 2 * val * np.ones(2 * 2), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 2 * val * np.ones(3 * 3), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], 2 * val * np.ones(4 * 4), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], 2 * val * np.ones(5 * 5), tol=tol)

    if dtype in np_float_types:
        for idx in range(len(outcomponents)):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[input], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
            tape.backward(loss=out)
            test.assertEqual(tape.gradients[input].numpy()[0], 2)
            tape.zero()

    input = wp.array(randvals(rng, [2 * 2 + 3 * 3 + 4 * 4 + 5 * 5], dtype), requires_grad=True, device=device)

    wp.launch(compkernel, dim=1, inputs=[input], outputs=[outcomponents], device=device)
    assert_np_equal(2 * input.numpy(), outcomponents.numpy(), tol=10 * tol)

    if dtype in np_float_types:
        for idx in range(len(outcomponents)):
            tape = wp.Tape()
            with tape:
                wp.launch(compkernel, dim=1, inputs=[input], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedgrads = np.zeros(len(input))
            expectedgrads[idx] = 2
            assert_np_equal(tape.gradients[input].numpy(), expectedgrads)
            tape.zero()

    wp.launch(veckernel, dim=1, inputs=[input], outputs=[outcomponents], device=device)
    assert_np_equal(2 * input.numpy(), outcomponents.numpy(), tol=10 * tol)

    if dtype in np_float_types:
        for idx in range(len(outcomponents)):
            tape = wp.Tape()
            with tape:
                wp.launch(veckernel, dim=1, inputs=[input], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedgrads = np.zeros(len(input))
            expectedgrads[idx] = 2
            assert_np_equal(tape.gradients[input].numpy(), expectedgrads)
            tape.zero()


def test_anon_type_instance(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_scalar_init(
        input: wp.array(dtype=wptype),
        output: wp.array(dtype=wptype),
    ):
        m2result = wp.matrix(input[0], shape=(2, 2))
        m3result = wp.matrix(input[1], shape=(3, 3))
        m4result = wp.matrix(input[2], shape=(4, 4))
        m5result = wp.matrix(input[3], shape=(5, 5))
        m32result = wp.matrix(input[4], shape=(3, 2))

        idx = 0
        for i in range(2):
            for j in range(2):
                output[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1
        for i in range(3):
            for j in range(3):
                output[idx] = wptype(2) * m3result[i, j]
                idx = idx + 1
        for i in range(4):
            for j in range(4):
                output[idx] = wptype(2) * m4result[i, j]
                idx = idx + 1
        for i in range(5):
            for j in range(5):
                output[idx] = wptype(2) * m5result[i, j]
                idx = idx + 1
        for i in range(3):
            for j in range(2):
                output[idx] = wptype(2) * m32result[i, j]
                idx = idx + 1

    def check_component_init(
        input: wp.array(dtype=wptype),
        output: wp.array(dtype=wptype),
    ):
        m2result = wp.matrix(input[0], input[1], input[2], input[3], shape=(2, 2))
        m3result = wp.matrix(
            input[4], input[5], input[6], input[7], input[8], input[9], input[10], input[11], input[12], shape=(3, 3)
        )
        m4result = wp.matrix(
            input[13],
            input[14],
            input[15],
            input[16],
            input[17],
            input[18],
            input[19],
            input[20],
            input[21],
            input[22],
            input[23],
            input[24],
            input[25],
            input[26],
            input[27],
            input[28],
            shape=(4, 4),
        )
        m5result = wp.matrix(
            input[29],
            input[30],
            input[31],
            input[32],
            input[33],
            input[34],
            input[35],
            input[36],
            input[37],
            input[38],
            input[39],
            input[40],
            input[41],
            input[42],
            input[43],
            input[44],
            input[45],
            input[46],
            input[47],
            input[48],
            input[49],
            input[50],
            input[51],
            input[52],
            input[53],
            shape=(5, 5),
        )
        m32result = wp.matrix(input[54], input[55], input[56], input[57], input[58], input[59], shape=(3, 2))

        idx = 0
        for i in range(2):
            for j in range(2):
                output[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1
        for i in range(3):
            for j in range(3):
                output[idx] = wptype(2) * m3result[i, j]
                idx = idx + 1
        for i in range(4):
            for j in range(4):
                output[idx] = wptype(2) * m4result[i, j]
                idx = idx + 1
        for i in range(5):
            for j in range(5):
                output[idx] = wptype(2) * m5result[i, j]
                idx = idx + 1
        for i in range(3):
            for j in range(2):
                output[idx] = wptype(2) * m32result[i, j]
                idx = idx + 1

    scalar_kernel = getkernel(check_scalar_init, suffix=dtype.__name__)
    component_kernel = getkernel(check_component_init, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    input = wp.array(randvals(rng, [5], dtype), requires_grad=True, device=device)
    output = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 3 * 2, dtype=wptype, requires_grad=True, device=device)

    wp.launch(scalar_kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy()[:4], 2 * np.array([input.numpy()[0]] * 2 * 2), tol=1.0e-6)
    assert_np_equal(output.numpy()[4:13], 2 * np.array([input.numpy()[1]] * 3 * 3), tol=1.0e-6)
    assert_np_equal(output.numpy()[13:29], 2 * np.array([input.numpy()[2]] * 4 * 4), tol=1.0e-6)
    assert_np_equal(output.numpy()[29:54], 2 * np.array([input.numpy()[3]] * 5 * 5), tol=1.0e-6)
    assert_np_equal(output.numpy()[54:], 2 * np.array([input.numpy()[4]] * 3 * 2), tol=1.0e-6)

    if dtype in np_float_types:
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for i in range(len(output)):
            tape = wp.Tape()
            with tape:
                wp.launch(scalar_kernel, dim=1, inputs=[input], outputs=[output], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[output, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(input.numpy())
            if i < 4:
                expected[0] = 2
            elif i < 13:
                expected[1] = 2
            elif i < 29:
                expected[2] = 2
            elif i < 54:
                expected[3] = 2
            else:
                expected[4] = 2

            assert_np_equal(tape.gradients[input].numpy(), expected, tol=tol)

            tape.reset()
            tape.zero()

    input = wp.array(randvals(rng, [2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 3 * 2], dtype), requires_grad=True, device=device)
    output = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 3 * 2, dtype=wptype, requires_grad=True, device=device)

    wp.launch(component_kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy(), 2 * input.numpy(), tol=1.0e-6)

    if dtype in np_float_types:
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for i in range(len(output)):
            tape = wp.Tape()
            with tape:
                wp.launch(component_kernel, dim=1, inputs=[input], outputs=[output], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[output, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(input.numpy())
            expected[i] = 2

            assert_np_equal(tape.gradients[input].numpy(), expected, tol=tol)

            tape.reset()
            tape.zero()


def test_identity(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_identity_mat(
        output: wp.array(dtype=wptype),
    ):
        m2result = wp.identity(dtype=wptype, n=2)
        m3result = wp.identity(dtype=wptype, n=3)
        m4result = wp.identity(dtype=wptype, n=4)
        m5result = wp.identity(dtype=wptype, n=5)

        idx = 0
        for i in range(2):
            for j in range(2):
                output[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1
        for i in range(3):
            for j in range(3):
                output[idx] = wptype(2) * m3result[i, j]
                idx = idx + 1
        for i in range(4):
            for j in range(4):
                output[idx] = wptype(2) * m4result[i, j]
                idx = idx + 1
        for i in range(5):
            for j in range(5):
                output[idx] = wptype(2) * m5result[i, j]
                idx = idx + 1

    id_kernel = getkernel(check_identity_mat, suffix=dtype.__name__)

    if register_kernels:
        return

    output = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)
    wp.launch(id_kernel, dim=1, inputs=[], outputs=[output], device=device)
    assert_np_equal(output.numpy()[:4], 2 * np.eye(2), tol=1.0e-6)
    assert_np_equal(output.numpy()[4:13], 2 * np.eye(3), tol=1.0e-6)
    assert_np_equal(output.numpy()[13:29], 2 * np.eye(4), tol=1.0e-6)
    assert_np_equal(output.numpy()[29:], 2 * np.eye(5), tol=1.0e-6)


def test_indexing(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_indexing(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * m2[0][i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * m3[0][i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * m4[0][i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * m5[0][i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_indexing, suffix=dtype.__name__)

    if register_kernels:
        return

    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m2, m3, m4, m5], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], 2 * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 2 * m3.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], 2 * m4.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], 2 * m5.numpy().reshape(-1), tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (3, m3), (4, m4), (5, m5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[m2, m3, m4, m5], outputs=[outcomponents], device=device)
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim, dim), dtype=dtype)
                    expectedresult[i, j] = 2
                    assert_np_equal(tape.gradients[input].numpy()[0], expectedresult)
                    tape.zero()
                    idx = idx + 1


def test_equality(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    def check_mat_equality():
        wp.expect_eq(
            mat22(wptype(1.0), wptype(2.0), wptype(3.0), wptype(4.0)),
            mat22(wptype(1.0), wptype(2.0), wptype(3.0), wptype(4.0)),
        )
        wp.expect_neq(
            mat22(wptype(1.0), wptype(2.0), wptype(3.0), -wptype(4.0)),
            mat22(wptype(1.0), wptype(2.0), wptype(3.0), wptype(4.0)),
        )

        wp.expect_eq(
            mat33(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
            ),
            mat33(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
            ),
        )
        wp.expect_neq(
            mat33(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
            ),
            mat33(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                -wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
            ),
        )

        wp.expect_eq(
            mat44(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
                wptype(10.0),
                wptype(11.0),
                wptype(12.0),
                wptype(13.0),
                wptype(14.0),
                wptype(15.0),
                wptype(16.0),
            ),
            mat44(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
                wptype(10.0),
                wptype(11.0),
                wptype(12.0),
                wptype(13.0),
                wptype(14.0),
                wptype(15.0),
                wptype(16.0),
            ),
        )

        wp.expect_neq(
            mat44(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
                wptype(10.0),
                wptype(11.0),
                wptype(12.0),
                wptype(13.0),
                wptype(14.0),
                wptype(15.0),
                wptype(16.0),
            ),
            mat44(
                -wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
                wptype(10.0),
                wptype(11.0),
                wptype(12.0),
                wptype(13.0),
                wptype(14.0),
                wptype(15.0),
                wptype(16.0),
            ),
        )

        wp.expect_eq(
            mat55(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
                wptype(10.0),
                wptype(11.0),
                wptype(12.0),
                wptype(13.0),
                wptype(14.0),
                wptype(15.0),
                wptype(16.0),
                wptype(17.0),
                wptype(18.0),
                wptype(19.0),
                wptype(20.0),
                wptype(21.0),
                wptype(22.0),
                wptype(23.0),
                wptype(24.0),
                wptype(25.0),
            ),
            mat55(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
                wptype(10.0),
                wptype(11.0),
                wptype(12.0),
                wptype(13.0),
                wptype(14.0),
                wptype(15.0),
                wptype(16.0),
                wptype(17.0),
                wptype(18.0),
                wptype(19.0),
                wptype(20.0),
                wptype(21.0),
                wptype(22.0),
                wptype(23.0),
                wptype(24.0),
                wptype(25.0),
            ),
        )

        wp.expect_neq(
            mat55(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
                wptype(10.0),
                wptype(11.0),
                wptype(12.0),
                wptype(13.0),
                wptype(14.0),
                wptype(15.0),
                wptype(16.0),
                wptype(17.0),
                wptype(18.0),
                wptype(19.0),
                wptype(20.0),
                wptype(21.0),
                wptype(22.0),
                wptype(23.0),
                wptype(24.0),
                wptype(25.0),
            ),
            mat55(
                wptype(1.0),
                wptype(2.0),
                wptype(3.0),
                wptype(4.0),
                wptype(5.0),
                wptype(6.0),
                wptype(7.0),
                wptype(8.0),
                wptype(9.0),
                wptype(10.0),
                wptype(11.0),
                wptype(12.0),
                wptype(13.0),
                wptype(14.0),
                wptype(15.0),
                wptype(16.0),
                -wptype(17.0),
                wptype(18.0),
                wptype(19.0),
                wptype(20.0),
                wptype(21.0),
                wptype(22.0),
                wptype(23.0),
                wptype(24.0),
                wptype(25.0),
            ),
        )

    kernel = getkernel(check_mat_equality, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=1, inputs=[], outputs=[], device=device)


def test_scalar_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_scalar_mul(
        s: wp.array(dtype=wptype),
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
        outcomponents_rightmul: wp.array(dtype=wptype),
    ):
        m2result = s[0] * m2[0]
        m3result = s[0] * m3[0]
        m4result = s[0] * m4[0]
        m5result = s[0] * m5[0]

        m2resultright = m2[0] * s[0]
        m3resultright = m3[0] * s[0]
        m4resultright = m4[0] * s[0]
        m5resultright = m5[0] * s[0]

        m2result_2 = s[0] * m2[0]
        m3result_2 = s[0] * m3[0]
        m4result_2 = s[0] * m4[0]
        m5result_2 = s[0] * m5[0]

        m2resultright_2 = m2[0] * s[0]
        m3resultright_2 = m3[0] * s[0]
        m4resultright_2 = m4[0] * s[0]
        m5resultright_2 = m5[0] * s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * m2result[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m2resultright[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * m3result[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m3resultright[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * m4result[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m4resultright[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * m5result[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m5resultright[i, j]
                idx = idx + 1

        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * m2result_2[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m2resultright_2[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * m3result_2[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m3resultright_2[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * m4result_2[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m4resultright_2[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * m5result_2[i, j]
                outcomponents_rightmul[idx] = wptype(2) * m5resultright_2[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_scalar_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * (2 * 2 + 3 * 3 + 4 * 4 + 5 * 5), dtype=wptype, requires_grad=True, device=device)
    outcomponents_rightmul = wp.zeros(
        2 * (2 * 2 + 3 * 3 + 4 * 4 + 5 * 5), dtype=wptype, requires_grad=True, device=device
    )

    wp.launch(kernel, dim=1, inputs=[s, m2, m3, m4, m5], outputs=[outcomponents, outcomponents_rightmul], device=device)

    sval = s.numpy()[0]
    assert_np_equal(outcomponents.numpy()[:4], 2 * sval * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 2 * sval * m3.numpy().reshape(-1), tol=10 * tol)
    assert_np_equal(outcomponents.numpy()[13:29], 2 * sval * m4.numpy().reshape(-1), tol=10 * tol)
    assert_np_equal(outcomponents.numpy()[29:54], 2 * sval * m5.numpy().reshape(-1), tol=10 * tol)

    assert_np_equal(outcomponents_rightmul.numpy()[:4], 2 * sval * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents_rightmul.numpy()[4:13], 2 * sval * m3.numpy().reshape(-1), tol=10 * tol)
    assert_np_equal(outcomponents_rightmul.numpy()[13:29], 2 * sval * m4.numpy().reshape(-1), tol=10 * tol)
    assert_np_equal(outcomponents_rightmul.numpy()[29:54], 2 * sval * m5.numpy().reshape(-1), tol=10 * tol)

    assert_np_equal(outcomponents.numpy()[54:58], 2 * sval * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[58:67], 2 * sval * m3.numpy().reshape(-1), tol=10 * tol)
    assert_np_equal(outcomponents.numpy()[67:83], 2 * sval * m4.numpy().reshape(-1), tol=10 * tol)
    assert_np_equal(outcomponents.numpy()[83:108], 2 * sval * m5.numpy().reshape(-1), tol=10 * tol)

    assert_np_equal(outcomponents_rightmul.numpy()[54:58], 2 * sval * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents_rightmul.numpy()[58:67], 2 * sval * m3.numpy().reshape(-1), tol=10 * tol)
    assert_np_equal(outcomponents_rightmul.numpy()[67:83], 2 * sval * m4.numpy().reshape(-1), tol=10 * tol)
    assert_np_equal(outcomponents_rightmul.numpy()[83:108], 2 * sval * m5.numpy().reshape(-1), tol=10 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (3, m3), (4, m4), (5, m5)]:
            for i in range(dim):
                for j in range(dim):
                    # test left mul gradient:
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[s, m2, m3, m4, m5],
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
                            inputs=[s, m2, m3, m4, m5],
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


def test_matvec_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-2,
        np.float32: 5.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat32 = wp.types.matrix(shape=(3, 2), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_vec_mul(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v32: wp.array(dtype=vec2),
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        m32: wp.array(dtype=mat32),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = m2[0] * v2[0]
        v3result = m3[0] * v3[0]
        v4result = m4[0] * v4[0]
        v5result = m5[0] * v5[0]
        v32result = m32[0] * v32[0]
        v2result_2 = m2[0] @ v2[0]
        v3result_2 = m3[0] @ v3[0]
        v4result_2 = m4[0] @ v4[0]
        v5result_2 = m5[0] @ v5[0]
        v32result_2 = m32[0] @ v32[0]

        idx = 0

        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(2):
            outcomponents[idx] = wptype(2) * v2result[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v3result[i]
            idx = idx + 1

        for i in range(4):
            outcomponents[idx] = wptype(2) * v4result[i]
            idx = idx + 1

        for i in range(5):
            outcomponents[idx] = wptype(2) * v5result[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v32result[i]
            idx = idx + 1

        for i in range(2):
            outcomponents[idx] = wptype(2) * v2result_2[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v3result_2[i]
            idx = idx + 1

        for i in range(4):
            outcomponents[idx] = wptype(2) * v4result_2[i]
            idx = idx + 1

        for i in range(5):
            outcomponents[idx] = wptype(2) * v5result_2[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v32result_2[i]
            idx = idx + 1

    kernel = getkernel(check_mat_vec_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4], dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5], dtype), dtype=vec5, requires_grad=True, device=device)
    v32 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    m32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * (2 + 3 + 4 + 5 + 3), dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v2, v3, v4, v5, v32, m2, m3, m4, m5, m32], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:2], 2 * np.matmul(m2.numpy()[0], v2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[2:5], 2 * np.matmul(m3.numpy()[0], v3.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[5:9], 2 * np.matmul(m4.numpy()[0], v4.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[9:14], 2 * np.matmul(m5.numpy()[0], v5.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[14:17], 2 * np.matmul(m32.numpy()[0], v32.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[17:19], 2 * np.matmul(m2.numpy()[0], v2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[19:22], 2 * np.matmul(m3.numpy()[0], v3.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[22:26], 2 * np.matmul(m4.numpy()[0], v4.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[26:31], 2 * np.matmul(m5.numpy()[0], v5.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[31:34], 2 * np.matmul(m32.numpy()[0], v32.numpy()[0]), tol=5 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, invec, inmat in [(2, v2, m2), (3, v3, m3), (4, v4, m4), (5, v5, m5), (3, v32, m32)]:
            for i in range(dim):
                tape = wp.Tape()
                with tape:
                    wp.launch(
                        kernel,
                        dim=1,
                        inputs=[v2, v3, v4, v5, v32, m2, m3, m4, m5, m32],
                        outputs=[outcomponents],
                        device=device,
                    )
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

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat23 = wp.types.matrix(shape=(2, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_vec_mat_mul(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v32: wp.array(dtype=vec2),
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        m23: wp.array(dtype=mat23),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = v2[0] * m2[0]
        v3result = v3[0] * m3[0]
        v4result = v4[0] * m4[0]
        v5result = v5[0] * m5[0]
        v32result = v32[0] * m23[0]
        v2result_2 = v2[0] @ m2[0]
        v3result_2 = v3[0] @ m3[0]
        v4result_2 = v4[0] @ m4[0]
        v5result_2 = v5[0] @ m5[0]
        v32result_2 = v32[0] @ m23[0]

        idx = 0

        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(2):
            outcomponents[idx] = wptype(2) * v2result[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v3result[i]
            idx = idx + 1

        for i in range(4):
            outcomponents[idx] = wptype(2) * v4result[i]
            idx = idx + 1

        for i in range(5):
            outcomponents[idx] = wptype(2) * v5result[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v32result[i]
            idx = idx + 1

        for i in range(2):
            outcomponents[idx] = wptype(2) * v2result_2[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v3result_2[i]
            idx = idx + 1

        for i in range(4):
            outcomponents[idx] = wptype(2) * v4result_2[i]
            idx = idx + 1

        for i in range(5):
            outcomponents[idx] = wptype(2) * v5result_2[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * v32result_2[i]
            idx = idx + 1

    kernel = getkernel(check_vec_mat_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4], dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5], dtype), dtype=vec5, requires_grad=True, device=device)
    v32 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    m23 = wp.array(randvals(rng, [1, 2, 3], dtype), dtype=mat23, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * (2 + 3 + 4 + 5 + 3), dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v2, v3, v4, v5, v32, m2, m3, m4, m5, m23], outputs=[outcomponents], device=device)

    outcomponents_np = outcomponents.numpy()

    assert_np_equal(outcomponents_np[:2], 2 * np.matmul(v2.numpy()[0], m2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[2:5], 2 * np.matmul(v3.numpy()[0], m3.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[5:9], 2 * np.matmul(v4.numpy()[0], m4.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[9:14], 2 * np.matmul(v5.numpy()[0], m5.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[14:17], 2 * np.matmul(v32.numpy()[0], m23.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[17:19], 2 * np.matmul(v2.numpy()[0], m2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[19:22], 2 * np.matmul(v3.numpy()[0], m3.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[22:26], 2 * np.matmul(v4.numpy()[0], m4.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[26:31], 2 * np.matmul(v5.numpy()[0], m5.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[31:34], 2 * np.matmul(v32.numpy()[0], m23.numpy()[0]), tol=5 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, inmat, invec in [(2, m2, v2), (3, m3, v3), (4, m4, v4), (5, m5, v5), (3, m23, v32)]:
            for i in range(dim):
                tape = wp.Tape()
                with tape:
                    wp.launch(
                        kernel,
                        dim=1,
                        inputs=[v2, v3, v4, v5, v32, m2, m3, m4, m5, m23],
                        outputs=[outcomponents],
                        device=device,
                    )
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

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat32 = wp.types.matrix(shape=(3, 2), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_mat_mul(
        a2: wp.array(dtype=mat22),
        a3: wp.array(dtype=mat33),
        a4: wp.array(dtype=mat44),
        a5: wp.array(dtype=mat55),
        a32: wp.array(dtype=mat32),
        b2: wp.array(dtype=mat22),
        b3: wp.array(dtype=mat33),
        b4: wp.array(dtype=mat44),
        b5: wp.array(dtype=mat55),
        b32: wp.array(dtype=mat32),
        outcomponents: wp.array(dtype=wptype),
    ):
        c2result = b2[0] * a2[0]
        c3result = b3[0] * a3[0]
        c4result = b4[0] * a4[0]
        c5result = b5[0] * a5[0]
        c32result = b32[0] * a2[0]
        c32result2 = b3[0] * a32[0]
        c2result_2 = b2[0] @ a2[0]
        c3result_2 = b3[0] @ a3[0]
        c4result_2 = b4[0] @ a4[0]
        c5result_2 = b5[0] @ a5[0]
        c32result_2 = b32[0] @ a2[0]
        c32result2_2 = b3[0] @ a32[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * c3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * c4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * c5result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c32result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c32result2[i, j]
                idx = idx + 1

        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c2result_2[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * c3result_2[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * c4result_2[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * c5result_2[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c32result_2[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(2):
                outcomponents[idx] = wptype(2) * c32result2_2[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_mat_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    v32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=True, device=device)
    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    m32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=True, device=device)
    outcomponents = wp.zeros(
        2 * (2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 3 * 2 + 3 * 2), dtype=wptype, requires_grad=True, device=device
    )

    wp.launch(kernel, dim=1, inputs=[v2, v3, v4, v5, v32, m2, m3, m4, m5, m32], outputs=[outcomponents], device=device)

    outcomponents_np = outcomponents.numpy()

    assert_np_equal(outcomponents_np[:4].reshape((2, 2)), 2 * np.matmul(m2.numpy()[0], v2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[4:13].reshape((3, 3)), 2 * np.matmul(m3.numpy()[0], v3.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[13:29].reshape((4, 4)), 2 * np.matmul(m4.numpy()[0], v4.numpy()[0]), tol=2 * tol)
    assert_np_equal(outcomponents_np[29:54].reshape((5, 5)), 2 * np.matmul(m5.numpy()[0], v5.numpy()[0]), tol=10 * tol)
    assert_np_equal(outcomponents_np[54:60].reshape((3, 2)), 2 * np.matmul(m32.numpy()[0], v2.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[60:66].reshape((3, 2)), 2 * np.matmul(m3.numpy()[0], v32.numpy()[0]), tol=5 * tol)
    assert_np_equal(outcomponents_np[66:70].reshape((2, 2)), 2 * np.matmul(m2.numpy()[0], v2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[70:79].reshape((3, 3)), 2 * np.matmul(m3.numpy()[0], v3.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents_np[79:95].reshape((4, 4)), 2 * np.matmul(m4.numpy()[0], v4.numpy()[0]), tol=2 * tol)
    assert_np_equal(outcomponents_np[95:120].reshape((5, 5)), 2 * np.matmul(m5.numpy()[0], v5.numpy()[0]), tol=10 * tol)
    assert_np_equal(
        outcomponents_np[120:126].reshape((3, 2)), 2 * np.matmul(m32.numpy()[0], v2.numpy()[0]), tol=5 * tol
    )
    assert_np_equal(
        outcomponents_np[126:132].reshape((3, 2)), 2 * np.matmul(m3.numpy()[0], v32.numpy()[0]), tol=5 * tol
    )

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for v, m in [(v2, m2), (v3, m3), (v4, m4), (v5, m5), (v2, m32), (v32, m3)]:
            rows, cols = m.dtype._shape_[0], v.dtype._shape_[1]
            for i in range(rows):
                for j in range(cols):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[v2, v3, v4, v5, v32, m2, m3, m4, m5, m32],
                            outputs=[outcomponents],
                            device=device,
                        )
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)

                    expected = np.zeros(v.dtype._shape_, dtype=dtype)
                    expected[:, j] = 2 * m.numpy()[0, i, :]
                    assert_np_equal(tape.gradients[v].numpy()[0], expected, tol=10 * tol)

                    expected = np.zeros(m.dtype._shape_, dtype=dtype)
                    expected[i, :] = 2 * v.numpy()[0, :, j]
                    assert_np_equal(tape.gradients[m].numpy()[0], expected, tol=10 * tol)

                    tape.zero()
                    idx = idx + 1


def test_cw_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_cw_mul(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = wptype(2) * wp.cw_mul(v2[0], s2[0])
        v3result = wptype(2) * wp.cw_mul(v3[0], s3[0])
        v4result = wptype(2) * wp.cw_mul(v4[0], s4[0])
        v5result = wptype(2) * wp.cw_mul(v5[0], s5[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = v2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = v3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = v4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = v5result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_cw_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            s2,
            s3,
            s4,
            s5,
            v2,
            v3,
            v4,
            v5,
        ],
        outputs=[outcomponents],
        device=device,
    )

    outcomponents_np = outcomponents.numpy()

    assert_np_equal(outcomponents_np[:4], 2 * (v2.numpy() * s2.numpy()).reshape(-1), tol=50 * tol)
    assert_np_equal(outcomponents_np[4:13], 2 * (v3.numpy() * s3.numpy()).reshape(-1), tol=50 * tol)
    assert_np_equal(outcomponents_np[13:29], 2 * (v4.numpy() * s4.numpy()).reshape(-1), tol=50 * tol)
    assert_np_equal(outcomponents_np[29:54], 2 * (v5.numpy() * s5.numpy()).reshape(-1), tol=50 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, in1, in2 in [(2, s2, v2), (3, s3, v3), (4, s4, v4), (5, s5, v5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[
                                s2,
                                s3,
                                s4,
                                s5,
                                v2,
                                v3,
                                v4,
                                v5,
                            ],
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

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_cw_div(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = wptype(2) * wp.cw_div(v2[0], s2[0])
        v3result = wptype(2) * wp.cw_div(v3[0], s3[0])
        v4result = wptype(2) * wp.cw_div(v4[0], s4[0])
        v5result = wptype(2) * wp.cw_div(v5[0], s5[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = v2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = v3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = v4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = v5result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_cw_div, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = randvals(rng, [1, 2, 2], dtype)
    s3 = randvals(rng, [1, 3, 3], dtype)
    s4 = randvals(rng, [1, 4, 4], dtype)
    s5 = randvals(rng, [1, 5, 5], dtype)

    # set denominators to 1 if their magnitudes are small
    # to prevent divide by zero, or overflows if we're testing
    # float16:
    s2[np.abs(s2) < 1.0e-2] = 1
    s3[np.abs(s3) < 1.0e-2] = 1
    s4[np.abs(s4) < 1.0e-2] = 1
    s5[np.abs(s5) < 1.0e-2] = 1

    s2 = wp.array(s2, dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(s3, dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(s4, dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(s5, dtype=mat55, requires_grad=True, device=device)

    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            s2,
            s3,
            s4,
            s5,
            v2,
            v3,
            v4,
            v5,
        ],
        outputs=[outcomponents],
        device=device,
    )

    if dtype in np_float_types:
        assert_np_equal(outcomponents.numpy()[:4], 2 * (v2.numpy() / s2.numpy()).reshape(-1), tol=50 * tol)
        assert_np_equal(outcomponents.numpy()[4:13], 2 * (v3.numpy() / s3.numpy()).reshape(-1), tol=50 * tol)
        assert_np_equal(outcomponents.numpy()[13:29], 2 * (v4.numpy() / s4.numpy()).reshape(-1), tol=50 * tol)
        assert_np_equal(outcomponents.numpy()[29:54], 2 * (v5.numpy() / s5.numpy()).reshape(-1), tol=50 * tol)
    else:
        assert_np_equal(outcomponents.numpy()[:4], 2 * (v2.numpy() // s2.numpy()).reshape(-1), tol=50 * tol)
        assert_np_equal(outcomponents.numpy()[4:13], 2 * (v3.numpy() // s3.numpy()).reshape(-1), tol=50 * tol)
        assert_np_equal(outcomponents.numpy()[13:29], 2 * (v4.numpy() // s4.numpy()).reshape(-1), tol=50 * tol)
        assert_np_equal(outcomponents.numpy()[29:54], 2 * (v5.numpy() // s5.numpy()).reshape(-1), tol=50 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, s, v in [(2, s2, v2), (3, s3, v3), (4, s4, v4), (5, s5, v5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[
                                s2,
                                s3,
                                s4,
                                s5,
                                v2,
                                v3,
                                v4,
                                v5,
                            ],
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


def test_outer_product(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_outer_product(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        outcomponents: wp.array(dtype=wptype),
    ):
        m22result = wptype(2) * wp.outer(s2[0], v2[0])
        m33result = wptype(2) * wp.outer(s3[0], v3[0])
        m44result = wptype(2) * wp.outer(s4[0], v4[0])
        m55result = wptype(2) * wp.outer(s5[0], v5[0])
        m25result = wptype(2) * wp.outer(s2[0], v5[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m22result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m33result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m44result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m55result[i, j]
                idx = idx + 1

        for i in range(2):
            for j in range(5):
                outcomponents[idx] = m25result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_outer_product, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4], dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, [1, 5], dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2], dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4], dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5], dtype), dtype=vec5, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 2 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[s2, s3, s4, s5, v2, v3, v4, v5], outputs=[outcomponents], device=device)

    outcomponents_np = outcomponents.numpy()

    assert_np_equal(outcomponents_np[:4].reshape((2, 2)), 2 * s2.numpy()[0, :, None] * v2.numpy()[0, None, :], tol=tol)
    assert_np_equal(
        outcomponents_np[4:13].reshape((3, 3)), 2 * s3.numpy()[0, :, None] * v3.numpy()[0, None, :], tol=10 * tol
    )
    assert_np_equal(
        outcomponents_np[13:29].reshape((4, 4)), 2 * s4.numpy()[0, :, None] * v4.numpy()[0, None, :], tol=10 * tol
    )
    assert_np_equal(
        outcomponents_np[29:54].reshape((5, 5)), 2 * s5.numpy()[0, :, None] * v5.numpy()[0, None, :], tol=10 * tol
    )
    assert_np_equal(
        outcomponents_np[54:].reshape(2, 5), 2 * s2.numpy()[0, :, None] * v5.numpy()[0, None, :], tol=10 * tol
    )

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for s, v in [(s2, v2), (s3, v3), (s4, v4), (s5, v5), (s2, v5)]:
            rows = s.dtype._length_
            cols = v.dtype._length_
            for i in range(rows):
                for j in range(cols):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[
                                s2,
                                s3,
                                s4,
                                s5,
                                v2,
                                v3,
                                v4,
                                v5,
                            ],
                            outputs=[outcomponents],
                            device=device,
                        )
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

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat32 = wp.types.matrix(shape=(3, 2), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_transpose(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        m32: wp.array(dtype=mat32),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        mat2 = wptype(2) * wp.transpose(m2[0])
        mat3 = wptype(2) * wp.transpose(m3[0])
        mat4 = wptype(2) * wp.transpose(m4[0])
        mat5 = wptype(2) * wp.transpose(m5[0])
        mat32 = wptype(2) * wp.transpose(m32[0])

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = mat2[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = mat3[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = mat4[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = mat5[i, j]
                idx = idx + 1

        for i in range(2):
            for j in range(3):
                outcomponents[idx] = mat32[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_transpose, suffix=dtype.__name__)

    if register_kernels:
        return

    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    m32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 2 * 3, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m2, m3, m4, m5, m32], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], 2 * m2.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 2 * m3.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], 2 * m4.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], 2 * m5.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[54:], 2 * m32.numpy()[0].T.reshape(-1), tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for input in [m2, m3, m4, m5]:
            for i in range(input.dtype._shape_[0]):
                for j in range(input.dtype._shape_[1]):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[m2, m3, m4, m5, m32], outputs=[outcomponents], device=device)
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expectedresult = np.zeros((input.dtype._shape_[1], input.dtype._shape_[0]), dtype=dtype)
                    expectedresult[j, i] = 2
                    assert_np_equal(tape.gradients[input].numpy()[0], expectedresult)
                    tape.zero()
                    idx = idx + 1


def test_scalar_division(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_scalar_div(
        s: wp.array(dtype=wptype),
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2result = m2[0] / s[0]
        m3result = m3[0] / s[0]
        m4result = m4[0] / s[0]
        m5result = m5[0] / s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * m3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * m4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * m5result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_scalar_div, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[s, m2, m3, m4, m5], outputs=[outcomponents], device=device)

    sval = s.numpy()[0]
    if dtype in np_float_types:
        assert_np_equal(outcomponents.numpy()[:4], 2 * m2.numpy().reshape(-1) / sval, tol=tol)
        assert_np_equal(outcomponents.numpy()[4:13], 2 * m3.numpy().reshape(-1) / sval, tol=10 * tol)
        assert_np_equal(outcomponents.numpy()[13:29], 2 * m4.numpy().reshape(-1) / sval, tol=10 * tol)
        assert_np_equal(outcomponents.numpy()[29:54], 2 * m5.numpy().reshape(-1) / sval, tol=10 * tol)
    else:
        assert_np_equal(outcomponents.numpy()[:4], 2 * (m2.numpy().reshape(-1) // sval), tol=tol)
        assert_np_equal(outcomponents.numpy()[4:13], 2 * (m3.numpy().reshape(-1) // sval), tol=10 * tol)
        assert_np_equal(outcomponents.numpy()[13:29], 2 * (m4.numpy().reshape(-1) // sval), tol=10 * tol)
        assert_np_equal(outcomponents.numpy()[29:54], 2 * (m5.numpy().reshape(-1) // sval), tol=10 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (3, m3), (4, m4), (5, m5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[s, m2, m3, m4, m5], outputs=[outcomponents], device=device)
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


def test_addition(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-2,
        np.float32: 5.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_add(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = v2[0] + s2[0]
        v3result = v3[0] + s3[0]
        v4result = v4[0] + s4[0]
        v5result = v5[0] + s5[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * v2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * v3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * v4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * v5result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_add, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            s2,
            s3,
            s4,
            s5,
            v2,
            v3,
            v4,
            v5,
        ],
        outputs=[outcomponents],
        device=device,
    )

    assert_np_equal(outcomponents.numpy()[:4], 2 * (v2.numpy() + s2.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 2 * (v3.numpy() + s3.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], 2 * (v4.numpy() + s4.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], 2 * (v5.numpy() + s5.numpy()).reshape(-1), tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, in1, in2 in [(2, s2, v2), (3, s3, v3), (4, s4, v4), (5, s5, v5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[
                                s2,
                                s3,
                                s4,
                                s5,
                                v2,
                                v3,
                                v4,
                                v5,
                            ],
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


def test_ddot(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    def check_mat_dot(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        dot2: wp.array(dtype=wptype),
        dot3: wp.array(dtype=wptype),
        dot4: wp.array(dtype=wptype),
        dot5: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        dot2[0] = wptype(2) * wp.ddot(v2[0], s2[0])
        dot3[0] = wptype(2) * wp.ddot(v3[0], s3[0])
        dot4[0] = wptype(2) * wp.ddot(v4[0], s4[0])
        dot5[0] = wptype(2) * wp.ddot(v5[0], s5[0])

    kernel = getkernel(check_mat_dot, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    dot2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot5 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                s2,
                s3,
                s4,
                s5,
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[dot2, dot3, dot4, dot5],
            device=device,
        )

    assert_np_equal(dot2.numpy()[0], 2 * (v2.numpy() * s2.numpy()).sum(), tol=10 * tol)
    assert_np_equal(dot3.numpy()[0], 2 * (v3.numpy() * s3.numpy()).sum(), tol=10 * tol)
    assert_np_equal(dot4.numpy()[0], 2 * (v4.numpy() * s4.numpy()).sum(), tol=50 * tol)
    assert_np_equal(dot5.numpy()[0], 2 * (v5.numpy() * s5.numpy()).sum(), tol=200 * tol)

    if dtype in np_float_types:
        tape.backward(loss=dot2)
        sgrads = tape.gradients[s2].numpy()[0]
        expected_grads = 2.0 * v2.numpy()[0]
        assert_np_equal(sgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v2].numpy()[0]
        expected_grads = 2.0 * s2.numpy()[0]
        assert_np_equal(vgrads, expected_grads, tol=10 * tol)

        tape.zero()

        tape.backward(loss=dot3)
        sgrads = tape.gradients[s3].numpy()[0]
        expected_grads = 2.0 * v3.numpy()[0]
        assert_np_equal(sgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v3].numpy()[0]
        expected_grads = 2.0 * s3.numpy()[0]
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

        tape.backward(loss=dot5)
        sgrads = tape.gradients[s5].numpy()[0]
        expected_grads = 2.0 * v5.numpy()[0]
        assert_np_equal(sgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v5].numpy()[0]
        expected_grads = 2.0 * s5.numpy()[0]
        assert_np_equal(vgrads, expected_grads, tol=10 * tol)

        tape.zero()


def test_trace(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    def check_mat_trace(
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        tr2: wp.array(dtype=wptype),
        tr3: wp.array(dtype=wptype),
        tr4: wp.array(dtype=wptype),
        tr5: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        tr2[0] = wptype(2) * wp.trace(v2[0])
        tr3[0] = wptype(2) * wp.trace(v3[0])
        tr4[0] = wptype(2) * wp.trace(v4[0])
        tr5[0] = wptype(2) * wp.trace(v5[0])

    kernel = getkernel(check_mat_trace, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    tr2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tr3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tr4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tr5 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[
                tr2,
                tr3,
                tr4,
                tr5,
            ],
            device=device,
        )

    assert_np_equal(tr2.numpy()[0], 2 * np.trace(v2.numpy()[0]), tol=10 * tol)
    assert_np_equal(tr3.numpy()[0], 2 * np.trace(v3.numpy()[0]), tol=10 * tol)
    assert_np_equal(tr4.numpy()[0], 2 * np.trace(v4.numpy()[0]), tol=200 * tol)
    assert_np_equal(tr4.numpy()[0], 2 * np.trace(v4.numpy()[0]), tol=200 * tol)

    if dtype in np_float_types:
        tape.backward(loss=tr2)
        vgrads = tape.gradients[v2].numpy()[0]
        assert_np_equal(vgrads, 2.0 * np.eye(2), tol=10 * tol)
        tape.zero()

        tape.backward(loss=tr3)
        vgrads = tape.gradients[v3].numpy()[0]
        assert_np_equal(vgrads, 2.0 * np.eye(3), tol=10 * tol)
        tape.zero()

        tape.backward(loss=tr4)
        vgrads = tape.gradients[v4].numpy()[0]
        assert_np_equal(vgrads, 2.0 * np.eye(4), tol=10 * tol)
        tape.zero()

        tape.backward(loss=tr5)
        vgrads = tape.gradients[v5].numpy()[0]
        assert_np_equal(vgrads, 2.0 * np.eye(5), tol=10 * tol)
        tape.zero()


def test_diag(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec5 = wp.types.vector(length=5, dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_diag(
        s5: wp.array(dtype=vec5),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        m55result = wptype(2) * wp.diag(s5[0])

        idx = 0
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m55result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_diag, suffix=dtype.__name__)

    if register_kernels:
        return

    s5 = wp.array(randvals(rng, [1, 5], dtype), dtype=vec5, requires_grad=True, device=device)
    outcomponents = wp.zeros(5 * 5, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[s5], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.reshape((5, 5)).numpy(), 2 * np.diag(s5.numpy()[0]), tol=tol)

    if dtype in np_float_types:
        idx = 0
        for i in range(5):
            for j in range(5):
                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[s5], outputs=[outcomponents], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                tape.backward(loss=out)
                expectedresult = np.zeros(5, dtype=dtype)
                if i == j:
                    expectedresult[i] = 2
                assert_np_equal(tape.gradients[s5].numpy()[0], expectedresult, tol=10 * tol)
                tape.zero()

                idx = idx + 1


def test_equivalent_types(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    # matrix types
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    # matrix types equivalent to the above
    mat22_equiv = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33_equiv = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44_equiv = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55_equiv = wp.types.matrix(shape=(5, 5), dtype=wptype)

    # declare kernel with original types
    def check_equivalence(
        m2: mat22,
        m3: mat33,
        m4: mat44,
        m5: mat55,
    ):
        wp.expect_eq(m2, mat22(wptype(42)))
        wp.expect_eq(m3, mat33(wptype(43)))
        wp.expect_eq(m4, mat44(wptype(44)))
        wp.expect_eq(m5, mat55(wptype(45)))

        wp.expect_eq(m2, mat22_equiv(wptype(42)))
        wp.expect_eq(m3, mat33_equiv(wptype(43)))
        wp.expect_eq(m4, mat44_equiv(wptype(44)))
        wp.expect_eq(m5, mat55_equiv(wptype(45)))

    kernel = getkernel(check_equivalence, suffix=dtype.__name__)

    if register_kernels:
        return

    # call kernel with equivalent types
    m2 = mat22_equiv(42)
    m3 = mat33_equiv(43)
    m4 = mat44_equiv(44)
    m5 = mat55_equiv(45)

    wp.launch(kernel, dim=1, inputs=[m2, m3, m4, m5], device=device)


def test_conversions(test, device, dtype, register_kernels=False):
    def check_matrices_equal(
        m0: wp.mat22,
        m1: wp.mat22,
        m2: wp.mat22,
        m3: wp.mat22,
        m4: wp.mat22,
        m5: wp.mat22,
        m6: wp.mat22,
    ):
        wp.expect_eq(m1, m0)
        wp.expect_eq(m2, m0)
        wp.expect_eq(m3, m0)
        wp.expect_eq(m4, m0)
        wp.expect_eq(m5, m0)
        wp.expect_eq(m6, m0)

    kernel = getkernel(check_matrices_equal, suffix=dtype.__name__)

    if register_kernels:
        return

    m0 = wp.mat22(1, 2, 3, 4)

    # test explicit conversions - constructing matrices from different containers
    m1 = wp.mat22(((1, 2), (3, 4)))  # nested tuples
    m2 = wp.mat22([[1, 2], [3, 4]])  # nested lists
    m3 = wp.mat22(np.array([[1, 2], [3, 4]], dtype=dtype))  # 2d array
    m4 = wp.mat22((1, 2, 3, 4))  # flat tuple
    m5 = wp.mat22([1, 2, 3, 4])  # flat list
    m6 = wp.mat22(np.array([1, 2, 3, 4], dtype=dtype))  # 1d array

    wp.launch(kernel, dim=1, inputs=[m0, m1, m2, m3, m4, m5, m6], device=device)

    # test implicit conversions - passing different containers as matrices to wp.launch()
    m1 = ((1, 2), (3, 4))  # nested tuples
    m2 = [[1, 2], [3, 4]]  # nested lists
    m3 = np.array([[1, 2], [3, 4]], dtype=dtype)  # 2d array
    m4 = (1, 2, 3, 4)  # flat tuple
    m5 = [1, 2, 3, 4]  # flat list
    m6 = np.array([1, 2, 3, 4], dtype=dtype)  # 1d array

    wp.launch(kernel, dim=1, inputs=[m0, m1, m2, m3, m4, m5, m6], device=device)


devices = get_test_devices()


class TestMatScalarOps(unittest.TestCase):
    pass


for dtype in np_scalar_types:
    add_function_test(TestMatScalarOps, f"test_arrays_{dtype.__name__}", test_arrays, devices=devices, dtype=dtype)
    add_function_test(TestMatScalarOps, f"test_components_{dtype.__name__}", test_components, devices=None, dtype=dtype)
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_constructors_{dtype.__name__}", test_constructors, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps,
        f"test_anon_type_instance_{dtype.__name__}",
        test_anon_type_instance,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_identity_{dtype.__name__}", test_identity, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_indexing_{dtype.__name__}", test_indexing, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_equality_{dtype.__name__}", test_equality, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps,
        f"test_scalar_multiplication_{dtype.__name__}",
        test_scalar_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatScalarOps,
        f"test_matvec_multiplication_{dtype.__name__}",
        test_matvec_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatScalarOps,
        f"test_vecmat_multiplication_{dtype.__name__}",
        test_vecmat_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatScalarOps,
        f"test_matmat_multiplication_{dtype.__name__}",
        test_matmat_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatScalarOps,
        f"test_cw_multiplication_{dtype.__name__}",
        test_cw_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_cw_division_{dtype.__name__}", test_cw_division, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_outer_product_{dtype.__name__}", test_outer_product, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_transpose_{dtype.__name__}", test_transpose, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_scalar_division_{dtype.__name__}", test_scalar_division, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_addition_{dtype.__name__}", test_addition, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_ddot_{dtype.__name__}", test_ddot, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_trace_{dtype.__name__}", test_trace, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_diag_{dtype.__name__}", test_diag, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_get_diag_{dtype.__name__}", test_diag, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_equivalent_types_{dtype.__name__}", test_equivalent_types, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_conversions_{dtype.__name__}", test_conversions, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatScalarOps, f"test_constants_{dtype.__name__}", test_constants, devices=None, dtype=dtype
    )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
