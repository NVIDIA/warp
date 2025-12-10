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
    randvals,
)
from warp.tests.unittest_utils import *

kernel_cache = {}


def setUpModule():
    wp.config.quiet = True


def tearDownModule():
    wp.config.quiet = False


def test_constructors(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp._src.types.vector(length=2, dtype=wptype)
    vec4 = wp._src.types.vector(length=4, dtype=wptype)
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_scalar_mat_constructor(
        input: wp.array(dtype=wptype),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        m2result = wptype(2) * mat22(input[0])
        m4result = wptype(2) * mat44(input[0])

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i, j]
                idx = idx + 1

    def check_component_mat_constructor(
        input: wp.array(dtype=wptype),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        m2result = wptype(2) * mat22(input[0], input[1], input[2], input[3])
        m4result = wptype(2) * mat44(
            input[4],
            input[5],
            input[6],
            input[7],
            input[8],
            input[9],
            input[10],
            input[11],
            input[12],
            input[13],
            input[14],
            input[15],
            input[16],
            input[17],
            input[18],
            input[19],
        )

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i, j]
                idx = idx + 1

    def check_vector_mat_constructor(
        input: wp.array(dtype=wptype),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        m2result = wptype(2) * wp.matrix_from_cols(vec2(input[0], input[2]), vec2(input[1], input[3]))
        m4result = wptype(2) * wp.matrix_from_cols(
            vec4(input[4], input[8], input[12], input[16]),
            vec4(input[5], input[9], input[13], input[17]),
            vec4(input[6], input[10], input[14], input[18]),
            vec4(input[7], input[11], input[15], input[19]),
        )

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_scalar_mat_constructor, suffix=dtype.__name__)
    compkernel = getkernel(kernel_cache, check_component_mat_constructor, suffix=dtype.__name__)
    veckernel = getkernel(kernel_cache, check_vector_mat_constructor, suffix=dtype.__name__)

    if register_kernels:
        return

    input = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    val = input.numpy()[0]
    outcomponents = wp.zeros(2 * 2 + 4 * 4, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[input], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], 2 * val * np.ones(2 * 2), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:20], 2 * val * np.ones(4 * 4), tol=tol)

    if dtype in np_float_types:
        for idx in range(len(outcomponents)):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[input], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
            tape.backward(loss=out)
            test.assertEqual(tape.gradients[input].numpy()[0], 2)
            tape.zero()

    input = wp.array(randvals(rng, [2 * 2 + 4 * 4], dtype), requires_grad=True, device=device)

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


def test_diag(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec5 = wp._src.types.vector(length=5, dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

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

    kernel = getkernel(kernel_cache, check_mat_diag, suffix=dtype.__name__)

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


def test_anon_constructor_error_shape_arg_missing(test, device):
    @wp.kernel
    def kernel():
        wp.types.matrix(1.0, 2.0, 3.0)

    with test.assertRaisesRegex(
        RuntimeError,
        r"the `shape` argument must be specified when initializing a matrix by value$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_constructor_error_shape_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.types.matrix(wp.types.matrix(shape=(1, 2), dtype=float), shape=(3, 4), dtype=float)

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible matrix of shape \(3, 4\) given when copy constructing a matrix of shape \(1, 2\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_constructor_error_type_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.types.matrix(1.0, shape=(3, 2), dtype=wp.float16)

    with test.assertRaisesRegex(
        RuntimeError,
        r"the value used to fill this matrix is expected to be of the type `float16`$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_constructor_error_invalid_arg_count(test, device):
    @wp.kernel
    def kernel():
        wp.types.matrix(1.0, 2.0, 3.0, shape=(2, 2), dtype=float)

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible number of values given \(3\) when constructing a matrix of shape \(2, 2\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_tpl_constructor_error_incompatible_sizes(test, device):
    @wp.kernel
    def kernel():
        wp.mat33(wp.mat22(1.0, 2.0, 3.0, 4.0))

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible matrix of shape \(3, 3\) given when copy constructing a matrix of shape \(2, 2\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_tpl_constructor_error_invalid_arg_count(test, device):
    @wp.kernel
    def kernel():
        wp.mat22(1.0, 2.0, 3.0)

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible number of values given \(3\) when constructing a matrix of shape \(2, 2\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_matrix_from_vecs_runtime(test, device):
    m1 = wp.matrix_from_cols(
        wp.vec3(1.0, 2.0, 3.0),
        wp.vec3(4.0, 5.0, 6.0),
        wp.vec3(7.0, 8.0, 9.0),
    )
    assert m1[0, 0] == 1.0
    assert m1[0, 1] == 4.0
    assert m1[0, 2] == 7.0
    assert m1[1, 0] == 2.0
    assert m1[1, 1] == 5.0
    assert m1[1, 2] == 8.0
    assert m1[2, 0] == 3.0
    assert m1[2, 1] == 6.0
    assert m1[2, 2] == 9.0

    assert m1.get_row(0) == wp.vec3(1.0, 4.0, 7.0)
    assert m1.get_row(1) == wp.vec3(2.0, 5.0, 8.0)
    assert m1.get_row(2) == wp.vec3(3.0, 6.0, 9.0)
    assert m1.get_col(0) == wp.vec3(1.0, 2.0, 3.0)
    assert m1.get_col(1) == wp.vec3(4.0, 5.0, 6.0)
    assert m1.get_col(2) == wp.vec3(7.0, 8.0, 9.0)

    m1.set_row(0, wp.vec3(8.0, 9.0, 10.0))
    m1.set_row(1, wp.vec3(11.0, 12.0, 13.0))
    m1.set_row(2, wp.vec3(14.0, 15.0, 16.0))

    assert m1 == wp.matrix_from_rows(
        wp.vec3(8.0, 9.0, 10.0),
        wp.vec3(11.0, 12.0, 13.0),
        wp.vec3(14.0, 15.0, 16.0),
    )

    m1.set_col(0, wp.vec3(8.0, 9.0, 10.0))
    m1.set_col(1, wp.vec3(11.0, 12.0, 13.0))
    m1.set_col(2, wp.vec3(14.0, 15.0, 16.0))

    assert m1 == wp.matrix_from_cols(
        wp.vec3(8.0, 9.0, 10.0),
        wp.vec3(11.0, 12.0, 13.0),
        wp.vec3(14.0, 15.0, 16.0),
    )

    m2 = wp.matrix_from_rows(
        wp.vec3(1.0, 2.0, 3.0),
        wp.vec3(4.0, 5.0, 6.0),
        wp.vec3(7.0, 8.0, 9.0),
    )
    assert m2[0, 0] == 1.0
    assert m2[0, 1] == 2.0
    assert m2[0, 2] == 3.0
    assert m2[1, 0] == 4.0
    assert m2[1, 1] == 5.0
    assert m2[1, 2] == 6.0
    assert m2[2, 0] == 7.0
    assert m2[2, 1] == 8.0
    assert m2[2, 2] == 9.0

    assert m2.get_row(0) == wp.vec3(1.0, 2.0, 3.0)
    assert m2.get_row(1) == wp.vec3(4.0, 5.0, 6.0)
    assert m2.get_row(2) == wp.vec3(7.0, 8.0, 9.0)
    assert m2.get_col(0) == wp.vec3(1.0, 4.0, 7.0)
    assert m2.get_col(1) == wp.vec3(2.0, 5.0, 8.0)
    assert m2.get_col(2) == wp.vec3(3.0, 6.0, 9.0)

    m2.set_row(0, wp.vec3(8.0, 9.0, 10.0))
    m2.set_row(1, wp.vec3(11.0, 12.0, 13.0))
    m2.set_row(2, wp.vec3(14.0, 15.0, 16.0))

    assert m2 == wp.matrix_from_rows(
        wp.vec3(8.0, 9.0, 10.0),
        wp.vec3(11.0, 12.0, 13.0),
        wp.vec3(14.0, 15.0, 16.0),
    )

    m2.set_col(0, wp.vec3(8.0, 9.0, 10.0))
    m2.set_col(1, wp.vec3(11.0, 12.0, 13.0))
    m2.set_col(2, wp.vec3(14.0, 15.0, 16.0))

    assert m2 == wp.matrix_from_cols(
        wp.vec3(8.0, 9.0, 10.0),
        wp.vec3(11.0, 12.0, 13.0),
        wp.vec3(14.0, 15.0, 16.0),
    )

    m3 = wp.matrix_from_cols(
        wp.vec3(1.0, 2.0, 3.0),
        wp.vec3(4.0, 5.0, 6.0),
    )
    assert m3[0, 0] == 1.0
    assert m3[0, 1] == 4.0
    assert m3[1, 0] == 2.0
    assert m3[1, 1] == 5.0
    assert m3[2, 0] == 3.0
    assert m3[2, 1] == 6.0

    assert m3.get_row(0) == wp.vec2(1.0, 4.0)
    assert m3.get_row(1) == wp.vec2(2.0, 5.0)
    assert m3.get_row(2) == wp.vec2(3.0, 6.0)
    assert m3.get_col(0) == wp.vec3(1.0, 2.0, 3.0)
    assert m3.get_col(1) == wp.vec3(4.0, 5.0, 6.0)

    m3.set_row(0, wp.vec2(7.0, 8.0))
    m3.set_row(1, wp.vec2(9.0, 10.0))
    m3.set_row(2, wp.vec2(11.0, 12.0))

    assert m3 == wp.matrix_from_rows(
        wp.vec2(7.0, 8.0),
        wp.vec2(9.0, 10.0),
        wp.vec2(11.0, 12.0),
    )

    m3.set_col(0, wp.vec3(7.0, 8.0, 9.0))
    m3.set_col(1, wp.vec3(10.0, 11.0, 12.0))

    assert m3 == wp.matrix_from_cols(
        wp.vec3(7.0, 8.0, 9.0),
        wp.vec3(10.0, 11.0, 12.0),
    )

    m4 = wp.matrix_from_rows(
        wp.vec3(1.0, 2.0, 3.0),
        wp.vec3(4.0, 5.0, 6.0),
    )
    assert m4[0, 0] == 1.0
    assert m4[0, 1] == 2.0
    assert m4[0, 2] == 3.0
    assert m4[1, 0] == 4.0
    assert m4[1, 1] == 5.0
    assert m4[1, 2] == 6.0

    assert m4.get_row(0) == wp.vec3(1.0, 2.0, 3.0)
    assert m4.get_row(1) == wp.vec3(4.0, 5.0, 6.0)
    assert m4.get_col(0) == wp.vec2(1.0, 4.0)
    assert m4.get_col(1) == wp.vec2(2.0, 5.0)
    assert m4.get_col(2) == wp.vec2(3.0, 6.0)

    m4.set_row(0, wp.vec3(7.0, 8.0, 9.0))
    m4.set_row(1, wp.vec3(10.0, 11.0, 12.0))

    assert m4 == wp.matrix_from_rows(
        wp.vec3(7.0, 8.0, 9.0),
        wp.vec3(10.0, 11.0, 12.0),
    )

    m4.set_col(0, wp.vec2(7.0, 8.0))
    m4.set_col(1, wp.vec2(9.0, 10.0))
    m4.set_col(2, wp.vec2(11.0, 12.0))

    assert m4 == wp.matrix_from_cols(
        wp.vec2(7.0, 8.0),
        wp.vec2(9.0, 10.0),
        wp.vec2(11.0, 12.0),
    )

    m4.set_row(0, 13.0)

    assert m4 == wp.matrix_from_rows(
        wp.vec3(13.0, 13.0, 13.0),
        wp.vec3(8.0, 10.0, 12.0),
    )

    m4.set_col(2, 14.0)

    assert m4 == wp.matrix_from_rows(
        wp.vec3(13.0, 13.0, 14.0),
        wp.vec3(8.0, 10.0, 14.0),
    )


# Test matrix constructors using explicit type (float16)
# note that these tests are specifically not using generics / closure
# args to create kernels dynamically (like the rest of this file)
# as those use different code paths to resolve arg types which
# has lead to regressions.
@wp.kernel
def test_constructors_explicit_precision():
    # construction for custom matrix types
    eye = wp.identity(dtype=wp.float16, n=2)
    zeros = wp.types.matrix(shape=(2, 2), dtype=wp.float16)
    custom = wp.types.matrix(wp.float16(0.0), wp.float16(1.0), wp.float16(2.0), wp.float16(3.0), shape=(2, 2))

    for i in range(2):
        for j in range(2):
            if i == j:
                wp.expect_eq(eye[i, j], wp.float16(1.0))
            else:
                wp.expect_eq(eye[i, j], wp.float16(0.0))

            wp.expect_eq(zeros[i, j], wp.float16(0.0))
            wp.expect_eq(custom[i, j], wp.float16(i) * wp.float16(2.0) + wp.float16(j))


# Same as above but with a default (float/int) type
# which tests some different code paths that
# need to ensure types are correctly canonicalized
# during codegen
@wp.kernel
def test_constructors_default_precision():
    # construction for default (float) matrix types
    eye = wp.identity(dtype=float, n=2)
    zeros = wp.types.matrix(shape=(2, 2), dtype=float)
    custom = wp.types.matrix(0.0, 1.0, 2.0, 3.0, shape=(2, 2))

    for i in range(2):
        for j in range(2):
            if i == j:
                wp.expect_eq(eye[i, j], 1.0)
            else:
                wp.expect_eq(eye[i, j], 0.0)

            wp.expect_eq(zeros[i, j], 0.0)
            wp.expect_eq(custom[i, j], float(i) * 2.0 + float(j))


# NOTE: Compile tile is highly sensitive to shape so we use small values now
CONSTANT_SHAPE_ROWS = wp.constant(2)
CONSTANT_SHAPE_COLS = wp.constant(2)


# tests that we can use global constants in shape keyword argument
# for matrix constructor
@wp.kernel
def test_constructors_constant_shape():
    m = wp.types.matrix(shape=(CONSTANT_SHAPE_ROWS, CONSTANT_SHAPE_COLS), dtype=float)

    for i in range(CONSTANT_SHAPE_ROWS):
        for j in range(CONSTANT_SHAPE_COLS):
            m[i, j] = float(i * j)


@wp.kernel
def test_matrix_from_vecs():
    m1 = wp.matrix_from_cols(
        wp.vec3(1.0, 2.0, 3.0),
        wp.vec3(4.0, 5.0, 6.0),
        wp.vec3(7.0, 8.0, 9.0),
    )
    wp.expect_eq(m1[0, 0], 1.0)
    wp.expect_eq(m1[0, 1], 4.0)
    wp.expect_eq(m1[0, 2], 7.0)
    wp.expect_eq(m1[1, 0], 2.0)
    wp.expect_eq(m1[1, 1], 5.0)
    wp.expect_eq(m1[1, 2], 8.0)
    wp.expect_eq(m1[2, 0], 3.0)
    wp.expect_eq(m1[2, 1], 6.0)
    wp.expect_eq(m1[2, 2], 9.0)

    m2 = wp.matrix_from_rows(
        wp.vec3(1.0, 2.0, 3.0),
        wp.vec3(4.0, 5.0, 6.0),
        wp.vec3(7.0, 8.0, 9.0),
    )
    wp.expect_eq(m2[0, 0], 1.0)
    wp.expect_eq(m2[0, 1], 2.0)
    wp.expect_eq(m2[0, 2], 3.0)
    wp.expect_eq(m2[1, 0], 4.0)
    wp.expect_eq(m2[1, 1], 5.0)
    wp.expect_eq(m2[1, 2], 6.0)
    wp.expect_eq(m2[2, 0], 7.0)
    wp.expect_eq(m2[2, 1], 8.0)
    wp.expect_eq(m2[2, 2], 9.0)

    m3 = wp.matrix_from_cols(
        wp.vec3(1.0, 2.0, 3.0),
        wp.vec3(4.0, 5.0, 6.0),
    )
    wp.expect_eq(m3[0, 0], 1.0)
    wp.expect_eq(m3[0, 1], 4.0)
    wp.expect_eq(m3[1, 0], 2.0)
    wp.expect_eq(m3[1, 1], 5.0)
    wp.expect_eq(m3[2, 0], 3.0)
    wp.expect_eq(m3[2, 1], 6.0)

    m4 = wp.matrix_from_rows(
        wp.vec3(1.0, 2.0, 3.0),
        wp.vec3(4.0, 5.0, 6.0),
    )
    wp.expect_eq(m4[0, 0], 1.0)
    wp.expect_eq(m4[0, 1], 2.0)
    wp.expect_eq(m4[0, 2], 3.0)
    wp.expect_eq(m4[1, 0], 4.0)
    wp.expect_eq(m4[1, 1], 5.0)
    wp.expect_eq(m4[1, 2], 6.0)


mat32d = wp.types.matrix(shape=(3, 2), dtype=wp.float64)


@wp.kernel
def test_matrix_constructor_value_func():
    a = wp.mat22()
    b = wp.types.matrix(a, shape=(2, 2))
    c = mat32d()
    d = mat32d(c, shape=(3, 2))
    e = mat32d(wp.float64(1.0), wp.float64(2.0), wp.float64(1.0), wp.float64(2.0), wp.float64(1.0), wp.float64(2.0))
    f = wp.types.matrix(1.0, 2.0, 3.0, 4.0, shape=(2, 2), dtype=float)


def test_quat_constructor(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.dtype_from_numpy(np.dtype(dtype))
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    quat = wp.types.quaternion(dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_quat_constructor(
        p: wp.array(dtype=vec3),
        r: wp.array(dtype=quat),
        s: wp.array(dtype=vec3),
        outcomponents: wp.array(dtype=wptype),
        outcomponents_alt: wp.array(dtype=wptype),
    ):
        m = wp.transform_compose(p[0], r[0], s[0])

        R = wp.transpose(wp.quat_to_matrix(r[0]))
        c0 = s[0][0] * R[0]
        c1 = s[0][1] * R[1]
        c2 = s[0][2] * R[2]
        m_alt = wp.matrix_from_cols(
            vec4(c0[0], c0[1], c0[2], wptype(0.0)),
            vec4(c1[0], c1[1], c1[2], wptype(0.0)),
            vec4(c2[0], c2[1], c2[2], wptype(0.0)),
            vec4(p[0][0], p[0][1], p[0][2], wptype(1.0)),
        )

        idx = 0
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m[i, j]
                outcomponents_alt[idx] = m_alt[i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_quat_constructor, suffix=dtype.__name__)

    if register_kernels:
        return

    # translation:
    p = wp.array(rng.standard_normal(size=(1, 3)).astype(dtype), dtype=vec3, requires_grad=True, device=device)

    # generate a normalized quaternion for the rotation:
    r = rng.standard_normal(size=(1, 4))
    r /= np.linalg.norm(r)
    r = wp.array(r.astype(dtype), dtype=quat, requires_grad=True, device=device)

    # scale:
    s = wp.array(rng.standard_normal(size=(1, 3)).astype(dtype), dtype=vec3, requires_grad=True, device=device)

    # just going to generate the matrix using the constructor, then
    # more manually, and make sure the values/gradients are the same:
    outcomponents = wp.zeros(4 * 4, dtype=wptype, requires_grad=True, device=device)
    outcomponents_alt = wp.zeros(4 * 4, dtype=wptype, requires_grad=True, device=device)
    wp.launch(kernel, dim=1, inputs=[p, r, s], outputs=[outcomponents, outcomponents_alt], device=device)
    assert_np_equal(outcomponents.numpy(), outcomponents_alt.numpy(), tol=1.0e-6)

    idx = 0
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    out_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for _i in range(4):
        for _j in range(4):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[p, r, s], outputs=[outcomponents, outcomponents_alt], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                wp.launch(
                    output_select_kernel, dim=1, inputs=[outcomponents_alt, idx], outputs=[out_alt], device=device
                )

            tape.backward(loss=out)
            p_grad = 1.0 * tape.gradients[p].numpy()[0]
            r_grad = 1.0 * tape.gradients[r].numpy()[0]
            s_grad = 1.0 * tape.gradients[s].numpy()[0]
            tape.zero()

            tape.backward(loss=out_alt)
            p_grad_alt = 1.0 * tape.gradients[p].numpy()[0]
            r_grad_alt = 1.0 * tape.gradients[r].numpy()[0]
            s_grad_alt = 1.0 * tape.gradients[s].numpy()[0]
            tape.zero()

            assert_np_equal(p_grad, p_grad_alt, tol=tol)
            assert_np_equal(r_grad, r_grad_alt, tol=tol)
            assert_np_equal(s_grad, s_grad_alt, tol=tol)

            idx = idx + 1


def test_identity(test, device, dtype, register_kernels=False):
    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_identity_mat(
        output: wp.array(dtype=wptype),
    ):
        m2result = wp.identity(dtype=wptype, n=2)
        m5result = wp.identity(dtype=wptype, n=5)

        idx = 0
        for i in range(2):
            for j in range(2):
                output[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1
        for i in range(5):
            for j in range(5):
                output[idx] = wptype(2) * m5result[i, j]
                idx = idx + 1

    id_kernel = getkernel(kernel_cache, check_identity_mat, suffix=dtype.__name__)

    if register_kernels:
        return

    output = wp.zeros(2 * 2 + 5 * 5, dtype=wptype, requires_grad=True, device=device)
    wp.launch(id_kernel, dim=1, inputs=[], outputs=[output], device=device)
    assert_np_equal(output.numpy()[:4], 2 * np.eye(2), tol=1.0e-6)
    assert_np_equal(output.numpy()[4:], 2 * np.eye(5), tol=1.0e-6)


def test_anon_type_instance(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_scalar_init(
        input: wp.array(dtype=wptype),
        output: wp.array(dtype=wptype),
    ):
        m2result = wp.types.matrix(input[0], shape=(2, 2))
        m4result = wp.types.matrix(input[1], shape=(4, 4))
        m32result = wp.types.matrix(input[2], shape=(3, 2))

        idx = 0
        for i in range(2):
            for j in range(2):
                output[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1
        for i in range(4):
            for j in range(4):
                output[idx] = wptype(2) * m4result[i, j]
                idx = idx + 1
        for i in range(3):
            for j in range(2):
                output[idx] = wptype(2) * m32result[i, j]
                idx = idx + 1

    def check_component_init(
        input: wp.array(dtype=wptype),
        output: wp.array(dtype=wptype),
    ):
        m2result = wp.types.matrix(input[0], input[1], input[2], input[3], shape=(2, 2))
        m4result = wp.types.matrix(
            input[4],
            input[5],
            input[6],
            input[7],
            input[8],
            input[9],
            input[10],
            input[11],
            input[12],
            input[13],
            input[14],
            input[15],
            input[16],
            input[17],
            input[18],
            input[19],
            shape=(4, 4),
        )
        m32result = wp.types.matrix(input[20], input[21], input[22], input[23], input[24], input[25], shape=(3, 2))

        idx = 0
        for i in range(2):
            for j in range(2):
                output[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1
        for i in range(4):
            for j in range(4):
                output[idx] = wptype(2) * m4result[i, j]
                idx = idx + 1
        for i in range(3):
            for j in range(2):
                output[idx] = wptype(2) * m32result[i, j]
                idx = idx + 1

    scalar_kernel = getkernel(kernel_cache, check_scalar_init, suffix=dtype.__name__)
    component_kernel = getkernel(kernel_cache, check_component_init, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    if register_kernels:
        return

    input = wp.array(randvals(rng, [3], dtype), requires_grad=True, device=device)
    output = wp.zeros(2 * 2 + 4 * 4 + 3 * 2, dtype=wptype, requires_grad=True, device=device)

    wp.launch(scalar_kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy()[:4], 2 * np.array([input.numpy()[0]] * 2 * 2), tol=1.0e-6)
    assert_np_equal(output.numpy()[4:20], 2 * np.array([input.numpy()[1]] * 4 * 4), tol=1.0e-6)
    assert_np_equal(output.numpy()[20:], 2 * np.array([input.numpy()[2]] * 3 * 2), tol=1.0e-6)

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
            elif i < 20:
                expected[1] = 2
            else:
                expected[2] = 2

            assert_np_equal(tape.gradients[input].numpy(), expected, tol=tol)

            tape.reset()
            tape.zero()

    input = wp.array(randvals(rng, [2 * 2 + 4 * 4 + 3 * 2], dtype), requires_grad=True, device=device)
    output = wp.zeros(2 * 2 + 4 * 4 + 3 * 2, dtype=wptype, requires_grad=True, device=device)

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


devices = get_test_devices()


class TestMatConstructors(unittest.TestCase):
    pass


add_function_test(
    TestMatConstructors,
    "test_anon_constructor_error_shape_arg_missing",
    test_anon_constructor_error_shape_arg_missing,
    devices=devices,
)
add_function_test(
    TestMatConstructors,
    "test_anon_constructor_error_shape_mismatch",
    test_anon_constructor_error_shape_mismatch,
    devices=devices,
)
add_function_test(
    TestMatConstructors,
    "test_anon_constructor_error_type_mismatch",
    test_anon_constructor_error_type_mismatch,
    devices=devices,
)
add_function_test(
    TestMatConstructors,
    "test_anon_constructor_error_invalid_arg_count",
    test_anon_constructor_error_invalid_arg_count,
    devices=devices,
)
add_function_test(
    TestMatConstructors,
    "test_tpl_constructor_error_incompatible_sizes",
    test_tpl_constructor_error_incompatible_sizes,
    devices=devices,
)
add_function_test(
    TestMatConstructors,
    "test_tpl_constructor_error_invalid_arg_count",
    test_tpl_constructor_error_invalid_arg_count,
    devices=devices,
)
add_function_test(TestMatConstructors, "test_matrix_from_vecs_runtime", test_matrix_from_vecs_runtime, devices=devices)


add_kernel_test(TestMatConstructors, test_constructors_explicit_precision, dim=1, devices=devices)
add_kernel_test(TestMatConstructors, test_constructors_default_precision, dim=1, devices=devices)
add_kernel_test(TestMatConstructors, test_constructors_constant_shape, dim=1, devices=devices)
add_kernel_test(TestMatConstructors, test_matrix_from_vecs, dim=1, devices=devices)
add_kernel_test(TestMatConstructors, test_matrix_constructor_value_func, dim=1, devices=devices)

for dtype in np_float_types:
    add_function_test_register_kernel(
        TestMatConstructors, f"test_constructors_{dtype.__name__}", test_constructors, devices=devices, dtype=dtype
    )

    add_function_test_register_kernel(
        TestMatConstructors, f"test_diag_{dtype.__name__}", test_diag, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatConstructors,
        f"test_quat_constructor_{dtype.__name__}",
        test_quat_constructor,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestMatConstructors, f"test_identity_{dtype.__name__}", test_identity, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatConstructors,
        f"test_anon_type_instance_{dtype.__name__}",
        test_anon_type_instance,
        devices=devices,
        dtype=dtype,
    )

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
