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
from warp.tests.matrix.utils import (
    get_select_kernel,
    getkernel,
    np_float_types,
    np_scalar_types,
    randvals,
)
from warp.tests.unittest_utils import *

kernel_cache = {}


def test_arrays(test, device, dtype):
    rng = np.random.default_rng(123)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]

    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)
    mat32 = wp._src.types.matrix(shape=(3, 2), dtype=wptype)

    v2_np = randvals(rng, [10, 2, 2], dtype)
    v4_np = randvals(rng, [10, 4, 4], dtype)
    v32_np = randvals(rng, [10, 3, 2], dtype)

    v2 = wp.array(v2_np, dtype=mat22, requires_grad=True, device=device)
    v4 = wp.array(v4_np, dtype=mat44, requires_grad=True, device=device)
    v32 = wp.array(v32_np, dtype=mat32, requires_grad=True, device=device)

    assert_np_equal(v2.numpy(), v2_np, tol=1.0e-6)
    assert_np_equal(v4.numpy(), v4_np, tol=1.0e-6)
    assert_np_equal(v32.numpy(), v32_np, tol=1.0e-6)


def test_components(test, device, dtype):
    # test accessing matrix components from Python - this is especially important
    # for float16, which requires special handling internally

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat23 = wp._src.types.matrix(shape=(2, 3), dtype=wptype)

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


def test_indexing(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(kernel_cache, wptype)

    def check_mat_indexing(
        m2: wp.array(dtype=mat22),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * m2[0][i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * m4[0][i, j]
                idx = idx + 1

    kernel = getkernel(kernel_cache, check_mat_indexing, suffix=dtype.__name__)

    if register_kernels:
        return

    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 4 * 4, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m2, m4], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], 2 * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:20], 2 * m4.numpy().reshape(-1), tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (4, m4)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[m2, m4], outputs=[outcomponents], device=device)
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
    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

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

    kernel = getkernel(kernel_cache, check_mat_equality, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=1, inputs=[], outputs=[], device=device)


def test_equivalent_types(test, device, dtype, register_kernels=False):
    wptype = wp._src.types.np_dtype_to_warp_type[np.dtype(dtype)]

    # matrix types
    mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44 = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    # matrix types equivalent to the above
    mat22_equiv = wp._src.types.matrix(shape=(2, 2), dtype=wptype)
    mat44_equiv = wp._src.types.matrix(shape=(4, 4), dtype=wptype)

    # declare kernel with original types
    def check_equivalence(
        m2: mat22,
        m4: mat44,
    ):
        wp.expect_eq(m2, mat22(wptype(42)))
        wp.expect_eq(m4, mat44(wptype(44)))

        wp.expect_eq(m2, mat22_equiv(wptype(42)))
        wp.expect_eq(m4, mat44_equiv(wptype(44)))

    kernel = getkernel(kernel_cache, check_equivalence, suffix=dtype.__name__)

    if register_kernels:
        return

    # call kernel with equivalent types
    m2 = mat22_equiv(42)
    m4 = mat44_equiv(44)

    wp.launch(kernel, dim=1, inputs=[m2, m4], device=device)


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

    kernel = getkernel(kernel_cache, check_matrices_equal, suffix=dtype.__name__)

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


class TestMatBasics(unittest.TestCase):
    pass


for dtype in np_scalar_types:
    add_function_test(TestMatBasics, f"test_arrays_{dtype.__name__}", test_arrays, devices=devices, dtype=dtype)
    add_function_test(TestMatBasics, f"test_components_{dtype.__name__}", test_components, devices=None, dtype=dtype)
    add_function_test_register_kernel(
        TestMatBasics, f"test_indexing_{dtype.__name__}", test_indexing, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatBasics, f"test_equality_{dtype.__name__}", test_equality, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatBasics, f"test_equivalent_types_{dtype.__name__}", test_equivalent_types, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMatBasics, f"test_conversions_{dtype.__name__}", test_conversions, devices=devices, dtype=dtype
    )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
