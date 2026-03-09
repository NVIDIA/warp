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

import math
import unittest

import warp as wp
import warp.tests.aux_test_compile_consts_dummy
from warp.tests.unittest_utils import *

LOCAL_ONE = wp.constant(1)

SQRT3_OVER_3 = wp.constant(0.57735026919)
UNIT_VEC = wp.constant(wp.vec3(SQRT3_OVER_3, SQRT3_OVER_3, SQRT3_OVER_3))
ONE_FP16 = wp.constant(wp.float16(1.0))
TEST_BOOL = wp.constant(True)

SHADOWED_GLOBAL = wp.constant(17)


class Foobar:
    ONE = wp.constant(1)
    TWO = wp.constant(2)


@wp.kernel
def test_bool():
    if TEST_BOOL:
        expect_eq(1.0, 1.0)
    else:
        expect_eq(1.0, -1.0)


@wp.kernel
def test_int(a: int):
    if Foobar.ONE > 0:
        a = 123 + Foobar.TWO + warp.tests.aux_test_compile_consts_dummy.MINUS_ONE
    else:
        a = 456 + LOCAL_ONE
    expect_eq(a, 124)


@wp.kernel
def test_float(x: float):
    x = SQRT3_OVER_3
    for i in range(3):
        expect_eq(UNIT_VEC[i], x)

    approx_one = wp.dot(UNIT_VEC, UNIT_VEC)
    expect_near(approx_one, 1.0, 1e-6)

    # test casting
    expect_near(wp.float32(ONE_FP16), 1.0, 1e-6)


def test_closure_capture(test, device):
    def make_closure_kernel(cst):
        def closure_kernel_fn(expected: int):
            wp.expect_eq(cst, expected)

        return wp.Kernel(func=closure_kernel_fn)

    one_closure = make_closure_kernel(Foobar.ONE)
    two_closure = make_closure_kernel(Foobar.TWO)

    wp.launch(one_closure, dim=(1), inputs=[1], device=device)
    wp.launch(two_closure, dim=(1), inputs=[2], device=device)


def test_closure_precedence(test, device):
    """Verifies that closure constants take precedence over globals"""

    SHADOWED_GLOBAL = wp.constant(42)

    @wp.kernel
    def closure_kernel():
        wp.expect_eq(SHADOWED_GLOBAL, 42)

    wp.launch(closure_kernel, dim=1, device=device)


def test_hash_global_capture(test, device):
    """Verifies that global variables are included in the module hash"""

    a = 0
    wp.launch(test_int, (1,), inputs=[a], device=device)


def test_hash_redefine_kernel(test, device):
    """This test defines a second ``test_function`` so that the second launch returns the correct result."""

    @wp.kernel
    def test_function(data: wp.array(dtype=wp.float32)):
        i = wp.tid()
        data[i] = TEST_CONSTANT

    TEST_CONSTANT = wp.constant(1.0)

    test_array = wp.empty(1, dtype=wp.float32, device=device)
    wp.launch(test_function, (1,), inputs=[test_array], device=device)
    test.assertEqual(test_array.numpy()[0], 1.0)

    module_hash_0 = wp.get_module(test_function.__module__).hash_module()

    @wp.kernel
    def test_function(data: wp.array(dtype=wp.float32)):
        i = wp.tid()
        data[i] = TEST_CONSTANT

    TEST_CONSTANT = wp.constant(2.0)

    wp.launch(test_function, (1,), inputs=[test_array], device=device)
    test.assertEqual(test_array.numpy()[0], 2.0)

    module_hash_1 = wp.get_module(test_function.__module__).hash_module()

    test.assertNotEqual(module_hash_0, module_hash_1)


def test_hash_redefine_constant_only(test, device):
    """This test does not define a second ``test_function``, so the second launch does not invalidate the cache.

    For now this is expected behavior, but we can verify that the content has is different.
    """

    @wp.kernel
    def test_function(data: wp.array(dtype=wp.float32)):
        i = wp.tid()
        data[i] = TEST_CONSTANT

    TEST_CONSTANT = wp.constant(1.0)

    test_array = wp.empty(1, dtype=wp.float32, device=device)
    wp.launch(test_function, (1,), inputs=[test_array], device=device)
    test.assertEqual(test_array.numpy()[0], 1.0)

    module_hash_0 = wp.get_module(test_function.__module__).hash_module()

    TEST_CONSTANT = wp.constant(2.0)
    module_hash_1 = wp.get_module(test_function.__module__).hash_module()
    test.assertNotEqual(module_hash_0, module_hash_1, "Module hashes should be different if TEST_CONSTANT is changed.")

    TEST_CONSTANT = wp.constant(1.0)
    module_hash_2 = wp.get_module(test_function.__module__).hash_module()
    test.assertEqual(module_hash_0, module_hash_2, "Module hashes should be the same if TEST_CONSTANT is the same.")


def test_hash_shadowed_var(test, device):
    """Tests to ensure shadowed variables are not mistakenly added to the module hash"""

    TEST_CONSTANT_SHADOW_0 = wp.constant(1.0)
    TEST_CONSTANT_SHADOW_1 = wp.constant(1.0)
    TEST_CONSTANT_SHADOW_2 = wp.constant(1.0)

    @wp.kernel
    def test_function(data: wp.array(dtype=wp.float32)):
        i = wp.tid()
        TEST_CONSTANT_SHADOW_0 = 2.0
        TEST_CONSTANT_SHADOW_1, TEST_CONSTANT_SHADOW_2 = 4.0, 8.0
        data[i] = TEST_CONSTANT_SHADOW_0 + TEST_CONSTANT_SHADOW_1 + TEST_CONSTANT_SHADOW_2

    test_array = wp.empty(1, dtype=wp.float32, device=device)
    wp.launch(test_function, (1,), inputs=[test_array], device=device)
    test.assertEqual(test_array.numpy()[0], 14.0)

    module_hash_0 = wp.get_module(test_function.__module__).hash_module()

    TEST_CONSTANT_SHADOW_0 = wp.constant(0.0)
    TEST_CONSTANT_SHADOW_1 = wp.constant(0.0)
    TEST_CONSTANT_SHADOW_2 = wp.constant(0.0)
    module_hash_1 = wp.get_module(test_function.__module__).hash_module()
    test.assertEqual(module_hash_0, module_hash_1, "Module hashes should be the same since all constants are shadowed.")


def test_uint64_large_constant(test, device):
    """Tests that uint64 literals larger than uint32 max are not truncated."""

    @wp.kernel
    def uint64_large_const_kernel(data: wp.array(dtype=wp.uint64)):
        i = wp.tid()
        h = data[i]
        h = h * wp.uint64(0xFF51AFD7ED558CCD)
        data[i] = h

    arr = wp.array([wp.uint64(1)], dtype=wp.uint64, device=device)
    wp.launch(uint64_large_const_kernel, dim=1, inputs=[arr], device=device)
    result = arr.numpy()[0]
    test.assertEqual(result, 0xFF51AFD7ED558CCD, f"Expected 0xFF51AFD7ED558CCD, got {hex(result)}")


def test_float64_precision(test, device):
    """Tests that float64 literals are not truncated to float32 precision."""

    @wp.kernel
    def float64_precision_kernel(data: wp.array(dtype=wp.float64)):
        i = wp.tid()
        x = data[i]
        x = x + wp.float64(3.141592653589793)
        data[i] = x

    arr = wp.array([0.0], dtype=wp.float64, device=device)
    wp.launch(float64_precision_kernel, dim=1, inputs=[arr], device=device)
    result = arr.numpy()[0]
    # float32(3.141592653589793) == 3.1415927410125732 (loses precision)
    # float64 should preserve the full value
    test.assertEqual(result, 3.141592653589793, f"Expected 3.141592653589793, got {result!r}")


def test_float64_wp_pi(test, device):
    """Tests that wp.PI preserves full float64 precision through wp.float64()."""

    @wp.kernel
    def wp_pi_kernel(data: wp.array(dtype=wp.float64)):
        i = wp.tid()
        pi = wp.float64(wp.PI)
        data[i] = pi + wp.float64(1.0)

    arr = wp.array([0.0], dtype=wp.float64, device=device)
    wp.launch(wp_pi_kernel, dim=1, inputs=[arr], device=device)
    result = arr.numpy()[0]
    expected = 3.14159265358979323846 + 1.0
    test.assertEqual(result, expected, f"Expected {expected!r}, got {result!r}")


def test_int64_negative(test, device):
    """Tests that negative int64 literals preserve precision."""

    @wp.kernel
    def int64_neg_kernel(data: wp.array(dtype=wp.int64)):
        i = wp.tid()
        data[i] = wp.int64(-9223372036854775807)

    arr = wp.array([wp.int64(0)], dtype=wp.int64, device=device)
    wp.launch(int64_neg_kernel, dim=1, inputs=[arr], device=device)
    result = arr.numpy()[0]
    test.assertEqual(result, -9223372036854775807, f"Expected -9223372036854775807, got {result}")


def test_scalar_constructor_edge_cases(test, device):
    """Tests wp.INF, wp.NAN, and -wp.INF through scalar constructors."""

    @wp.kernel
    def special_float_kernel(
        inf_data: wp.array(dtype=wp.float64),
        nan_data: wp.array(dtype=wp.float64),
        neg_inf_data: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        inf_data[i] = wp.float64(wp.INF)
        nan_data[i] = wp.float64(wp.NAN)
        neg_inf_data[i] = wp.float64(-wp.INF)

    inf_arr = wp.array([0.0], dtype=wp.float64, device=device)
    nan_arr = wp.array([0.0], dtype=wp.float64, device=device)
    neg_inf_arr = wp.array([0.0], dtype=wp.float64, device=device)
    wp.launch(special_float_kernel, dim=1, inputs=[inf_arr, nan_arr, neg_inf_arr], device=device)
    test.assertEqual(inf_arr.numpy()[0], math.inf)
    test.assertTrue(math.isnan(nan_arr.numpy()[0]))
    test.assertEqual(neg_inf_arr.numpy()[0], -math.inf)


def test_negative_constant_codegen(test, device):
    """Verifies negative float/int constants emit as negative literals in C++."""
    from warp._src.codegen import codegen_func_forward  # noqa: PLC0415

    @wp.kernel
    def neg_codegen_kernel(
        f64_data: wp.array(dtype=wp.float64),
        i64_data: wp.array(dtype=wp.int64),
    ):
        i = wp.tid()
        f64_data[i] = wp.float64(-1.5)
        i64_data[i] = wp.int64(-42)

    f64_arr = wp.array([0.0], dtype=wp.float64, device=device)
    i64_arr = wp.array([wp.int64(0)], dtype=wp.int64, device=device)
    wp.launch(neg_codegen_kernel, dim=1, inputs=[f64_arr, i64_arr], device=device)

    neg_codegen_kernel.adj.build(builder=None)
    source = codegen_func_forward(neg_codegen_kernel.adj, func_type="kernel", device="cpu")

    # Negative values must appear as literal negative constants
    test.assertIn("-1.5", source, "float64(-1.5) should emit as a negative literal")
    test.assertIn("-42ll", source, "int64(-42) should emit as a negative literal")

    # There must be no neg() call for these constants
    test.assertNotIn("neg(", source, "Negative constants should not generate neg() calls")


class TestConstants(unittest.TestCase):
    def test_constant_math(self):
        # test doing math with python defined constants in *python* scope
        twopi = wp.pi * 2.0
        self.assertEqual(twopi, math.pi * 2.0)


a = 0
x = 0.0

devices = get_test_devices()

add_kernel_test(TestConstants, test_bool, dim=1, inputs=[], devices=devices)
add_kernel_test(TestConstants, test_int, dim=1, inputs=[a], devices=devices)
add_kernel_test(TestConstants, test_float, dim=1, inputs=[x], devices=devices)

add_function_test(TestConstants, "test_closure_capture", test_closure_capture, devices=devices)
add_function_test(TestConstants, "test_closure_precedence", test_closure_precedence, devices=devices)
add_function_test(TestConstants, "test_hash_global_capture", test_hash_global_capture, devices=devices)
add_function_test(TestConstants, "test_hash_redefine_kernel", test_hash_redefine_kernel, devices=devices)
add_function_test(TestConstants, "test_hash_redefine_constant_only", test_hash_redefine_constant_only, devices=devices)
add_function_test(TestConstants, "test_hash_shadowed_var", test_hash_shadowed_var, devices=devices)
add_function_test(TestConstants, "test_uint64_large_constant", test_uint64_large_constant, devices=devices)
add_function_test(TestConstants, "test_float64_precision", test_float64_precision, devices=devices)
add_function_test(TestConstants, "test_float64_wp_pi", test_float64_wp_pi, devices=devices)
add_function_test(TestConstants, "test_int64_negative", test_int64_negative, devices=devices)
add_function_test(
    TestConstants, "test_scalar_constructor_edge_cases", test_scalar_constructor_edge_cases, devices=devices
)
add_function_test(TestConstants, "test_negative_constant_codegen", test_negative_constant_codegen, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
