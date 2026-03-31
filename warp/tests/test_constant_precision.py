# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for strongly typed construction.

Typed constructors like ``wp.vec3d(1.0, 2.0, 3.0)`` should preserve
double precision of the literal arguments (not truncate to float32).
Likewise ``wp.vec3h(1.0, 2.0, 3.0)`` should convert literals to
half-precision. This applies to vectors, matrices, quaternions, and
transformations.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# ---------------------------------------------------------------------------
# vec3d — float64 precision preservation
# ---------------------------------------------------------------------------


@wp.kernel
def test_vec3d_precision_kernel(result: wp.array[wp.float64]):
    """Test vec3d constructor preserves double precision."""
    v = wp.vec3d(3.141592653589793, 2.718281828459045, 1.414213562373095)
    result[0] = v[0]


@wp.kernel
def test_vec3d_element_by_element_kernel(result: wp.array[wp.float64]):
    """Test vec3d preserves each component's full precision."""
    v = wp.vec3d(1.00000005, 2.00000005, 3.00000005)
    result[0] = v[0]
    result[1] = v[1]
    result[2] = v[2]


# ---------------------------------------------------------------------------
# vec3 — float32 backward compatibility
# ---------------------------------------------------------------------------


@wp.kernel
def test_vec3_precision_kernel(result: wp.array[wp.float32]):
    """Test vec3 constructor uses single precision."""
    v = wp.vec3(3.141592653589793, 2.718281828459045, 1.414213562373095)
    result[0] = v[0]


# ---------------------------------------------------------------------------
# vec3h — float16
# ---------------------------------------------------------------------------


@wp.kernel
def test_vec3h_float_literals_kernel(result: wp.array[wp.float16]):
    """Test wp.vec3h(float, float, float) converts to half precision."""
    v = wp.vec3h(1.0, 2.0, 3.0)
    result[0] = v[0] + v[1] + v[2]


@wp.kernel
def test_vec3h_int_literals_kernel(result: wp.array[wp.float16]):
    """Test wp.vec3h(int, int, int) works correctly."""
    v = wp.vec3h(1, 2, 3)
    result[0] = v[0] + v[1] + v[2]


# ---------------------------------------------------------------------------
# mat22d — double precision matrix
# ---------------------------------------------------------------------------


@wp.kernel
def test_mat22d_precision_kernel(result: wp.array[wp.float64]):
    """Test mat22d constructor preserves double precision."""
    m = wp.mat22d(1.111111111111111, 2.222222222222222, 3.333333333333333, 4.444444444444444)
    result[0] = m[0, 0]


# ---------------------------------------------------------------------------
# quatd — double precision quaternion
# ---------------------------------------------------------------------------


@wp.kernel
def test_quatd_precision_kernel(result: wp.array[wp.float64]):
    """Test quatd constructor preserves double precision."""
    q = wp.quatd(0.707106781186547, 0.0, 0.707106781186547, 0.0)
    result[0] = q[0]


# ---------------------------------------------------------------------------
# transformd — double precision transformation
# ---------------------------------------------------------------------------


@wp.kernel
def test_transformd_precision_kernel(result: wp.array[wp.float64]):
    """Test transformd constructor preserves double precision."""
    p = wp.vec3d(1.234567890123456, 2.345678901234567, 3.456789012345678)
    q = wp.quatd(1.0, 0.0, 0.0, 0.0)
    t = wp.transformd(p, q)
    result[0] = wp.transform_get_translation(t)[0]


# ---------------------------------------------------------------------------
# Int literals in typed constructors
# ---------------------------------------------------------------------------


@wp.kernel
def test_vec3d_int_literals_kernel(result: wp.array[wp.float64]):
    """Test wp.vec3d(int, int, int) works correctly."""
    v = wp.vec3d(1, 2, 3)
    result[0] = v[0] + v[1] + v[2]


@wp.kernel
def test_vec3_int_literals_kernel(result: wp.array[wp.float32]):
    """Test wp.vec3(int, int, int) works correctly (single precision)."""
    v = wp.vec3(1, 2, 3)
    result[0] = v[0] + v[1] + v[2]


@wp.kernel
def test_int_literal_float64_constructor_kernel(result: wp.array[wp.float64]):
    """Test that int literals in float64 constructors preserve exact values."""
    v = wp.vec3d(16777217, 16777219, 16777221)
    result[0] = v[0]
    result[1] = v[1]
    result[2] = v[2]


# ---------------------------------------------------------------------------
# Fill constructors
# ---------------------------------------------------------------------------


@wp.kernel
def test_vec3d_fill_kernel(result: wp.array[wp.float64]):
    """Test wp.vec3d(scalar) fill preserves precision."""
    v = wp.vec3d(1.00000005)
    result[0] = v[0]


# ---------------------------------------------------------------------------
# Module constants in typed constructors
# ---------------------------------------------------------------------------


@wp.kernel
def test_warp_constant_constructor_kernel(result: wp.array[wp.float64]):
    """Test that wp.vec3d(wp.PI, wp.E, wp.TAU) preserves precision of each component."""
    v = wp.vec3d(wp.PI, wp.E, wp.TAU)
    result[0] = v[0]
    result[1] = v[1]
    result[2] = v[2]


# ---------------------------------------------------------------------------
# Mixed literals and variables
# ---------------------------------------------------------------------------


@wp.kernel
def test_vec3h_mixed_literal_variable_kernel(h: wp.float16, result: wp.array[wp.float16]):
    """Test vec3h(literal, variable, literal) where variable matches dtype."""
    v = wp.vec3h(1.0, h, 3.0)
    result[0] = v[0]
    result[1] = v[1]
    result[2] = v[2]


@wp.kernel
def test_vec3d_mixed_literal_variable_kernel(d: wp.float64, result: wp.array[wp.float64]):
    """Test vec3d(literal, variable, literal) where variable matches dtype."""
    v = wp.vec3d(1.00000005, d, 3.00000005)
    result[0] = v[0]
    result[1] = v[1]
    result[2] = v[2]


@wp.kernel
def test_quatd_mixed_literal_variable_kernel(d: wp.float64, result: wp.array[wp.float64]):
    """Test quatd(literal, literal, literal, variable)."""
    q = wp.quatd(0.0, 0.0, 0.0, d)
    result[0] = q[3]


# ---------------------------------------------------------------------------
# Negative literals in typed constructors
# ---------------------------------------------------------------------------


@wp.kernel
def test_vec3d_negative_literals_kernel(result: wp.array[wp.float64]):
    """Test vec3d with negative literals preserves double precision.

    Python's AST represents -3.14 as UnaryOp(USub, Constant(3.14)), which
    goes through a different codegen path than positive literals.
    """
    v = wp.vec3d(-3.141592653589793, -2.718281828459045, -1.414213562373095)
    result[0] = v[0]
    result[1] = v[1]
    result[2] = v[2]


# ---------------------------------------------------------------------------
# Typed constructor stores to array
# ---------------------------------------------------------------------------


@wp.kernel
def test_typed_constructor_accepts_literals_kernel(result: wp.array[wp.vec3d]):
    """Test that float literals are accepted by typed constructors."""
    v = wp.vec3d(1.0, 2.0, 3.0)
    result[0] = v


# ---------------------------------------------------------------------------
# dtype=float means float32
# ---------------------------------------------------------------------------


@wp.kernel
def test_vector_dtype_float_is_float32_kernel(result: wp.array[wp.vec3]):
    """wp.types.vector(..., dtype=float) produces float32 elements."""
    v = wp.types.vector(1.0, 2.0, 3.0, dtype=float)
    result[0] = v


# ---------------------------------------------------------------------------
# Test functions (for add_function_test with multi-device coverage)
# ---------------------------------------------------------------------------


def test_vec3d_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_vec3d_precision_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(float(result.numpy()[0]), 3.141592653589793)


def test_vec3d_element_by_element(test, device):
    result = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(test_vec3d_element_by_element_kernel, dim=1, inputs=[result], device=device)
    vals = result.numpy()
    test.assertEqual(float(vals[0]), 1.00000005)
    test.assertEqual(float(vals[1]), 2.00000005)
    test.assertEqual(float(vals[2]), 3.00000005)


def test_vec3_backward_compat(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_vec3_precision_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    expected = float(np.float32(3.141592653589793))
    test.assertEqual(val, expected)


def test_vec3h_float_literals(test, device):
    result = wp.zeros(1, dtype=wp.float16, device=device)
    wp.launch(test_vec3h_float_literals_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 6.0)


def test_vec3h_int_literals(test, device):
    result = wp.zeros(1, dtype=wp.float16, device=device)
    wp.launch(test_vec3h_int_literals_kernel, dim=1, inputs=[result], device=device)
    val = float(result.numpy()[0])
    test.assertEqual(val, 6.0)


def test_mat22d_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_mat22d_precision_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(float(result.numpy()[0]), 1.111111111111111)


def test_quatd_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_quatd_precision_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(float(result.numpy()[0]), 0.707106781186547)


def test_transformd_precision(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_transformd_precision_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(float(result.numpy()[0]), 1.234567890123456)


def test_vec3d_int_literals(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_vec3d_int_literals_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(float(result.numpy()[0]), 6.0)


def test_vec3_int_literals(test, device):
    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(test_vec3_int_literals_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(float(result.numpy()[0]), 6.0)


def test_int_literal_float64_constructor(test, device):
    """Int literals in float64 constructors (vec3d(1, 2, 3)) produce exact values."""
    result = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(test_int_literal_float64_constructor_kernel, dim=1, inputs=[result], device=device)
    vals = result.numpy()
    test.assertEqual(float(vals[0]), 16777217.0)
    test.assertEqual(float(vals[1]), 16777219.0)
    test.assertEqual(float(vals[2]), 16777221.0)


def test_vec3d_fill(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_vec3d_fill_kernel, dim=1, inputs=[result], device=device)
    test.assertEqual(float(result.numpy()[0]), 1.00000005)


def test_warp_constant_constructor(test, device):
    """Module constants in typed constructors (vec3d) preserve precision of each component."""
    result = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(test_warp_constant_constructor_kernel, dim=1, inputs=[result], device=device)
    vals = result.numpy()
    test.assertEqual(float(vals[0]), 3.141592653589793)  # wp.PI
    test.assertEqual(float(vals[1]), 2.718281828459045)  # wp.E
    test.assertEqual(float(vals[2]), 6.283185307179586)  # wp.TAU


def test_typed_constructor_accepts_literals(test, device):
    result = wp.zeros(1, dtype=wp.vec3d, device=device)
    wp.launch(test_typed_constructor_accepts_literals_kernel, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy()[0], [1.0, 2.0, 3.0])


def test_vector_dtype_float_is_float32(test, device):
    """wp.types.vector(..., dtype=float) produces float32 elements."""
    result = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(test_vector_dtype_float_is_float32_kernel, dim=1, inputs=[result], device=device)
    np.testing.assert_allclose(result.numpy()[0], [1.0, 2.0, 3.0])


def test_vec3d_negative_literals(test, device):
    result = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(test_vec3d_negative_literals_kernel, dim=1, inputs=[result], device=device)
    vals = result.numpy()
    test.assertEqual(float(vals[0]), -3.141592653589793)
    test.assertEqual(float(vals[1]), -2.718281828459045)
    test.assertEqual(float(vals[2]), -1.414213562373095)


def test_vec3h_mixed_literal_variable(test, device):
    result = wp.zeros(3, dtype=wp.float16, device=device)
    wp.launch(test_vec3h_mixed_literal_variable_kernel, dim=1, inputs=[wp.float16(2.0), result], device=device)
    vals = result.numpy()
    test.assertEqual(float(vals[0]), 1.0)
    test.assertEqual(float(vals[1]), 2.0)
    test.assertEqual(float(vals[2]), 3.0)


def test_vec3d_mixed_literal_variable(test, device):
    result = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(test_vec3d_mixed_literal_variable_kernel, dim=1, inputs=[wp.float64(2.00000005), result], device=device)
    vals = result.numpy()
    test.assertEqual(float(vals[0]), 1.00000005)
    test.assertEqual(float(vals[1]), 2.00000005)
    test.assertEqual(float(vals[2]), 3.00000005)


def test_quatd_mixed_literal_variable(test, device):
    result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(test_quatd_mixed_literal_variable_kernel, dim=1, inputs=[wp.float64(1.0), result], device=device)
    test.assertEqual(float(result.numpy()[0]), 1.0)


def test_typed_constructor_rejects_mismatched_variable(test, device):
    """Passing a float64 variable to a float32 constructor errors."""

    @wp.kernel
    def kernel(x: wp.float64):
        v = wp.vec3f(x, x, x)

    with test.assertRaisesRegex(RuntimeError, r"expected to be of the type"):
        wp.launch(kernel, dim=1, inputs=[wp.float64(1.0)], device=device)


def test_matrix_constructor_rejects_mismatched_variable(test, device):
    """Passing a float64 variable to a float32 matrix constructor errors."""
    mat22f = wp.types.matrix((2, 2), wp.float32)

    @wp.kernel
    def kernel(x: wp.float64):
        m = mat22f(x, x, x, x)

    with test.assertRaisesRegex(RuntimeError, r"expected to be of the type"):
        wp.launch(kernel, dim=1, inputs=[wp.float64(1.0)], device=device)


def test_matrix_fill_rejects_mismatched_variable(test, device):
    """Filling a float32 matrix with a float64 variable errors."""
    mat22f = wp.types.matrix((2, 2), wp.float32)

    @wp.kernel
    def kernel(x: wp.float64):
        m = mat22f(x)

    with test.assertRaisesRegex(RuntimeError, r"expected to be of the type"):
        wp.launch(kernel, dim=1, inputs=[wp.float64(1.0)], device=device)


def test_mixed_literal_variable_rejects_wrong_type(test, device):
    """vec3h(literal, float32_var, literal) errors — variable type doesn't match constructor dtype."""

    @wp.kernel
    def kernel(x: wp.float32):
        v = wp.vec3h(1.0, x, 3.0)

    with test.assertRaisesRegex(RuntimeError, r"expected to be of the type"):
        wp.launch(kernel, dim=1, inputs=[wp.float32(2.0)], device=device)


class TestConstantPrecision(unittest.TestCase):
    """Test suite for constant precision preservation."""

    pass


devices = get_test_devices()

add_function_test(TestConstantPrecision, "test_vec3d_precision", test_vec3d_precision, devices=devices)
add_function_test(
    TestConstantPrecision, "test_vec3d_element_by_element", test_vec3d_element_by_element, devices=devices
)
add_function_test(TestConstantPrecision, "test_vec3_backward_compat", test_vec3_backward_compat, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3h_float_literals", test_vec3h_float_literals, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3h_int_literals", test_vec3h_int_literals, devices=devices)
add_function_test(TestConstantPrecision, "test_mat22d_precision", test_mat22d_precision, devices=devices)
add_function_test(TestConstantPrecision, "test_quatd_precision", test_quatd_precision, devices=devices)
add_function_test(TestConstantPrecision, "test_transformd_precision", test_transformd_precision, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3d_int_literals", test_vec3d_int_literals, devices=devices)
add_function_test(TestConstantPrecision, "test_vec3_int_literals", test_vec3_int_literals, devices=devices)
add_function_test(
    TestConstantPrecision, "test_int_literal_float64_constructor", test_int_literal_float64_constructor, devices=devices
)
add_function_test(TestConstantPrecision, "test_vec3d_fill", test_vec3d_fill, devices=devices)
add_function_test(
    TestConstantPrecision, "test_warp_constant_constructor", test_warp_constant_constructor, devices=devices
)
add_function_test(
    TestConstantPrecision,
    "test_typed_constructor_accepts_literals",
    test_typed_constructor_accepts_literals,
    devices=devices,
)
add_function_test(
    TestConstantPrecision, "test_vector_dtype_float_is_float32", test_vector_dtype_float_is_float32, devices=devices
)
add_function_test(TestConstantPrecision, "test_vec3d_negative_literals", test_vec3d_negative_literals, devices=devices)
add_function_test(
    TestConstantPrecision, "test_vec3h_mixed_literal_variable", test_vec3h_mixed_literal_variable, devices=devices
)
add_function_test(
    TestConstantPrecision, "test_vec3d_mixed_literal_variable", test_vec3d_mixed_literal_variable, devices=devices
)
add_function_test(
    TestConstantPrecision, "test_quatd_mixed_literal_variable", test_quatd_mixed_literal_variable, devices=devices
)
add_function_test(
    TestConstantPrecision,
    "test_typed_constructor_rejects_mismatched_variable",
    test_typed_constructor_rejects_mismatched_variable,
    devices=devices,
)
add_function_test(
    TestConstantPrecision,
    "test_matrix_constructor_rejects_mismatched_variable",
    test_matrix_constructor_rejects_mismatched_variable,
    devices=devices,
)
add_function_test(
    TestConstantPrecision,
    "test_matrix_fill_rejects_mismatched_variable",
    test_matrix_fill_rejects_mismatched_variable,
    devices=devices,
)
add_function_test(
    TestConstantPrecision,
    "test_mixed_literal_variable_rejects_wrong_type",
    test_mixed_literal_variable_rejects_wrong_type,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
