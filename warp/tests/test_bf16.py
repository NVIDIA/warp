# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp

try:
    import ml_dtypes as _ml_dtypes

    _has_ml_dtypes = True
except ImportError:
    _has_ml_dtypes = False
from warp._src.types import _np_bfloat16_bits_to_float32, _np_float32_to_bfloat16_bits, dtype_from_numpy
from warp.tests.unittest_utils import *


@wp.kernel
def bf16_conversion_kernel(input: wp.array[wp.float32], output: wp.array[wp.bfloat16]):
    tid = wp.tid()
    output[tid] = wp.bfloat16(input[tid])


@wp.kernel
def bf16_to_f32_kernel(input: wp.array[wp.bfloat16], output: wp.array[wp.float32]):
    tid = wp.tid()
    output[tid] = wp.float32(input[tid])


@wp.kernel
def bf16_param_kernel(x: wp.bfloat16, output: wp.array[wp.bfloat16]):
    output[0] = x


@wp.kernel
def bf16_arithmetic_kernel(
    a: wp.array[wp.bfloat16],
    b: wp.array[wp.bfloat16],
    out_add: wp.array[wp.bfloat16],
    out_sub: wp.array[wp.bfloat16],
    out_mul: wp.array[wp.bfloat16],
    out_div: wp.array[wp.bfloat16],
):
    tid = wp.tid()
    out_add[tid] = a[tid] + b[tid]
    out_sub[tid] = a[tid] - b[tid]
    out_mul[tid] = a[tid] * b[tid]
    out_div[tid] = a[tid] / b[tid]


def test_bf16_conversion(test, device):
    n = 10
    input_data = np.array([1.0, 2.0, 3.0, -1.0, 0.0, 0.5, 100.0, -100.0, 0.001, 1000.0], dtype=np.float32)
    input_arr = wp.array(input_data, dtype=wp.float32, device=device)
    output_arr = wp.zeros(n, dtype=wp.bfloat16, device=device)

    wp.launch(bf16_conversion_kernel, dim=n, inputs=[input_arr, output_arr], device=device)

    result_arr = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[output_arr, result_arr], device=device)
    result = result_arr.numpy()

    np.testing.assert_allclose(result, input_data, rtol=1e-2)


def test_bf16_kernel_parameter(test, device):
    output = wp.zeros(1, dtype=wp.bfloat16, device=device)
    wp.launch(bf16_param_kernel, dim=1, inputs=[wp.bfloat16(3.14), output], device=device)

    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=1, inputs=[output, result], device=device)
    np.testing.assert_allclose(result.numpy()[0], 3.14, rtol=1e-2)


def test_bf16_arithmetic(test, device):
    n = 4
    a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_data = np.array([0.5, 1.5, 2.0, 0.25], dtype=np.float32)

    a = wp.array(a_data, dtype=wp.bfloat16, device=device)
    b = wp.array(b_data, dtype=wp.bfloat16, device=device)

    out_add = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_sub = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_mul = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_div = wp.zeros(n, dtype=wp.bfloat16, device=device)

    wp.launch(bf16_arithmetic_kernel, dim=n, inputs=[a, b, out_add, out_sub, out_mul, out_div], device=device)

    add_f32 = wp.zeros(n, dtype=wp.float32, device=device)
    sub_f32 = wp.zeros(n, dtype=wp.float32, device=device)
    mul_f32 = wp.zeros(n, dtype=wp.float32, device=device)
    div_f32 = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_add, add_f32], device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_sub, sub_f32], device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_mul, mul_f32], device=device)
    wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_div, div_f32], device=device)

    np.testing.assert_allclose(add_f32.numpy(), a_data + b_data, rtol=1e-2)
    np.testing.assert_allclose(sub_f32.numpy(), a_data - b_data, rtol=1e-2)
    np.testing.assert_allclose(mul_f32.numpy(), a_data * b_data, rtol=1e-2)
    np.testing.assert_allclose(div_f32.numpy(), a_data / b_data, rtol=1e-2)


@wp.kernel
def bf16_grad_kernel(
    x: wp.array[wp.bfloat16],
    loss: wp.array[wp.float32],
):
    tid = wp.tid()
    v = wp.float32(x[tid])
    wp.atomic_add(loss, 0, v * v)


@wp.kernel
def bf16_atomic_kernel(output: wp.array[wp.bfloat16]):
    wp.atomic_add(output, 0, wp.bfloat16(1.0))


def test_bf16_numpy(test, device):
    n = 4
    arr = wp.zeros(n, dtype=wp.bfloat16, device=device)
    np_arr = arr.numpy()
    # dtype is ml_dtypes.bfloat16 when ml_dtypes is installed, np.uint16 otherwise
    if _has_ml_dtypes:
        test.assertEqual(np_arr.dtype, _ml_dtypes.bfloat16)
    else:
        test.assertEqual(np_arr.dtype, np.uint16)


def test_bf16_array_from_list(test, device):
    """Test that wp.array([floats], dtype=wp.bfloat16) correctly converts float values."""
    input_values = [1.0, 2.0, 3.0, -1.0, 0.0, 0.5, 100.0, -100.0]
    arr = wp.array(input_values, dtype=wp.bfloat16, device=device)

    test.assertEqual(arr.dtype, wp.bfloat16)
    test.assertEqual(arr.shape, (len(input_values),))

    # Verify the values survived the conversion by reading them back through a kernel
    result = wp.zeros(len(input_values), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_values), inputs=[arr, result], device=device)

    np.testing.assert_allclose(result.numpy(), input_values, rtol=1e-2)


def test_bf16_array_from_tuple(test, device):
    """Test that wp.array(tuple_of_floats, dtype=wp.bfloat16) works."""
    input_values = (1.5, 2.5, -3.5, 0.0)
    arr = wp.array(input_values, dtype=wp.bfloat16, device=device)

    result = wp.zeros(len(input_values), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_values), inputs=[arr, result], device=device)

    np.testing.assert_allclose(result.numpy(), input_values, rtol=1e-2)


def test_bf16_array_from_numpy_float(test, device):
    """Test that wp.array(np.array([...], dtype=np.float32), dtype=wp.bfloat16) works."""
    input_data = np.array([1.0, 2.5, 3.14, -0.5, 0.0], dtype=np.float32)
    arr = wp.array(input_data, dtype=wp.bfloat16, device=device)

    result = wp.zeros(len(input_data), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_data), inputs=[arr, result], device=device)

    np.testing.assert_allclose(result.numpy(), input_data, rtol=1e-2)


def test_bf16_array_from_numpy_uint16(test, device):
    """Test that wp.array(np.array([...], dtype=np.uint16), dtype=wp.bfloat16) passes raw bits through."""
    # Pre-encoded bfloat16 bits: 0x3F80 = 1.0, 0x4000 = 2.0, 0x4040 = 3.0
    raw_bits = np.array([0x3F80, 0x4000, 0x4040], dtype=np.uint16)
    arr = wp.array(raw_bits, dtype=wp.bfloat16, device=device)

    result = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=3, inputs=[arr, result], device=device)

    np.testing.assert_allclose(result.numpy(), [1.0, 2.0, 3.0], rtol=1e-2)


def test_bf16_array_special_values(test, device):
    """Test that special float values (inf, -inf, very small) convert correctly."""
    input_values = [float("inf"), float("-inf"), 0.0, -0.0, 1e-6]
    arr = wp.array(input_values, dtype=wp.bfloat16, device=device)

    result = wp.zeros(len(input_values), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_values), inputs=[arr, result], device=device)

    r = result.numpy()
    test.assertTrue(np.isinf(r[0]) and r[0] > 0, "Expected +inf")
    test.assertTrue(np.isinf(r[1]) and r[1] < 0, "Expected -inf")
    test.assertEqual(r[2], 0.0)
    # -0.0 should preserve sign
    test.assertEqual(r[3], 0.0)
    test.assertTrue(np.signbit(r[3]), "Expected negative zero to preserve sign bit")
    # Very small value — nearest bfloat16 is ~9.98e-7 (0.16% relative error)
    np.testing.assert_allclose(r[4], 1e-6, rtol=0.01)


def test_bf16_grad(test, device):
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = wp.array(x_data, dtype=wp.bfloat16, device=device, requires_grad=True)

    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(bf16_grad_kernel, dim=3, inputs=[x, loss], device=device)

    tape.backward(loss)

    x_grad = tape.gradients[x]
    grad_f32 = wp.zeros(3, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=3, inputs=[x_grad, grad_f32], device=device)

    # Values [1.0, 2.0, 3.0] and expected gradients [2.0, 4.0, 6.0]
    # are exactly representable in bfloat16, so use tighter tolerance.
    np.testing.assert_allclose(grad_f32.numpy(), 2.0 * x_data, rtol=1e-2)


def test_bf16_atomics(test, device):
    output = wp.zeros(1, dtype=wp.bfloat16, device=device)
    wp.launch(bf16_atomic_kernel, dim=10, inputs=[output], device=device)

    result = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=1, inputs=[output, result], device=device)
    test.assertEqual(result.numpy()[0], 10.0)


@wp.kernel
def bf16_math_kernel(
    input: wp.array[wp.bfloat16],
    out_sin: wp.array[wp.bfloat16],
    out_cos: wp.array[wp.bfloat16],
    out_sqrt: wp.array[wp.bfloat16],
    out_exp: wp.array[wp.bfloat16],
    out_log: wp.array[wp.bfloat16],
    out_pow: wp.array[wp.bfloat16],
    out_tan: wp.array[wp.bfloat16],
    out_abs: wp.array[wp.bfloat16],
    out_div_approx: wp.array[wp.bfloat16],
):
    tid = wp.tid()
    x = input[tid]
    out_sin[tid] = wp.sin(x)
    out_cos[tid] = wp.cos(x)
    out_sqrt[tid] = wp.sqrt(wp.abs(x))
    out_exp[tid] = wp.exp(x)
    out_log[tid] = wp.log(x)
    out_pow[tid] = wp.pow(x, wp.bfloat16(2.0))
    out_tan[tid] = wp.tan(x)
    out_abs[tid] = wp.abs(x)
    out_div_approx[tid] = wp.div_approx(x, wp.bfloat16(2.0))


def test_bf16_math_builtins(test, device):
    # Use positive values to avoid domain errors for log; avoid pi/2 for tan
    n = 4
    input_data = np.array([0.25, 0.5, 1.0, 1.5], dtype=np.float32)
    input_bf16 = wp.array(input_data, dtype=wp.bfloat16, device=device)

    out_sin = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_cos = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_sqrt = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_exp = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_log = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_pow = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_tan = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_abs = wp.zeros(n, dtype=wp.bfloat16, device=device)
    out_div_approx = wp.zeros(n, dtype=wp.bfloat16, device=device)

    wp.launch(
        bf16_math_kernel,
        dim=n,
        inputs=[input_bf16, out_sin, out_cos, out_sqrt, out_exp, out_log, out_pow, out_tan, out_abs, out_div_approx],
        device=device,
    )

    for out_arr, expected_fn in [
        (out_sin, np.sin),
        (out_cos, np.cos),
        (out_sqrt, lambda x: np.sqrt(np.abs(x))),
        (out_exp, np.exp),
        (out_log, np.log),
        (out_pow, lambda x: np.power(x, 2.0)),
        (out_tan, np.tan),
        (out_abs, np.abs),
        (out_div_approx, lambda x: x / 2.0),
    ]:
        result = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(bf16_to_f32_kernel, dim=n, inputs=[out_arr, result], device=device)
        # bfloat16 has ~2 decimal digits of precision; transcendental functions
        # incur additional rounding from input quantization + output quantization.
        # Measured max relative error is ~2.9e-3 (from exp); use 4e-3 for margin.
        np.testing.assert_allclose(result.numpy(), expected_fn(input_data), rtol=4e-3)


# Test bfloat16 in user-defined vector/matrix types
bf16_vec3 = wp.types.vector(3, dtype=wp.bfloat16)
bf16_mat22 = wp.types.matrix(shape=(2, 2), dtype=wp.bfloat16)


@wp.kernel
def bf16_vec3_kernel(
    out_x: wp.array[wp.float32],
    out_y: wp.array[wp.float32],
    out_z: wp.array[wp.float32],
):
    v = bf16_vec3(wp.bfloat16(1.0), wp.bfloat16(2.0), wp.bfloat16(3.0))
    out_x[0] = wp.float32(v[0])
    out_y[0] = wp.float32(v[1])
    out_z[0] = wp.float32(v[2])


@wp.kernel
def bf16_mat22_kernel(
    out_00: wp.array[wp.float32],
    out_01: wp.array[wp.float32],
    out_10: wp.array[wp.float32],
    out_11: wp.array[wp.float32],
):
    m = bf16_mat22(wp.bfloat16(1.0), wp.bfloat16(2.0), wp.bfloat16(3.0), wp.bfloat16(4.0))
    out_00[0] = wp.float32(m[0, 0])
    out_01[0] = wp.float32(m[0, 1])
    out_10[0] = wp.float32(m[1, 0])
    out_11[0] = wp.float32(m[1, 1])


def test_bf16_vector_matrix(test, device):
    out_x = wp.zeros(1, dtype=wp.float32, device=device)
    out_y = wp.zeros(1, dtype=wp.float32, device=device)
    out_z = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(bf16_vec3_kernel, dim=1, inputs=[out_x, out_y, out_z], device=device)

    np.testing.assert_allclose(out_x.numpy()[0], 1.0, rtol=1e-2)
    np.testing.assert_allclose(out_y.numpy()[0], 2.0, rtol=1e-2)
    np.testing.assert_allclose(out_z.numpy()[0], 3.0, rtol=1e-2)

    out_00 = wp.zeros(1, dtype=wp.float32, device=device)
    out_01 = wp.zeros(1, dtype=wp.float32, device=device)
    out_10 = wp.zeros(1, dtype=wp.float32, device=device)
    out_11 = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(bf16_mat22_kernel, dim=1, inputs=[out_00, out_01, out_10, out_11], device=device)

    np.testing.assert_allclose(out_00.numpy()[0], 1.0, rtol=1e-2)
    np.testing.assert_allclose(out_01.numpy()[0], 2.0, rtol=1e-2)
    np.testing.assert_allclose(out_10.numpy()[0], 3.0, rtol=1e-2)
    np.testing.assert_allclose(out_11.numpy()[0], 4.0, rtol=1e-2)


# Test bfloat16 struct fields
@wp.struct
class Bf16Struct:
    bf16_val: wp.bfloat16
    f32_val: wp.float32


@wp.kernel
def bf16_struct_kernel(
    s: Bf16Struct,
    out_bf16: wp.array[wp.float32],
    out_f32: wp.array[wp.float32],
):
    out_bf16[0] = wp.float32(s.bf16_val)
    out_f32[0] = s.f32_val


def test_bf16_struct(test, device):
    s = Bf16Struct()
    s.bf16_val = wp.bfloat16(3.5)
    s.f32_val = wp.float32(7.25)

    out_bf16 = wp.zeros(1, dtype=wp.float32, device=device)
    out_f32 = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(bf16_struct_kernel, dim=1, inputs=[s, out_bf16, out_f32], device=device)

    np.testing.assert_allclose(out_bf16.numpy()[0], 3.5, rtol=1e-2)
    np.testing.assert_allclose(out_f32.numpy()[0], 7.25, rtol=1e-7)


# Test comparison operators
@wp.kernel
def bf16_comparison_kernel(
    results: wp.array[wp.int32],
):
    a = wp.bfloat16(1.0)
    b = wp.bfloat16(2.0)
    c = wp.bfloat16(1.0)

    # equal values
    results[0] = wp.where(a == c, 1, 0)  # true -> 1
    # unequal values
    results[1] = wp.where(a != b, 1, 0)  # true -> 1
    # less than
    results[2] = wp.where(a < b, 1, 0)  # true -> 1
    # greater than
    results[3] = wp.where(b > a, 1, 0)  # true -> 1
    # less than or equal (equal case)
    results[4] = wp.where(a <= c, 1, 0)  # true -> 1
    # greater than or equal (equal case)
    results[5] = wp.where(a >= c, 1, 0)  # true -> 1
    # false cases
    results[6] = wp.where(a == b, 1, 0)  # false -> 0
    results[7] = wp.where(a > b, 1, 0)  # false -> 0

    # negative zero vs positive zero
    neg_zero = wp.bfloat16(-0.0)
    pos_zero = wp.bfloat16(0.0)
    results[8] = wp.where(neg_zero == pos_zero, 1, 0)  # true -> 1 (IEEE 754)


def test_bf16_comparisons(test, device):
    results = wp.zeros(9, dtype=wp.int32, device=device)
    wp.launch(bf16_comparison_kernel, dim=1, inputs=[results], device=device)

    r = results.numpy()
    # True cases
    test.assertEqual(r[0], 1, "a == c should be true")
    test.assertEqual(r[1], 1, "a != b should be true")
    test.assertEqual(r[2], 1, "a < b should be true")
    test.assertEqual(r[3], 1, "b > a should be true")
    test.assertEqual(r[4], 1, "a <= c should be true")
    test.assertEqual(r[5], 1, "a >= c should be true")
    # False cases
    test.assertEqual(r[6], 0, "a == b should be false")
    test.assertEqual(r[7], 0, "a > b should be false")
    # Negative zero == positive zero (IEEE 754)
    test.assertEqual(r[8], 1, "-0.0 == 0.0 should be true")


def test_bf16_nan_preservation(test, device):
    """Test that NaN values are preserved through bfloat16 conversion, including signaling NaNs
    whose top mantissa bits would be truncated to zero by a naive implementation."""
    import struct  # noqa: PLC0415

    # Quiet NaN (standard): 0x7FC00000 — mantissa bits survive truncation
    # Signaling NaN with low mantissa: 0x7F800001 — top 7 mantissa bits are zero,
    # so a naive truncation (bits >> 16) produces 0x7F80 = infinity
    snan_bits = 0x7F800001
    qnan_bits = 0x7FC00000
    snan_f32 = struct.unpack("f", struct.pack("I", snan_bits))[0]
    qnan_f32 = struct.unpack("f", struct.pack("I", qnan_bits))[0]

    input_data = np.array([snan_f32, qnan_f32, float("nan")], dtype=np.float32)
    bf16_arr = wp.array(input_data, dtype=wp.bfloat16, device=device)
    result = wp.zeros(len(input_data), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_data), inputs=[bf16_arr, result], device=device)

    r = result.numpy()
    for i in range(len(input_data)):
        test.assertTrue(np.isnan(r[i]), f"Expected NaN at index {i}, got {r[i]}")


def test_bf16_cross_device_consistency(test, device):
    """Test that bfloat16 conversion and arithmetic produce identical bits on CPU and CUDA."""
    # Test values include: normal, subnormal, negative, zero, large, rounding tie boundaries.
    # 1.00390625 = 0x3F808000: the LSB of the bf16 mantissa is 0, and the round bit is 1
    # with zero trailing bits, exercising the round-to-nearest-even tie-break (rounds to 1.0).
    input_f32 = np.array(
        [
            1.0,
            -1.0,
            0.0,
            3.140625,
            1.00390625,
            65504.0,
            1.0e-38,
            float("inf"),
            float("-inf"),
        ],
        dtype=np.float32,
    )

    # Conversion: float32 -> bfloat16 -> float32
    input_cpu = wp.array(input_f32, dtype=wp.float32, device="cpu")
    input_gpu = wp.array(input_f32, dtype=wp.float32, device=device)
    bf16_cpu = wp.zeros(len(input_f32), dtype=wp.bfloat16, device="cpu")
    bf16_gpu = wp.zeros(len(input_f32), dtype=wp.bfloat16, device=device)
    wp.launch(bf16_conversion_kernel, dim=len(input_f32), inputs=[input_cpu, bf16_cpu], device="cpu")
    wp.launch(bf16_conversion_kernel, dim=len(input_f32), inputs=[input_gpu, bf16_gpu], device=device)

    result_cpu = wp.zeros(len(input_f32), dtype=wp.float32, device="cpu")
    result_gpu = wp.zeros(len(input_f32), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=len(input_f32), inputs=[bf16_cpu, result_cpu], device="cpu")
    wp.launch(bf16_to_f32_kernel, dim=len(input_f32), inputs=[bf16_gpu, result_gpu], device=device)

    np.testing.assert_array_equal(result_cpu.numpy(), result_gpu.numpy())

    # Arithmetic: operate on identical bf16 inputs, compare bit-exact results
    a_f32 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_f32 = np.array([0.5, 1.5, 2.0, 0.25], dtype=np.float32)

    a_cpu = wp.array(a_f32, dtype=wp.bfloat16, device="cpu")
    b_cpu = wp.array(b_f32, dtype=wp.bfloat16, device="cpu")
    a_gpu = wp.array(a_f32, dtype=wp.bfloat16, device=device)
    b_gpu = wp.array(b_f32, dtype=wp.bfloat16, device=device)

    ops = ["add", "sub", "mul", "div"]
    cpu_results = {op: wp.zeros(len(a_f32), dtype=wp.bfloat16, device="cpu") for op in ops}
    gpu_results = {op: wp.zeros(len(a_f32), dtype=wp.bfloat16, device=device) for op in ops}

    wp.launch(
        bf16_arithmetic_kernel,
        dim=len(a_f32),
        inputs=[a_cpu, b_cpu, cpu_results["add"], cpu_results["sub"], cpu_results["mul"], cpu_results["div"]],
        device="cpu",
    )
    wp.launch(
        bf16_arithmetic_kernel,
        dim=len(a_f32),
        inputs=[a_gpu, b_gpu, gpu_results["add"], gpu_results["sub"], gpu_results["mul"], gpu_results["div"]],
        device=device,
    )

    for op in ops:
        cpu_f32 = wp.zeros(len(a_f32), dtype=wp.float32, device="cpu")
        gpu_f32 = wp.zeros(len(a_f32), dtype=wp.float32, device=device)
        wp.launch(bf16_to_f32_kernel, dim=len(a_f32), inputs=[cpu_results[op], cpu_f32], device="cpu")
        wp.launch(bf16_to_f32_kernel, dim=len(a_f32), inputs=[gpu_results[op], gpu_f32], device=device)
        np.testing.assert_array_equal(cpu_f32.numpy(), gpu_f32.numpy(), err_msg=f"Mismatch for {op}")


class TestBf16(unittest.TestCase):
    pass


def test_bf16_ml_dtypes_conversion_helpers(test, device):
    """Test that conversion helpers produce correct results (uses ml_dtypes internally when available)."""
    input_f32 = np.array([1.0, 2.5, -3.14, 0.0, float("inf"), float("-inf")], dtype=np.float32)

    # float32 -> bfloat16 bits -> float32 round-trip
    bf16_bits = _np_float32_to_bfloat16_bits(input_f32)
    test.assertEqual(bf16_bits.dtype, np.uint16)
    recovered = _np_bfloat16_bits_to_float32(bf16_bits)
    test.assertEqual(recovered.dtype, np.float32)

    # bfloat16 has ~2 decimal digits of precision, so rtol=1e-2 is appropriate
    finite_mask = np.isfinite(input_f32)
    np.testing.assert_allclose(recovered[finite_mask], input_f32[finite_mask], rtol=1e-2)
    # Infinities should be preserved exactly
    np.testing.assert_equal(recovered[~finite_mask], input_f32[~finite_mask])

    # NaN should remain NaN
    nan_bits = _np_float32_to_bfloat16_bits(np.array([float("nan")], dtype=np.float32))
    nan_recovered = _np_bfloat16_bits_to_float32(nan_bits)
    test.assertTrue(np.isnan(nan_recovered[0]))

    # Regression: signaling NaN with many mantissa bits set (e.g. 0x7FBFFFFF)
    # must stay NaN after conversion instead of overflowing into negative zero.
    snan_bits = np.array([0x7FBFFFFF, 0xFFBFFFFF], dtype=np.uint32)
    snan_f32 = snan_bits.view(np.float32)
    snan_bf16 = _np_float32_to_bfloat16_bits(snan_f32)
    snan_roundtrip = _np_bfloat16_bits_to_float32(snan_bf16)
    test.assertTrue(np.all(np.isnan(snan_roundtrip)), f"Expected NaN, got bits {snan_bf16}")


@wp.kernel
def bf16_math_extended_kernel(
    input: wp.array[wp.bfloat16],
    out_acos: wp.array[wp.bfloat16],
    out_asin: wp.array[wp.bfloat16],
    out_atan: wp.array[wp.bfloat16],
    out_atan2: wp.array[wp.bfloat16],
    out_cosh: wp.array[wp.bfloat16],
    out_sinh: wp.array[wp.bfloat16],
    out_tanh: wp.array[wp.bfloat16],
    out_log2: wp.array[wp.bfloat16],
    out_log10: wp.array[wp.bfloat16],
    out_cbrt: wp.array[wp.bfloat16],
    out_degrees: wp.array[wp.bfloat16],
    out_radians: wp.array[wp.bfloat16],
):
    tid = wp.tid()
    x = input[tid]
    out_acos[tid] = wp.acos(x)
    out_asin[tid] = wp.asin(x)
    out_atan[tid] = wp.atan(x)
    out_atan2[tid] = wp.atan2(x, wp.bfloat16(1.0))
    out_cosh[tid] = wp.cosh(x)
    out_sinh[tid] = wp.sinh(x)
    out_tanh[tid] = wp.tanh(x)
    out_log2[tid] = wp.log2(x)
    out_log10[tid] = wp.log10(x)
    out_cbrt[tid] = wp.cbrt(x)
    out_degrees[tid] = wp.degrees(x)
    out_radians[tid] = wp.radians(x)


def test_bf16_math_extended(test, device):
    """Test inverse trig, hyperbolic, log variants, cbrt, degrees/radians."""
    n = 4
    # Values in (0, 1] for asin/acos domain; positive for log
    input_data = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    input_bf16 = wp.array(input_data, dtype=wp.bfloat16, device=device)

    names = [
        "acos",
        "asin",
        "atan",
        "atan2",
        "cosh",
        "sinh",
        "tanh",
        "log2",
        "log10",
        "cbrt",
        "degrees",
        "radians",
    ]
    outputs = {name: wp.zeros(n, dtype=wp.bfloat16, device=device) for name in names}

    wp.launch(
        bf16_math_extended_kernel,
        dim=n,
        inputs=[input_bf16] + [outputs[name] for name in names],
        device=device,
    )

    expected = {
        "acos": np.arccos,
        "asin": np.arcsin,
        "atan": np.arctan,
        "atan2": lambda x: np.arctan2(x, 1.0),
        "cosh": np.cosh,
        "sinh": np.sinh,
        "tanh": np.tanh,
        "log2": np.log2,
        "log10": np.log10,
        "cbrt": np.cbrt,
        "degrees": np.degrees,
        "radians": np.radians,
    }

    for name in names:
        result = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(bf16_to_f32_kernel, dim=n, inputs=[outputs[name], result], device=device)
        np.testing.assert_allclose(result.numpy(), expected[name](input_data), rtol=4e-3, err_msg=f"wp.{name}")


@wp.kernel
def bf16_rounding_kernel(
    input: wp.array[wp.bfloat16],
    out_ceil: wp.array[wp.bfloat16],
    out_floor: wp.array[wp.bfloat16],
    out_round: wp.array[wp.bfloat16],
    out_rint: wp.array[wp.bfloat16],
    out_trunc: wp.array[wp.bfloat16],
    out_frac: wp.array[wp.bfloat16],
    out_neg: wp.array[wp.bfloat16],
    out_sign: wp.array[wp.bfloat16],
    out_step: wp.array[wp.bfloat16],
    out_nonzero: wp.array[wp.bfloat16],
    out_clamp: wp.array[wp.bfloat16],
    out_min: wp.array[wp.bfloat16],
    out_max: wp.array[wp.bfloat16],
    out_mod: wp.array[wp.bfloat16],
    out_floordiv: wp.array[wp.bfloat16],
):
    tid = wp.tid()
    x = input[tid]
    out_ceil[tid] = wp.ceil(x)
    out_floor[tid] = wp.floor(x)
    out_round[tid] = wp.round(x)
    out_rint[tid] = wp.rint(x)
    out_trunc[tid] = wp.trunc(x)
    out_frac[tid] = wp.frac(x)
    out_neg[tid] = -x
    out_sign[tid] = wp.sign(x)
    out_step[tid] = wp.step(x)
    out_nonzero[tid] = wp.nonzero(x)
    out_clamp[tid] = wp.clamp(x, wp.bfloat16(-1.0), wp.bfloat16(1.0))
    out_min[tid] = wp.min(x, wp.bfloat16(0.5))
    out_max[tid] = wp.max(x, wp.bfloat16(0.5))
    out_mod[tid] = wp.mod(x, wp.bfloat16(1.0))
    out_floordiv[tid] = wp.floordiv(x, wp.bfloat16(2.0))


def test_bf16_rounding_and_arithmetic(test, device):
    """Test rounding, sign, clamp, min/max, mod, floordiv, negation."""
    n = 4
    input_data = np.array([-1.75, -0.25, 0.75, 2.25], dtype=np.float32)
    input_bf16 = wp.array(input_data, dtype=wp.bfloat16, device=device)

    names = [
        "ceil",
        "floor",
        "round",
        "rint",
        "trunc",
        "frac",
        "neg",
        "sign",
        "step",
        "nonzero",
        "clamp",
        "min",
        "max",
        "mod",
        "floordiv",
    ]
    outputs = {name: wp.zeros(n, dtype=wp.bfloat16, device=device) for name in names}

    wp.launch(
        bf16_rounding_kernel,
        dim=n,
        inputs=[input_bf16] + [outputs[name] for name in names],
        device=device,
    )

    expected = {
        "ceil": np.ceil,
        "floor": np.floor,
        "round": lambda x: np.sign(x) * np.floor(np.abs(x) + 0.5),  # C roundf: half away from zero
        "rint": np.rint,
        "trunc": np.trunc,
        "frac": lambda x: x - np.trunc(x),
        "neg": lambda x: -x,
        "sign": np.sign,
        "step": lambda x: np.where(x < 0.0, 1.0, 0.0),
        "nonzero": lambda x: np.where(x != 0.0, 1.0, 0.0),
        "clamp": lambda x: np.clip(x, -1.0, 1.0),
        "min": lambda x: np.minimum(x, 0.5),
        "max": lambda x: np.maximum(x, 0.5),
        "mod": lambda x: np.fmod(x, 1.0),
        "floordiv": lambda x: np.floor_divide(x, 2.0),
    }

    # All rounding/arithmetic ops produce exact results for these inputs
    for name in names:
        result = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(bf16_to_f32_kernel, dim=n, inputs=[outputs[name], result], device=device)
        np.testing.assert_allclose(result.numpy(), expected[name](input_data), rtol=0, atol=0, err_msg=f"wp.{name}")


@wp.kernel
def bf16_special_math_kernel(
    input_a: wp.array[wp.bfloat16],
    input_b: wp.array[wp.bfloat16],
    out_lerp: wp.array[wp.bfloat16],
    out_smoothstep: wp.array[wp.bfloat16],
    out_erf: wp.array[wp.bfloat16],
    out_erfc: wp.array[wp.bfloat16],
    out_erfinv: wp.array[wp.bfloat16],
    out_erfcinv: wp.array[wp.bfloat16],
):
    tid = wp.tid()
    a = input_a[tid]
    b = input_b[tid]
    out_lerp[tid] = wp.lerp(wp.bfloat16(0.0), wp.bfloat16(2.0), a)
    out_smoothstep[tid] = wp.smoothstep(wp.bfloat16(0.0), wp.bfloat16(1.0), a)
    out_erf[tid] = wp.erf(a)
    out_erfc[tid] = wp.erfc(a)
    # erfinv/erfcinv need values in (-1, 1) and (0, 2) respectively
    out_erfinv[tid] = wp.erfinv(b)
    out_erfcinv[tid] = wp.erfcinv(wp.bfloat16(1.0) + b)


def test_bf16_special_math(test, device):
    """Test lerp, smoothstep, erf, erfc, erfinv, erfcinv."""
    n = 4
    input_a_data = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    # Values in (-1, 1) for erfinv; erfcinv uses 1+b so in (0, 2)
    input_b_data = np.array([-0.5, -0.25, 0.25, 0.5], dtype=np.float32)
    input_a_bf16 = wp.array(input_a_data, dtype=wp.bfloat16, device=device)
    input_b_bf16 = wp.array(input_b_data, dtype=wp.bfloat16, device=device)

    names = ["lerp", "smoothstep", "erf", "erfc", "erfinv", "erfcinv"]
    outputs = {name: wp.zeros(n, dtype=wp.bfloat16, device=device) for name in names}

    wp.launch(
        bf16_special_math_kernel,
        dim=n,
        inputs=[input_a_bf16, input_b_bf16] + [outputs[name] for name in names],
        device=device,
    )

    def np_smoothstep(x):
        t = np.clip(x, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # Reference values computed with high-precision math libraries
    # erf([0.0, 0.25, 0.5, 1.0])
    ref_erf = np.array([0.0, 0.2763263902, 0.5204998778, 0.8427007929], dtype=np.float32)
    # erfc([0.0, 0.25, 0.5, 1.0])
    ref_erfc = np.array([1.0, 0.7236736098, 0.4795001222, 0.1572992071], dtype=np.float32)
    # erfinv([-0.5, -0.25, 0.25, 0.5])
    ref_erfinv = np.array([-0.4769362762, -0.2253121285, 0.2253121285, 0.4769362762], dtype=np.float32)
    # erfcinv(1+b) = erfcinv([0.5, 0.75, 1.25, 1.5])
    ref_erfcinv = np.array([0.4769362762, 0.2253121285, -0.2253121285, -0.4769362762], dtype=np.float32)

    expected = {
        "lerp": lambda: input_a_data * 2.0,
        "smoothstep": lambda: np_smoothstep(input_a_data),
        "erf": lambda: ref_erf,
        "erfc": lambda: ref_erfc,
        "erfinv": lambda: ref_erfinv,
        "erfcinv": lambda: ref_erfcinv,
    }

    # lerp and smoothstep are exact for these inputs; error functions have
    # quantization error (measured max_rel ~3.4e-3 for erf)
    exact_names = {"lerp", "smoothstep"}
    for name in names:
        result = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(bf16_to_f32_kernel, dim=n, inputs=[outputs[name], result], device=device)
        if name in exact_names:
            np.testing.assert_allclose(result.numpy(), expected[name](), rtol=0, atol=0, err_msg=f"wp.{name}")
        else:
            np.testing.assert_allclose(result.numpy(), expected[name](), rtol=4e-3, atol=0, err_msg=f"wp.{name}")


@wp.kernel
def bf16_classification_kernel(
    input: wp.array[wp.bfloat16],
    out_isfinite: wp.array[wp.bool],
    out_isinf: wp.array[wp.bool],
    out_isnan: wp.array[wp.bool],
):
    tid = wp.tid()
    x = input[tid]
    out_isfinite[tid] = wp.isfinite(x)
    out_isinf[tid] = wp.isinf(x)
    out_isnan[tid] = wp.isnan(x)


def test_bf16_classification(test, device):
    """Test isfinite, isinf, isnan with bfloat16."""
    input_data = np.array([1.0, float("inf"), float("-inf"), float("nan")], dtype=np.float32)
    n = len(input_data)
    input_bf16 = wp.array(input_data, dtype=wp.bfloat16, device=device)

    out_isfinite = wp.zeros(n, dtype=wp.bool, device=device)
    out_isinf = wp.zeros(n, dtype=wp.bool, device=device)
    out_isnan = wp.zeros(n, dtype=wp.bool, device=device)

    wp.launch(
        bf16_classification_kernel,
        dim=n,
        inputs=[input_bf16, out_isfinite, out_isinf, out_isnan],
        device=device,
    )

    np.testing.assert_array_equal(out_isfinite.numpy(), [True, False, False, False])
    np.testing.assert_array_equal(out_isinf.numpy(), [False, True, True, False])
    np.testing.assert_array_equal(out_isnan.numpy(), [False, False, False, True])


cuda_bf16_devices = [d for d in get_selected_cuda_test_devices() if d.arch >= 80]
devices = [wp.get_device("cpu"), *cuda_bf16_devices]
add_function_test(TestBf16, "test_bf16_conversion", test_bf16_conversion, devices=devices)
add_function_test(TestBf16, "test_bf16_kernel_parameter", test_bf16_kernel_parameter, devices=devices)
add_function_test(TestBf16, "test_bf16_arithmetic", test_bf16_arithmetic, devices=devices)
add_function_test(TestBf16, "test_bf16_numpy", test_bf16_numpy, devices=devices, check_output=False)
add_function_test(TestBf16, "test_bf16_array_from_list", test_bf16_array_from_list, devices=devices)
add_function_test(TestBf16, "test_bf16_array_from_tuple", test_bf16_array_from_tuple, devices=devices)
add_function_test(TestBf16, "test_bf16_array_from_numpy_float", test_bf16_array_from_numpy_float, devices=devices)
add_function_test(TestBf16, "test_bf16_array_from_numpy_uint16", test_bf16_array_from_numpy_uint16, devices=devices)
add_function_test(TestBf16, "test_bf16_array_special_values", test_bf16_array_special_values, devices=devices)
add_function_test(TestBf16, "test_bf16_grad", test_bf16_grad, devices=devices)
add_function_test(TestBf16, "test_bf16_atomics", test_bf16_atomics, devices=devices)
add_function_test(TestBf16, "test_bf16_math_builtins", test_bf16_math_builtins, devices=devices)
add_function_test(TestBf16, "test_bf16_math_extended", test_bf16_math_extended, devices=devices)
add_function_test(TestBf16, "test_bf16_rounding_and_arithmetic", test_bf16_rounding_and_arithmetic, devices=devices)
add_function_test(TestBf16, "test_bf16_special_math", test_bf16_special_math, devices=devices)
add_function_test(TestBf16, "test_bf16_classification", test_bf16_classification, devices=devices)
add_function_test(TestBf16, "test_bf16_vector_matrix", test_bf16_vector_matrix, devices=devices)
add_function_test(TestBf16, "test_bf16_struct", test_bf16_struct, devices=devices)
add_function_test(TestBf16, "test_bf16_comparisons", test_bf16_comparisons, devices=devices)
add_function_test(TestBf16, "test_bf16_nan_preservation", test_bf16_nan_preservation, devices=devices)
add_function_test(
    TestBf16, "test_bf16_cross_device_consistency", test_bf16_cross_device_consistency, devices=cuda_bf16_devices
)
add_function_test(
    TestBf16, "test_bf16_ml_dtypes_conversion_helpers", test_bf16_ml_dtypes_conversion_helpers, devices=devices
)


@unittest.skipUnless(_has_ml_dtypes, "ml_dtypes not installed")
class TestBf16MlDtypes(unittest.TestCase):
    """Tests for ml_dtypes interop (skipped when ml_dtypes is not installed)."""

    def test_dtype_from_numpy(self):
        """Test that dtype_from_numpy recognizes ml_dtypes.bfloat16."""
        self.assertIs(dtype_from_numpy(_ml_dtypes.bfloat16), wp.bfloat16)
        # Also test the np.dtype form
        self.assertIs(dtype_from_numpy(np.dtype(_ml_dtypes.bfloat16)), wp.bfloat16)

    def test_array_from_ml_dtypes_infer_dtype(self):
        """Test that wp.array(ml_dtypes_bf16_array) infers dtype=wp.bfloat16."""
        ml_bf16 = _ml_dtypes.bfloat16
        data = np.array([1.0, 2.0, 3.0, -1.0], dtype=ml_bf16)
        arr = wp.array(data, device="cpu")

        self.assertIs(arr.dtype, wp.bfloat16)
        # Verify values round-trip correctly
        result = arr.list()
        np.testing.assert_allclose([float(v) for v in result], [1.0, 2.0, 3.0, -1.0], rtol=1e-2)

    def test_array_from_ml_dtypes_explicit_dtype(self):
        """Test that wp.array(ml_dtypes_bf16_array, dtype=wp.bfloat16) works."""
        ml_bf16 = _ml_dtypes.bfloat16
        data = np.array([1.5, 2.5, -0.5], dtype=ml_bf16)
        arr = wp.array(data, dtype=wp.bfloat16, device="cpu")

        self.assertIs(arr.dtype, wp.bfloat16)
        result = arr.list()
        np.testing.assert_allclose([float(v) for v in result], [1.5, 2.5, -0.5], rtol=1e-2)

    def test_numpy_returns_ml_dtypes(self):
        """Test that .numpy() returns ml_dtypes.bfloat16 dtype when ml_dtypes is available."""
        arr = wp.array([1.0, 2.0, 3.0], dtype=wp.bfloat16, device="cpu")
        np_arr = arr.numpy()

        self.assertEqual(np_arr.dtype, _ml_dtypes.bfloat16)
        # Values should be readable as floats
        np.testing.assert_allclose(np_arr.astype(np.float32), [1.0, 2.0, 3.0], rtol=1e-2)

    def test_numpy_round_trip(self):
        """Test wp.array -> .numpy() -> wp.array round-trip preserves values."""
        original_values = [1.0, -2.5, 0.0, 100.0, 0.001]
        arr1 = wp.array(original_values, dtype=wp.bfloat16, device="cpu")
        np_arr = arr1.numpy()

        self.assertEqual(np_arr.dtype, _ml_dtypes.bfloat16)

        # Construct a new warp array from the ml_dtypes numpy array
        arr2 = wp.array(np_arr, device="cpu")
        self.assertIs(arr2.dtype, wp.bfloat16)

        # Values should match
        result1 = [float(v) for v in arr1.list()]
        result2 = [float(v) for v in arr2.list()]
        np.testing.assert_allclose(result1, result2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
