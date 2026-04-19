# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for compiling CUDA kernels with the bundled Clang/LLVM compiler."""

import unittest

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def trivial_kernel(a: wp.array[float], b: wp.array[float]):
    i = wp.tid()
    b[i] = a[i] * 2.0


@wp.kernel
def math_kernel(
    a: wp.array[float],
    out_sin: wp.array[float],
    out_cos: wp.array[float],
    out_sqrt: wp.array[float],
):
    i = wp.tid()
    out_sin[i] = wp.sin(a[i])
    out_cos[i] = wp.cos(a[i])
    out_sqrt[i] = wp.sqrt(wp.abs(a[i]))


@wp.kernel
def vec_kernel(
    positions: wp.array[wp.vec3],
    result: wp.array[float],
):
    i = wp.tid()
    p = positions[i]
    result[i] = wp.length(p)


@wp.kernel
def conditional_kernel(
    a: wp.array[float],
    b: wp.array[float],
):
    i = wp.tid()
    if a[i] > 0.0:
        b[i] = a[i]
    else:
        b[i] = -a[i]


def test_trivial_kernel(test, device):
    n = 10
    a = wp.array(np.arange(n, dtype=np.float32), device=device)
    b = wp.zeros(n, dtype=float, device=device)
    wp.launch(trivial_kernel, dim=n, inputs=[a, b], device=device)
    result = b.numpy()
    np.testing.assert_allclose(result, np.arange(n, dtype=np.float32) * 2.0)


def test_math_kernel(test, device):
    n = 8
    a_np = np.array([0.0, 0.5, 1.0, 1.5, 2.0, -1.0, 3.14, 0.25], dtype=np.float32)
    a = wp.array(a_np, device=device)
    out_sin = wp.zeros(n, dtype=float, device=device)
    out_cos = wp.zeros(n, dtype=float, device=device)
    out_sqrt = wp.zeros(n, dtype=float, device=device)
    wp.launch(math_kernel, dim=n, inputs=[a, out_sin, out_cos, out_sqrt], device=device)
    np.testing.assert_allclose(out_sin.numpy(), np.sin(a_np), rtol=1e-5)
    np.testing.assert_allclose(out_cos.numpy(), np.cos(a_np), rtol=1e-5)
    np.testing.assert_allclose(out_sqrt.numpy(), np.sqrt(np.abs(a_np)), rtol=1e-5)


def test_vec_kernel(test, device):
    positions = wp.array(
        [[1.0, 0.0, 0.0], [0.0, 3.0, 4.0], [1.0, 1.0, 1.0]],
        dtype=wp.vec3,
        device=device,
    )
    result = wp.zeros(3, dtype=float, device=device)
    wp.launch(vec_kernel, dim=3, inputs=[positions, result], device=device)
    expected = np.array([1.0, 5.0, np.sqrt(3.0)], dtype=np.float32)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


def test_conditional_kernel(test, device):
    a_np = np.array([1.0, -2.0, 3.0, -4.0, 0.0], dtype=np.float32)
    a = wp.array(a_np, device=device)
    b = wp.zeros(5, dtype=float, device=device)
    wp.launch(conditional_kernel, dim=5, inputs=[a, b], device=device)
    np.testing.assert_allclose(b.numpy(), np.abs(a_np))


@wp.kernel
def bf16_round_trip_kernel(input: wp.array[wp.float32], output: wp.array[wp.float32]):
    i = wp.tid()
    bf = wp.bfloat16(input[i])
    output[i] = wp.float32(bf)


@wp.kernel
def bf16_arithmetic_kernel(
    a: wp.array[wp.float32],
    b: wp.array[wp.float32],
    output: wp.array[wp.float32],
):
    i = wp.tid()
    output[i] = wp.float32(wp.bfloat16(a[i]) + wp.bfloat16(b[i]))


def test_bf16_round_trip(test, device):
    input_data = np.array([1.0, 2.0, -3.0, 0.5, 100.0], dtype=np.float32)
    a = wp.array(input_data, device=device)
    b = wp.zeros(5, dtype=float, device=device)
    wp.launch(bf16_round_trip_kernel, dim=5, inputs=[a, b], device=device)
    np.testing.assert_allclose(b.numpy(), input_data, rtol=1e-2)


def test_bf16_arithmetic(test, device):
    a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_np = np.array([0.5, 1.5, 2.0, 0.25], dtype=np.float32)
    a = wp.array(a_np, device=device)
    b = wp.array(b_np, device=device)
    out = wp.zeros(4, dtype=float, device=device)
    wp.launch(bf16_arithmetic_kernel, dim=4, inputs=[a, b, out], device=device)
    np.testing.assert_allclose(out.numpy(), a_np + b_np, rtol=1e-2)


class TestClangCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_llvm_cuda = wp.config.llvm_cuda
        wp.config.llvm_cuda = True

    @classmethod
    def tearDownClass(cls):
        wp.config.llvm_cuda = cls._saved_llvm_cuda


devices = get_selected_cuda_test_devices()
add_function_test(TestClangCUDA, "test_trivial_kernel", test_trivial_kernel, devices=devices)
add_function_test(TestClangCUDA, "test_math_kernel", test_math_kernel, devices=devices)
add_function_test(TestClangCUDA, "test_vec_kernel", test_vec_kernel, devices=devices)
add_function_test(TestClangCUDA, "test_conditional_kernel", test_conditional_kernel, devices=devices)
bf16_devices = [d for d in devices if d.arch >= 80]
add_function_test(TestClangCUDA, "test_bf16_round_trip", test_bf16_round_trip, devices=bf16_devices)
add_function_test(TestClangCUDA, "test_bf16_arithmetic", test_bf16_arithmetic, devices=bf16_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
