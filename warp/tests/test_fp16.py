# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def load_store_half(f32: wp.array(dtype=wp.float32), f16: wp.array(dtype=wp.float16)):
    tid = wp.tid()

    # check conversion from f32->f16
    a = wp.float16(f32[tid])
    b = f16[tid]

    wp.expect_eq(a, b)

    # check stores
    f16[tid] = a


def test_fp16_conversion(test, device):
    s = [1.0, 2.0, 3.0, -3.14159]

    np_f32 = np.array(s, dtype=np.float32)
    np_f16 = np.array(s, dtype=np.float16)

    wp_f32 = wp.array(s, dtype=wp.float32, device=device)
    wp_f16 = wp.array(s, dtype=wp.float16, device=device)

    assert_np_equal(np_f32, wp_f32.numpy())
    assert_np_equal(np_f16, wp_f16.numpy())

    wp.launch(load_store_half, dim=len(s), inputs=[wp_f32, wp_f16], device=device)

    # check that stores worked
    assert_np_equal(np_f16, wp_f16.numpy())


@wp.kernel
def value_load_store_half(f16_value: wp.float16, f16_array: wp.array(dtype=wp.float16)):
    wp.expect_eq(f16_value, f16_array[0])

    # check stores
    f16_array[0] = f16_value


def test_fp16_kernel_parameter(test, device):
    """Test the ability to pass in fp16 into kernels as parameters"""

    s = [1.0, 2.0, 3.0, -3.14159]

    for test_val in s:
        np_f16 = np.array([test_val], dtype=np.float16)
        wp_f16 = wp.array([test_val], dtype=wp.float16, device=device)

        wp.launch(value_load_store_half, (1,), inputs=[wp.float16(test_val), wp_f16], device=device)

        # check that stores worked
        assert_np_equal(np_f16, wp_f16.numpy())

        # Do the same thing but pass in test_val as a Python float to test automatic conversion
        wp_f16 = wp.array([test_val], dtype=wp.float16, device=device)
        wp.launch(value_load_store_half, (1,), inputs=[test_val, wp_f16], device=device)
        assert_np_equal(np_f16, wp_f16.numpy())


@wp.kernel
def mul_half(input: wp.array(dtype=wp.float16), output: wp.array(dtype=wp.float16)):
    tid = wp.tid()

    # convert to compute type fp32
    x = wp.float(input[tid]) * 2.0

    # store back as fp16
    output[tid] = wp.float16(x)


def test_fp16_grad(test, device):
    rng = np.random.default_rng(123)

    # checks that gradients are correctly propagated for
    # fp16 arrays, even when intermediate calculations
    # are performed in e.g.: fp32

    s = rng.random(size=15).astype(np.float16)

    input = wp.array(s, dtype=wp.float16, device=device, requires_grad=True)
    output = wp.zeros_like(input)

    tape = wp.Tape()
    with tape:
        wp.launch(mul_half, dim=len(s), inputs=[input, output], device=device)

    ones = wp.array(np.ones(len(output)), dtype=wp.float16, device=device)

    tape.backward(grads={output: ones})

    assert_np_equal(input.grad.numpy(), np.ones(len(s)) * 2.0)


class TestFp16(unittest.TestCase):
    pass


devices = []
if wp.is_cpu_available():
    devices.append("cpu")
for cuda_device in get_selected_cuda_test_devices():
    if cuda_device.arch >= 70:
        devices.append(cuda_device)

add_function_test(TestFp16, "test_fp16_conversion", test_fp16_conversion, devices=devices)
add_function_test(TestFp16, "test_fp16_grad", test_fp16_grad, devices=devices)
add_function_test(TestFp16, "test_fp16_kernel_parameter", test_fp16_kernel_parameter, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
