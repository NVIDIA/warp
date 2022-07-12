import warp as wp
import numpy as np

import unittest

import warp as wp
from warp.tests.test_base import *


wp.init()

@wp.kernel
def load_store_half(f32: wp.array(dtype=wp.float32),
            f16: wp.array(dtype=wp.float16)):

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
def mul_half(input: wp.array(dtype=wp.float16),
             output: wp.array(dtype=wp.float16)):

    tid = wp.tid()

    # convert to compute type fp32
    x = wp.float(input[tid])*2.0

    # store back as fp16
    output[tid] = wp.float16(x)


def test_fp16_grad(test, device):
    # checks that gradients are correctly propagated for
    # fp16 arrays, even when intermediate calcualtions
    # are performed in e.g.: fp32

    s = np.random.rand(15).astype(np.float16)

    input = wp.array(s, dtype=wp.float16, device=device, requires_grad=True)
    output = wp.zeros_like(input)

    tape = wp.Tape()
    with tape:
        wp.launch(mul_half, dim=len(s), inputs=[input, output], device=device)

    ones = wp.array(np.ones(len(output)), dtype=wp.float16, device=device)

    tape.backward(grads={output: ones})
    
    assert_np_equal(input.grad.numpy(), np.ones(len(s))*2.0)


def register(parent):

    class TestFp16(parent):
        pass
    
    devices = []
    if wp.is_cpu_available():
        devices.append("cpu")
    if wp.is_cuda_available() and wp.context.runtime.core.cuda_get_device_arch() >= 70:
        devices.append("cuda")

    add_function_test(TestFp16, "test_fp16_conversion", test_fp16_conversion, devices=devices)
    add_function_test(TestFp16, "test_fp16_grad", test_fp16_grad, devices=devices)

    return TestFp16


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)


