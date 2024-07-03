# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


# construct kernel + test function for atomic ops on each vec/matrix type
def make_atomic_test(type):
    def test_atomic_kernel(
        out_add: wp.array(dtype=type),
        out_min: wp.array(dtype=type),
        out_max: wp.array(dtype=type),
        val: wp.array(dtype=type),
    ):
        tid = wp.tid()

        wp.atomic_add(out_add, 0, val[tid])
        wp.atomic_min(out_min, 0, val[tid])
        wp.atomic_max(out_max, 0, val[tid])

    # register a custom kernel (no decorator) function
    # this lets us register the same function definition
    # against multiple symbols, with different arg types
    kernel = wp.Kernel(func=test_atomic_kernel, key=f"test_atomic_{type.__name__}_kernel")

    def test_atomic(test, device):
        n = 1024

        rng = np.random.default_rng(42)

        if type == wp.int32:
            base = (rng.random(size=1, dtype=np.float32) * 100.0).astype(np.int32)
            val = (rng.random(size=n, dtype=np.float32) * 100.0).astype(np.int32)

        elif type == wp.float32:
            base = rng.random(size=1, dtype=np.float32)
            val = rng.random(size=n, dtype=np.float32)

        else:
            base = rng.random(size=(1, *type._shape_), dtype=float)
            val = rng.random(size=(n, *type._shape_), dtype=float)

        add_array = wp.array(base, dtype=type, device=device, requires_grad=True)
        min_array = wp.array(base, dtype=type, device=device, requires_grad=True)
        max_array = wp.array(base, dtype=type, device=device, requires_grad=True)
        add_array.zero_()
        min_array.fill_(10000)
        max_array.fill_(-10000)

        val_array = wp.array(val, dtype=type, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel, n, inputs=[add_array, min_array, max_array, val_array], device=device)

        assert_np_equal(add_array.numpy(), np.sum(val, axis=0), tol=1.0e-2)
        assert_np_equal(min_array.numpy(), np.min(val, axis=0), tol=1.0e-2)
        assert_np_equal(max_array.numpy(), np.max(val, axis=0), tol=1.0e-2)

        if type != wp.int32:
            add_array.grad.fill_(1)
            tape.backward()
            assert_np_equal(val_array.grad.numpy(), np.ones_like(val))
            tape.zero()

            min_array.grad.fill_(1)
            tape.backward()
            min_grad_array = np.zeros_like(val)
            argmin = val.argmin(axis=0)
            if val.ndim == 1:
                min_grad_array[argmin] = 1
            elif val.ndim == 2:
                for i in range(val.shape[1]):
                    min_grad_array[argmin[i], i] = 1
            elif val.ndim == 3:
                for i in range(val.shape[1]):
                    for j in range(val.shape[2]):
                        min_grad_array[argmin[i, j], i, j] = 1
            assert_np_equal(val_array.grad.numpy(), min_grad_array)
            tape.zero()

            max_array.grad.fill_(1)
            tape.backward()
            max_grad_array = np.zeros_like(val)
            argmax = val.argmax(axis=0)
            if val.ndim == 1:
                max_grad_array[argmax] = 1
            elif val.ndim == 2:
                for i in range(val.shape[1]):
                    max_grad_array[argmax[i], i] = 1
            elif val.ndim == 3:
                for i in range(val.shape[1]):
                    for j in range(val.shape[2]):
                        max_grad_array[argmax[i, j], i, j] = 1
            assert_np_equal(val_array.grad.numpy(), max_grad_array)

    return test_atomic


# generate test functions for atomic types
test_atomic_int = make_atomic_test(wp.int32)
test_atomic_float = make_atomic_test(wp.float32)
test_atomic_vec2 = make_atomic_test(wp.vec2)
test_atomic_vec3 = make_atomic_test(wp.vec3)
test_atomic_vec4 = make_atomic_test(wp.vec4)
test_atomic_mat22 = make_atomic_test(wp.mat22)
test_atomic_mat33 = make_atomic_test(wp.mat33)
test_atomic_mat44 = make_atomic_test(wp.mat44)


devices = get_test_devices()


class TestAtomic(unittest.TestCase):
    pass


add_function_test(TestAtomic, "test_atomic_int", test_atomic_int, devices=devices)
add_function_test(TestAtomic, "test_atomic_float", test_atomic_float, devices=devices)
add_function_test(TestAtomic, "test_atomic_vec2", test_atomic_vec2, devices=devices)
add_function_test(TestAtomic, "test_atomic_vec3", test_atomic_vec3, devices=devices)
add_function_test(TestAtomic, "test_atomic_vec4", test_atomic_vec4, devices=devices)
add_function_test(TestAtomic, "test_atomic_mat22", test_atomic_mat22, devices=devices)
add_function_test(TestAtomic, "test_atomic_mat33", test_atomic_mat33, devices=devices)
add_function_test(TestAtomic, "test_atomic_mat44", test_atomic_mat44, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
