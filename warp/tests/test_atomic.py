# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

import unittest

wp.init()

# construct kernel + test function for atomic ops on each vec/matrix type
def make_atomic_test(type):
   
    def test_atomic_kernel(out_add: wp.array(dtype=type),
                           out_min: wp.array(dtype=type),
                           out_max: wp.array(dtype=type),
                           val: wp.array(dtype=type)):

        tid = wp.tid()
        
        wp.atomic_add(out_add, 0, val[tid])
        wp.atomic_min(out_min, 0, val[tid])
        wp.atomic_max(out_max, 0, val[tid])


    # register a custom kernel (no decorator) function
    # this lets us register the same function definition
    # against multiple symbols, with different arg types
    module = wp.get_module(test_atomic_kernel.__module__)
    kernel = wp.Kernel(func=test_atomic_kernel, key=f"test_atomic_{type.__name__}_kernel", module=module)
        
    def test_atomic(test, device):

        n = 1024

        rng = np.random.default_rng(42)
        
        if type == wp.int32:
            base = (rng.random(size=1, dtype=np.float32)*100.0).astype(np.int32)
            val = (rng.random(size=n, dtype=np.float32)*100.0).astype(np.int32)
        
        elif type == wp.float32:
            base = rng.random(size=1, dtype=np.float32)
            val = rng.random(size=n, dtype=np.float32)
        
        else:
            base = rng.random(size=(1, *type._shape_), dtype=float)
            val = rng.random(size=(n, *type._shape_), dtype=float)
        
        add_array = wp.array(base, dtype=type, device=device)
        min_array = wp.array(base, dtype=type, device=device)
        max_array = wp.array(base, dtype=type, device=device)
        
        val_array = wp.array(val, dtype=type, device=device)

        wp.launch(kernel, n, inputs=[add_array, min_array, max_array, val_array], device=device)

        val = np.append(val, [base[0]], axis=0)

        assert_np_equal(add_array.numpy(), np.sum(val, axis=0), tol=1.e-2)
        assert_np_equal(min_array.numpy(), np.min(val, axis=0), tol=1.e-2)
        assert_np_equal(max_array.numpy(), np.max(val, axis=0), tol=1.e-2)

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


def register(parent):

    devices = wp.get_devices()

    class TestAtomic(parent):
        pass
    
    add_function_test(TestAtomic, "test_atomic_int", test_atomic_int, devices=devices)
    add_function_test(TestAtomic, "test_atomic_float", test_atomic_float, devices=devices)
    add_function_test(TestAtomic, "test_atomic_vec2", test_atomic_vec2, devices=devices)
    add_function_test(TestAtomic, "test_atomic_vec3", test_atomic_vec3, devices=devices)
    add_function_test(TestAtomic, "test_atomic_vec4", test_atomic_vec4, devices=devices)
    add_function_test(TestAtomic, "test_atomic_mat22", test_atomic_mat22, devices=devices)
    add_function_test(TestAtomic, "test_atomic_mat33", test_atomic_mat33, devices=devices)
    add_function_test(TestAtomic, "test_atomic_mat44", test_atomic_mat44, devices=devices)

    return TestAtomic

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)