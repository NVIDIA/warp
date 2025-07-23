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
from warp.tests.unittest_utils import *

kernel_cache = {}


def getkernel(func, suffix=""):
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key)
    return kernel_cache[key]


def test_atomic_cas(test, device, dtype, register_kernels=False):
    warp_type = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.array([0], dtype=warp_type, device=device)

    @wp.func
    def spinlock_acquire_1d(lock: wp.array(dtype=warp_type)):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, warp_type(0), warp_type(1)) == 1:
            pass

    @wp.func
    def spinlock_release_1d(lock: wp.array(dtype=warp_type)):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, warp_type(0))

    @wp.func
    def volatile_read_1d(ptr: wp.array(dtype=warp_type), index: int):
        value = wp.atomic_exch(ptr, index, warp_type(0))
        wp.atomic_exch(ptr, index, value)
        return value

    def test_spinlock_counter_1d(counter: wp.array(dtype=warp_type), lock: wp.array(dtype=warp_type)):
        # Try to acquire the lock
        spinlock_acquire_1d(lock)

        # Critical section - increment counter
        # counter[0] = counter[0] + 1 # This gives wrong results - counter should be marked as volatile

        # Work around since warp arrays cannot be marked as volatile
        value = volatile_read_1d(counter, 0)
        counter[0] = value + warp_type(1)

        # Release the lock
        spinlock_release_1d(lock)

    kernel = getkernel(test_spinlock_counter_1d, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=dtype)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


def test_atomic_cas_2d(test, device, dtype, register_kernels=False):
    warp_type = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.zeros(shape=(1, 1), dtype=warp_type, device=device)

    @wp.func
    def spinlock_acquire_2d(lock: wp.array2d(dtype=warp_type)):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, warp_type(0), warp_type(1)) == 1:
            pass

    @wp.func
    def spinlock_release_2d(lock: wp.array2d(dtype=warp_type)):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, warp_type(0))

    @wp.func
    def volatile_read_2d(ptr: wp.array(dtype=warp_type), index: int):
        value = wp.atomic_exch(ptr, index, warp_type(0))
        wp.atomic_exch(ptr, index, value)
        return value

    def test_spinlock_counter_2d(counter: wp.array(dtype=warp_type), lock: wp.array2d(dtype=warp_type)):
        # Try to acquire the lock
        spinlock_acquire_2d(lock)

        # Critical section - increment counter
        # counter[0] = counter[0] + 1 # This gives wrong results - counter should be marked as volatile

        # Work around since warp arrays cannot be marked as volatile
        value = volatile_read_2d(counter, 0)
        counter[0] = value + warp_type(1)

        # Release the lock
        spinlock_release_2d(lock)

    kernel = getkernel(test_spinlock_counter_2d, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=dtype)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


def test_atomic_cas_3d(test, device, dtype, register_kernels=False):
    warp_type = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.zeros(shape=(1, 1, 1), dtype=warp_type, device=device)

    @wp.func
    def spinlock_acquire_3d(lock: wp.array3d(dtype=warp_type)):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, 0, warp_type(0), warp_type(1)) == 1:
            pass

    @wp.func
    def spinlock_release_3d(lock: wp.array3d(dtype=warp_type)):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, 0, warp_type(0))

    @wp.func
    def volatile_read_3d(ptr: wp.array(dtype=warp_type), index: int):
        value = wp.atomic_exch(ptr, index, warp_type(0))
        wp.atomic_exch(ptr, index, value)
        return value

    def test_spinlock_counter_3d(counter: wp.array(dtype=warp_type), lock: wp.array3d(dtype=warp_type)):
        # Try to acquire the lock
        spinlock_acquire_3d(lock)

        # Critical section - increment counter
        # counter[0] = counter[0] + 1 # This gives wrong results - counter should be marked as volatile

        # Work around since warp arrays cannot be marked as volatile
        value = volatile_read_3d(counter, 0)
        counter[0] = value + warp_type(1)

        # Release the lock
        spinlock_release_3d(lock)

    kernel = getkernel(test_spinlock_counter_3d, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=dtype)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


def create_spinlock_test_4d(dtype):
    @wp.func
    def spinlock_acquire(lock: wp.array(dtype=dtype, ndim=4)):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, 0, 0, dtype(0), dtype(1)) == 1:
            pass

    @wp.func
    def spinlock_release(lock: wp.array(dtype=dtype, ndim=4)):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, 0, 0, dtype(0))

    @wp.func
    def volatile_read(ptr: wp.array(dtype=dtype), index: int):
        value = wp.atomic_exch(ptr, index, dtype(0))
        wp.atomic_exch(ptr, index, value)
        return value

    @wp.kernel
    def test_spinlock_counter(counter: wp.array(dtype=dtype), lock: wp.array(dtype=dtype, ndim=4)):
        # Try to acquire the lock
        spinlock_acquire(lock)

        # Critical section - increment counter
        # counter[0] = counter[0] + 1 # This gives wrong results - counter should be marked as volatile

        # Work around since warp arrays cannot be marked as volatile
        value = volatile_read(counter, 0)
        counter[0] = value + dtype(1)

        # Release the lock
        spinlock_release(lock)

    return test_spinlock_counter


def test_atomic_cas_4d(test, device, dtype, register_kernels=False):
    warp_type = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.zeros(shape=(1, 1, 1, 1), dtype=warp_type, device=device)

    @wp.func
    def spinlock_acquire_4d(lock: wp.array4d(dtype=warp_type)):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, 0, 0, warp_type(0), warp_type(1)) == 1:
            pass

    @wp.func
    def spinlock_release_4d(lock: wp.array4d(dtype=warp_type)):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, 0, 0, warp_type(0))

    @wp.func
    def volatile_read_4d(ptr: wp.array(dtype=warp_type), index: int):
        value = wp.atomic_exch(ptr, index, warp_type(0))
        wp.atomic_exch(ptr, index, value)
        return value

    def test_spinlock_counter_4d(counter: wp.array(dtype=warp_type), lock: wp.array4d(dtype=warp_type)):
        # Try to acquire the lock
        spinlock_acquire_4d(lock)

        # Critical section - increment counter
        # counter[0] = counter[0] + 1 # This gives wrong results - counter should be marked as volatile

        # Work around since warp arrays cannot be marked as volatile
        value = volatile_read_4d(counter, 0)
        counter[0] = value + warp_type(1)

        # Release the lock
        spinlock_release_4d(lock)

    kernel = getkernel(test_spinlock_counter_4d, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=dtype)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


devices = get_test_devices()


class TestAtomicCAS(unittest.TestCase):
    pass


# Test all supported types
np_test_types = (np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)

for dtype in np_test_types:
    type_name = dtype.__name__
    add_function_test_register_kernel(
        TestAtomicCAS, f"test_cas_{type_name}", test_atomic_cas, devices=devices, dtype=dtype
    )
    # Add 2D test for each type
    add_function_test_register_kernel(
        TestAtomicCAS, f"test_cas_2d_{type_name}", test_atomic_cas_2d, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestAtomicCAS, f"test_cas_3d_{type_name}", test_atomic_cas_3d, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestAtomicCAS, f"test_cas_4d_{type_name}", test_atomic_cas_4d, devices=devices, dtype=dtype
    )

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
