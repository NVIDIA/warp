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


def create_spinlock_test(dtype):
    @wp.func
    def spinlock_acquire(lock: wp.array(dtype=dtype)):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, dtype(0), dtype(1)) == 1:
            pass

    @wp.func
    def spinlock_release(lock: wp.array(dtype=dtype)):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, dtype(0))

    @wp.func
    def volatile_read(ptr: wp.array(dtype=dtype), index: int):
        value = wp.atomic_exch(ptr, index, dtype(0))
        wp.atomic_exch(ptr, index, value)
        return value

    @wp.kernel
    def test_spinlock_counter(counter: wp.array(dtype=dtype), lock: wp.array(dtype=dtype)):
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


def test_atomic_cas(test, device, warp_type, numpy_type):
    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.array([0], dtype=warp_type, device=device)

    test_spinlock_counter = create_spinlock_test(warp_type)
    wp.launch(test_spinlock_counter, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=numpy_type)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


def create_spinlock_test_2d(dtype):
    @wp.func
    def spinlock_acquire(lock: wp.array(dtype=dtype, ndim=2)):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, dtype(0), dtype(1)) == 1:
            pass

    @wp.func
    def spinlock_release(lock: wp.array(dtype=dtype, ndim=2)):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, dtype(0))

    @wp.func
    def volatile_read(ptr: wp.array(dtype=dtype), index: int):
        value = wp.atomic_exch(ptr, index, dtype(0))
        wp.atomic_exch(ptr, index, value)
        return value

    @wp.kernel
    def test_spinlock_counter(counter: wp.array(dtype=dtype), lock: wp.array(dtype=dtype, ndim=2)):
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


def test_atomic_cas_2d(test, device, warp_type, numpy_type):
    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.zeros(shape=(1, 1), dtype=warp_type, device=device)

    test_spinlock_counter = create_spinlock_test_2d(warp_type)
    wp.launch(test_spinlock_counter, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=numpy_type)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


def create_spinlock_test_3d(dtype):
    @wp.func
    def spinlock_acquire(lock: wp.array(dtype=dtype, ndim=3)):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, 0, dtype(0), dtype(1)) == 1:
            pass

    @wp.func
    def spinlock_release(lock: wp.array(dtype=dtype, ndim=3)):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, 0, dtype(0))

    @wp.func
    def volatile_read(ptr: wp.array(dtype=dtype), index: int):
        value = wp.atomic_exch(ptr, index, dtype(0))
        wp.atomic_exch(ptr, index, value)
        return value

    @wp.kernel
    def test_spinlock_counter(counter: wp.array(dtype=dtype), lock: wp.array(dtype=dtype, ndim=3)):
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


def test_atomic_cas_3d(test, device, warp_type, numpy_type):
    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.zeros(shape=(1, 1, 1), dtype=warp_type, device=device)

    test_spinlock_counter = create_spinlock_test_3d(warp_type)
    wp.launch(test_spinlock_counter, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=numpy_type)

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


def test_atomic_cas_4d(test, device, warp_type, numpy_type):
    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.zeros(shape=(1, 1, 1, 1), dtype=warp_type, device=device)

    test_spinlock_counter = create_spinlock_test_4d(warp_type)
    wp.launch(test_spinlock_counter, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=numpy_type)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


devices = get_test_devices()


class TestAtomicCAS(unittest.TestCase):
    pass


# Test all supported types
test_types = [
    (wp.int32, np.int32),
    (wp.uint32, np.uint32),
    (wp.int64, np.int64),
    (wp.uint64, np.uint64),
    (wp.float32, np.float32),
    (wp.float64, np.float64),
]

for warp_type, numpy_type in test_types:
    type_name = warp_type.__name__
    add_function_test(
        TestAtomicCAS,
        f"test_cas_{type_name}",
        test_atomic_cas,
        devices=devices,
        warp_type=warp_type,
        numpy_type=numpy_type,
    )

    # Add 2D test for each type
    add_function_test(
        TestAtomicCAS,
        f"test_cas_2d_{type_name}",
        test_atomic_cas_2d,
        devices=devices,
        warp_type=warp_type,
        numpy_type=numpy_type,
    )

    # Add 3D test for each type
    add_function_test(
        TestAtomicCAS,
        f"test_cas_3d_{type_name}",
        test_atomic_cas_3d,
        devices=devices,
        warp_type=warp_type,
        numpy_type=numpy_type,
    )

    # Add 4D test for each type
    add_function_test(
        TestAtomicCAS,
        f"test_cas_4d_{type_name}",
        test_atomic_cas_4d,
        devices=devices,
        warp_type=warp_type,
        numpy_type=numpy_type,
    )

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
