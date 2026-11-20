# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
    warp_type = wp.dtype_from_numpy(np.dtype(dtype))

    @wp.func
    def spinlock_acquire_1d(lock: wp.array[warp_type]):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, warp_type(0), warp_type(1)) == 1:
            pass

    @wp.func
    def spinlock_release_1d(lock: wp.array[warp_type]):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, warp_type(0))

    @wp.func
    def volatile_read_1d(ptr: wp.array[warp_type], index: int):
        value = wp.atomic_exch(ptr, index, warp_type(0))
        wp.atomic_exch(ptr, index, value)
        return value

    def test_spinlock_counter_1d(counter: wp.array[warp_type], lock: wp.array[warp_type]):
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

    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.array([0], dtype=warp_type, device=device)
    wp.launch(kernel, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=dtype)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


def test_atomic_cas_2d(test, device, dtype, register_kernels=False):
    warp_type = wp.dtype_from_numpy(np.dtype(dtype))

    @wp.func
    def spinlock_acquire_2d(lock: wp.array2d[warp_type]):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, warp_type(0), warp_type(1)) == 1:
            pass

    @wp.func
    def spinlock_release_2d(lock: wp.array2d[warp_type]):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, warp_type(0))

    @wp.func
    def volatile_read_2d(ptr: wp.array[warp_type], index: int):
        value = wp.atomic_exch(ptr, index, warp_type(0))
        wp.atomic_exch(ptr, index, value)
        return value

    def test_spinlock_counter_2d(counter: wp.array[warp_type], lock: wp.array2d[warp_type]):
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

    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.zeros(shape=(1, 1), dtype=warp_type, device=device)
    wp.launch(kernel, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=dtype)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


def test_atomic_cas_3d(test, device, dtype, register_kernels=False):
    warp_type = wp.dtype_from_numpy(np.dtype(dtype))

    @wp.func
    def spinlock_acquire_3d(lock: wp.array3d[warp_type]):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, 0, warp_type(0), warp_type(1)) == 1:
            pass

    @wp.func
    def spinlock_release_3d(lock: wp.array3d[warp_type]):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, 0, warp_type(0))

    @wp.func
    def volatile_read_3d(ptr: wp.array[warp_type], index: int):
        value = wp.atomic_exch(ptr, index, warp_type(0))
        wp.atomic_exch(ptr, index, value)
        return value

    def test_spinlock_counter_3d(counter: wp.array[warp_type], lock: wp.array3d[warp_type]):
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

    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.zeros(shape=(1, 1, 1), dtype=warp_type, device=device)
    wp.launch(kernel, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=dtype)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


def create_spinlock_test_4d(dtype):
    @wp.func
    def spinlock_acquire(lock: wp.array4d[dtype]):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, 0, 0, dtype(0), dtype(1)) == 1:
            pass

    @wp.func
    def spinlock_release(lock: wp.array4d[dtype]):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, 0, 0, dtype(0))

    @wp.func
    def volatile_read(ptr: wp.array[dtype], index: int):
        value = wp.atomic_exch(ptr, index, dtype(0))
        wp.atomic_exch(ptr, index, value)
        return value

    @wp.kernel
    def test_spinlock_counter(counter: wp.array[dtype], lock: wp.array4d[dtype]):
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
    warp_type = wp.dtype_from_numpy(np.dtype(dtype))

    @wp.func
    def spinlock_acquire_4d(lock: wp.array4d[warp_type]):
        # Try to acquire the lock by setting it to 1 if it's 0
        while wp.atomic_cas(lock, 0, 0, 0, 0, warp_type(0), warp_type(1)) == 1:
            pass

    @wp.func
    def spinlock_release_4d(lock: wp.array4d[warp_type]):
        # Release the lock by setting it back to 0
        wp.atomic_exch(lock, 0, 0, 0, 0, warp_type(0))

    @wp.func
    def volatile_read_4d(ptr: wp.array[warp_type], index: int):
        value = wp.atomic_exch(ptr, index, warp_type(0))
        wp.atomic_exch(ptr, index, value)
        return value

    def test_spinlock_counter_4d(counter: wp.array[warp_type], lock: wp.array4d[warp_type]):
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

    n = 100
    counter = wp.array([0], dtype=warp_type, device=device)
    lock = wp.zeros(shape=(1, 1, 1, 1), dtype=warp_type, device=device)
    wp.launch(kernel, dim=n, inputs=[counter, lock], device=device)

    # Verify counter reached n
    counter_np = counter.numpy()
    expected = np.array([n], dtype=dtype)

    if not np.array_equal(counter_np, expected):
        print(f"Counter mismatch: expected {expected}, got {counter_np}")

    assert_np_equal(counter_np, expected)


def test_atomic_cas_float_bitwise(test, device, dtype, register_kernels=False):
    warp_type = wp.dtype_from_numpy(np.dtype(dtype))
    uint_type = np.uint32 if dtype == np.float32 else np.uint64

    def cas_one(
        arr: wp.array[warp_type],
        compare: wp.array[warp_type],
        value: wp.array[warp_type],
        old: wp.array[warp_type],
    ):
        old[0] = wp.atomic_cas(arr, 0, compare[0], value[0])

    kernel = getkernel(cas_one, suffix=dtype.__name__)

    if register_kernels:
        return

    def bits(x):
        return np.array([x], dtype=dtype).view(uint_type)[0]

    pos_zero = bits(0.0)
    neg_zero = bits(-0.0)
    one = bits(1.0)
    two = bits(2.0)
    three = bits(3.0)

    # Quiet NaNs with different payloads.
    if dtype == np.float32:
        nan_a = uint_type(0x7FC00001)
        nan_b = uint_type(0x7FC00002)
    else:
        nan_a = uint_type(0x7FF8000000000001)
        nan_b = uint_type(0x7FF8000000000002)

    def array_from_bits(b):
        return wp.array(np.array([b], dtype=uint_type).view(dtype), dtype=warp_type, device=device)

    # Swap in 2.0 if the stored bits equal the compare bits, then return the
    # stored and returned values as raw bits.
    def cas(stored_bits, compare_bits):
        arr = array_from_bits(stored_bits)
        old = wp.empty(1, dtype=warp_type, device=device)
        wp.launch(kernel, dim=1, inputs=[arr, array_from_bits(compare_bits), array_from_bits(two), old], device=device)
        return arr.numpy().view(uint_type)[0], old.numpy().view(uint_type)[0]

    stored, old = cas(pos_zero, neg_zero)
    test.assertEqual(stored, pos_zero)
    test.assertEqual(old, pos_zero)

    stored, old = cas(neg_zero, pos_zero)
    test.assertEqual(stored, neg_zero)
    test.assertEqual(old, neg_zero)

    stored, old = cas(neg_zero, neg_zero)
    test.assertEqual(stored, two)
    test.assertEqual(old, neg_zero)

    stored, old = cas(nan_a, nan_a)
    test.assertEqual(stored, two)
    test.assertEqual(old, nan_a)

    stored, old = cas(nan_a, nan_b)
    test.assertEqual(stored, nan_a)
    test.assertEqual(old, nan_a)

    stored, old = cas(one, one)
    test.assertEqual(stored, two)
    test.assertEqual(old, one)

    stored, old = cas(one, three)
    test.assertEqual(stored, one)
    test.assertEqual(old, one)


# Pre-Volta CUDA does not provide independent thread scheduling, so a
# spinlock contended by lanes in the same warp can deadlock. CPU launches
# remain covered because Warp executes these kernels serially by default.
devices = [d for d in get_test_devices() if not d.is_cuda or d.arch >= 70]


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

for dtype in (np.float32, np.float64):
    add_function_test_register_kernel(
        TestAtomicCAS,
        f"test_cas_float_bitwise_{dtype.__name__}",
        test_atomic_cas_float_bitwise,
        devices=get_test_devices(),
        dtype=dtype,
    )

if __name__ == "__main__":
    unittest.main(verbosity=2)
