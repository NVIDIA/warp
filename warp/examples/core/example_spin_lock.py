# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Spin Lock
#
# Shows how to use a spin lock to synchronize access to a shared resource.
#
###########################################################################

import warp as wp
from warp.tests.unittest_utils import *


@wp.func
def spinlock_acquire(lock: wp.array[int]):
    # Try to acquire the lock by setting it to 1 if it's 0
    while wp.atomic_cas(lock, 0, 0, 1) == 1:
        pass


@wp.func
def spinlock_release(lock: wp.array[int]):
    # Release the lock by setting it back to 0
    wp.atomic_exch(lock, 0, 0)


@wp.func
def volatile_read(ptr: wp.array[int], index: int):
    value = wp.atomic_exch(ptr, index, 0)
    wp.atomic_exch(ptr, index, value)
    return value


@wp.kernel
def test_spinlock_counter(counter: wp.array[int], atomic_counter: wp.array[int], lock: wp.array[int]):
    # Try to acquire the lock
    spinlock_acquire(lock)

    # Critical section - increment counter
    # counter[0] = counter[0] + 1 # This gives wrong results - counter should be marked as volatile

    # Work around since warp arrays cannot be marked as volatile
    value = volatile_read(counter, 0)
    counter[0] = value + 1

    # Release the lock
    spinlock_release(lock)

    # Increment atomic counter for comparison
    wp.atomic_add(atomic_counter, 0, 1)


def test_spinlock(device):
    # Create a lock array initialized to 0 (unlocked)
    lock = wp.array([0], dtype=int, device=device)

    # Create counter arrays initialized to 0
    counter = wp.array([0], dtype=int, device=device)
    atomic_counter = wp.array([0], dtype=int, device=device)

    # Number of threads to test with
    n = 1024

    # Launch the test kernel
    wp.launch(test_spinlock_counter, dim=n, inputs=[counter, atomic_counter, lock], device=device)

    # Verify results
    atomic_counter_value = atomic_counter.numpy()[0]
    counter_value = counter.numpy()[0]
    lock_value = lock.numpy()[0]
    if atomic_counter_value != n:
        raise RuntimeError(f"Atomic counter must equal the thread count {n}, got {atomic_counter_value}")
    if counter_value != n:
        raise RuntimeError(f"Spin-lock counter must equal the thread count {n}, got {counter_value}")
    if lock_value != 0:
        raise RuntimeError(f"Lock must be released with value 0, got {lock_value}")

    print(f"Final counter value: {counter_value}")
    print(f"Final atomic counter value: {atomic_counter_value}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    test_spinlock(device="cuda")
