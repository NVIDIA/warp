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

###########################################################################
# Example Spin Lock
#
# Shows how to use a spin lock to synchronize access to a shared resource.
#
###########################################################################

import warp as wp
from warp.tests.unittest_utils import *


@wp.func
def spinlock_acquire(lock: wp.array(dtype=wp.int32)):
    # Try to acquire the lock by setting it to 1 if it's 0
    while wp.atomic_cas(lock, 0, 0, 1) == 1:
        pass


@wp.func
def spinlock_release(lock: wp.array(dtype=wp.int32)):
    # Release the lock by setting it back to 0
    wp.atomic_exch(lock, 0, 0)


@wp.func
def volatile_read(ptr: wp.array(dtype=wp.int32), index: int):
    value = wp.atomic_exch(ptr, index, 0)
    wp.atomic_exch(ptr, index, value)
    return value


@wp.kernel
def test_spinlock_counter(
    counter: wp.array(dtype=wp.int32), atomic_counter: wp.array(dtype=wp.int32), lock: wp.array(dtype=wp.int32)
):
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
    lock = wp.array([0], dtype=wp.int32, device=device)

    # Create counter arrays initialized to 0
    counter = wp.array([0], dtype=wp.int32, device=device)
    atomic_counter = wp.array([0], dtype=wp.int32, device=device)

    # Number of threads to test with
    n = 1024

    # Launch the test kernel
    wp.launch(test_spinlock_counter, dim=n, inputs=[counter, atomic_counter, lock], device=device)

    # Verify results
    assert atomic_counter.numpy()[0] == n, f"Atomic counter should be {n}, got {atomic_counter.numpy()[0]}"
    assert counter.numpy()[0] == n, f"Counter should be {n}, got {counter.numpy()[0]}"
    assert lock.numpy()[0] == 0, "Lock was not properly released"

    print(f"Final counter value: {counter.numpy()[0]}")
    print(f"Final atomic counter value: {atomic_counter.numpy()[0]}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    test_spinlock(device="cuda")
