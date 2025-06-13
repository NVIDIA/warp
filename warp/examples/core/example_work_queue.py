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
# Example Work Queue
#
# Shows how to use a work queue to synchronize access to a shared resource.
#
###########################################################################

import warp as wp
from warp.tests.unittest_utils import *


@wp.func
def volatile_read(ptr: wp.array(dtype=wp.int32), index: int):
    value = wp.atomic_add(ptr, index, 0)
    return value


@wp.struct
class WorkQueue:
    buffer: wp.array(dtype=wp.int32)
    capacity: int
    head: wp.array(dtype=wp.int32)
    tail: wp.array(dtype=wp.int32)


@wp.func
def enqueue(queue: WorkQueue, item: int) -> bool:
    while True:
        # Read current head and tail atomically
        current_tail = volatile_read(queue.tail, 0)
        current_head = volatile_read(queue.head, 0)

        # Check if queue is full
        if (current_tail - current_head) >= queue.capacity:
            return False

        # Try to increment tail atomically
        index = current_tail % queue.capacity
        if wp.atomic_cas(queue.tail, 0, current_tail, current_tail + 1) == current_tail:
            queue.buffer[index] = item
            return True

        # Retry if another thread changed tail


@wp.func
def dequeue(queue: WorkQueue) -> tuple[bool, int]:
    while True:
        # Read current head and tail atomically
        current_head = volatile_read(queue.head, 0)
        current_tail = volatile_read(queue.tail, 0)

        # Check if queue is empty
        if current_head >= current_tail:
            return False, 0

        # Get item at current head
        index = current_head % queue.capacity
        item = queue.buffer[index]

        # Try to increment head atomically
        if wp.atomic_cas(queue.head, 0, current_head, current_head + 1) == current_head:
            return True, item

        # Retry if another thread changed head


@wp.kernel
def process_queue(queue: WorkQueue):
    counter = int(0)
    while True:
        success, item = dequeue(queue)
        if not success:
            break
        wp.printf("Processed item: %d\n", item)
        if item < 1000000:
            if not enqueue(queue, item + 1000000):
                wp.printf("Failed to enqueue item: %d\n", item + 1000000)
        counter = counter + 1


def test_work_queue(device):
    # Create a work queue with capacity 1024
    capacity = 8192
    head = wp.array([0], dtype=wp.int32, device=device)
    tail = wp.array([4096], dtype=wp.int32, device=device)
    buffer = wp.array(np.arange(4096, dtype=np.int32), dtype=wp.int32, device=device)

    queue = WorkQueue()
    queue.capacity = capacity
    queue.head = head
    queue.tail = tail
    queue.buffer = buffer

    # Launch processing kernel
    wp.launch(process_queue, dim=1024, inputs=[queue], device=device)

    wp.synchronize()


if __name__ == "__main__":
    wp.clear_kernel_cache()
    test_work_queue(device="cuda")
