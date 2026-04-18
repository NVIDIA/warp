# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demonstrates how to write and use a custom GPU memory allocator with Warp.

This example implements a simple logging allocator that wraps the built-in
allocator and prints allocation/deallocation events. The same pattern can be
used to integrate any external memory manager.
"""

import warp as wp


class LoggingAllocator:
    """A custom allocator that logs allocation and deallocation events.

    Wraps a device's built-in allocator and prints a message for each
    ``allocate`` and ``deallocate`` call.
    """

    def __init__(self, device):
        # Use the non-mempool default allocator; use device.get_allocator()
        # to wrap whichever allocator the device is currently configured with.
        self._inner = device.default_allocator
        self.total_allocated = 0
        self.total_deallocated = 0

    def allocate(self, size_in_bytes: int) -> int:
        ptr = self._inner.allocate(size_in_bytes)
        self.total_allocated += size_in_bytes
        print(f"  [alloc]   {size_in_bytes:>10} bytes -> ptr=0x{ptr:016X}  (total: {self.total_allocated})")
        return ptr

    def deallocate(self, ptr: int, size_in_bytes: int) -> None:
        self._inner.deallocate(ptr, size_in_bytes)
        self.total_deallocated += size_in_bytes
        print(f"  [dealloc] {size_in_bytes:>10} bytes    ptr=0x{ptr:016X}  (freed: {self.total_deallocated})")


class Example:
    def __init__(self):
        self.device = wp.get_device()
        self.alloc = LoggingAllocator(self.device)

        print("Setting custom allocator...")
        wp.set_device_allocator(self.device, self.alloc)

        print("\nCreating arrays:")
        self.a = wp.zeros(1000, dtype=wp.float32)
        self.b = wp.ones(500, dtype=wp.float32)

        print("\nUsing ScopedAllocator to temporarily restore the default:")
        with wp.ScopedAllocator(self.device, None):
            c = wp.zeros(100, dtype=wp.float32)
            print(f"  (allocated {c.size} elements with built-in allocator, no log)")

        print("\nDeleting arrays:")
        del self.a
        del self.b

        print(f"\nTotal allocated:   {self.alloc.total_allocated} bytes")
        print(f"Total deallocated: {self.alloc.total_deallocated} bytes")

        wp.set_device_allocator(self.device, None)
        print("\nRestored default allocator.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        Example()
