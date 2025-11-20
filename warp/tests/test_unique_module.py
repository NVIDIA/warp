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

"""
Tests for unique module kernel behavior.

Unique modules (module="unique") create a separate, isolated module for each kernel.
This is primarily useful for runtime kernel additions, as it avoids recompiling existing
modules. These tests verify correct behavior of kernel and module object reuse when the
same unique kernel is defined multiple times.
"""

import unittest
from typing import Any

import warp as wp
from warp.tests.unittest_utils import *


def test_unique_module_kernel_object_reuse(test, device):
    """Test that identical unique kernel definitions return the same kernel object.

    When the same kernel is defined twice with module="unique", the second
    definition should return the exact same kernel object as the first (not just
    an equivalent one). This ensures consistent behavior and avoids potential
    issues with hash mismatches.
    """

    # First definition
    @wp.kernel(module="unique")
    def my_unique_kernel(x: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        x[tid] = x[tid] * 2.0

    first_kernel = my_unique_kernel
    first_module = my_unique_kernel.module

    # Second definition (same code, should reuse)
    @wp.kernel(module="unique")
    def my_unique_kernel(x: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        x[tid] = x[tid] * 2.0

    second_kernel = my_unique_kernel
    second_module = my_unique_kernel.module

    # Verify same object identity
    test.assertIs(first_kernel, second_kernel, "Kernel objects should be identical (same object in memory)")
    test.assertIs(first_module, second_module, "Modules should be identical when reusing unique module")

    # Verify it actually works
    data = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
    wp.launch(my_unique_kernel, dim=3, inputs=[data], device=device)
    assert_np_equal(data.numpy(), [2.0, 4.0, 6.0])


class TestUniqueModule(unittest.TestCase):
    def test_unique_module_generic_kernel_object_reuse(self):
        """Test that generic unique kernels reuse the same kernel object across redefinitions.

        When a generic kernel with module="unique" is defined multiple times in a loop,
        each redefinition should return the same kernel object (the registry containing
        all overloads), ensuring consistent behavior.
        """
        kernel_objects = []
        module_objects = []

        for _ in range(3):
            # Define generic kernel - should reuse after first iteration
            @wp.kernel(module="unique")
            def generic_unique_kernel(x: wp.array(dtype=Any), scale: Any):
                tid = wp.tid()
                x[tid] = x[tid] * scale

            kernel_objects.append(generic_unique_kernel)
            module_objects.append(generic_unique_kernel.module)

        # Verify all definitions returned the same kernel and module objects
        for i in range(1, 3):
            self.assertIs(
                kernel_objects[i],
                kernel_objects[0],
                f"Kernel object at iteration {i} should be identical to first iteration",
            )
            self.assertIs(
                module_objects[i],
                module_objects[0],
                f"Module object at iteration {i} should be identical to first iteration",
            )

        # Verify the generic kernel is actually generic
        self.assertTrue(generic_unique_kernel.is_generic, "Kernel should be generic")

        # Launch with different types to create overloads
        data_f32 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        wp.launch(generic_unique_kernel, dim=3, inputs=[data_f32, wp.float32(2.0)], device="cpu")
        assert_np_equal(data_f32.numpy(), [2.0, 4.0, 6.0])

        data_f64 = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device="cpu")
        wp.launch(generic_unique_kernel, dim=3, inputs=[data_f64, wp.float64(3.0)], device="cpu")
        assert_np_equal(data_f64.numpy(), [3.0, 6.0, 9.0])

        # Verify overloads were added to the same kernel object
        self.assertEqual(len(generic_unique_kernel.overloads), 2, "Should have 2 overloads (float32 and float64)")

    def test_unique_module_generic_multiple_overloads(self):
        """Test that multiple overloads of a generic unique kernel work correctly.

        Define a generic kernel once with module="unique", then launch it with
        several different type combinations. Verify that:
        1. Overloads are stored in the same kernel object
        2. Module hash changes when new overloads are added
        3. Reusing existing overloads doesn't change the hash
        4. All overloads work correctly after recompilation
        """

        @wp.kernel(module="unique")
        def multi_type_kernel(x: wp.array(dtype=Any), y: wp.array(dtype=Any)):
            tid = wp.tid()
            y[tid] = x[tid] + x[tid]

        self.assertTrue(multi_type_kernel.is_generic, "Kernel should be generic")
        self.assertEqual(len(multi_type_kernel.overloads), 0, "Should start with no overloads")

        with wp.ScopedDevice("cpu"):
            # Launch with float32
            x_f32 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
            y_f32 = wp.zeros(3, dtype=wp.float32)
            wp.launch(multi_type_kernel, dim=3, inputs=[x_f32, y_f32])
            assert_np_equal(y_f32.numpy(), [2.0, 4.0, 6.0])
            self.assertEqual(len(multi_type_kernel.overloads), 1, "Should have 1 overload after float32 launch")
            hash_after_f32 = multi_type_kernel.module.get_module_hash()

            # Launch with float64 (new overload triggers hash change)
            x_f64 = wp.array([1.0, 2.0, 3.0], dtype=wp.float64)
            y_f64 = wp.zeros(3, dtype=wp.float64)
            wp.launch(multi_type_kernel, dim=3, inputs=[x_f64, y_f64])
            assert_np_equal(y_f64.numpy(), [2.0, 4.0, 6.0])
            self.assertEqual(len(multi_type_kernel.overloads), 2, "Should have 2 overloads after float64 launch")
            hash_after_f64 = multi_type_kernel.module.get_module_hash()
            self.assertNotEqual(hash_after_f32, hash_after_f64, "Module hash should change when new overload is added")

            # Launch with int32 (another new overload)
            x_i32 = wp.array([1, 2, 3], dtype=wp.int32)
            y_i32 = wp.zeros(3, dtype=wp.int32)
            wp.launch(multi_type_kernel, dim=3, inputs=[x_i32, y_i32])
            assert_np_equal(y_i32.numpy(), [2, 4, 6])
            self.assertEqual(len(multi_type_kernel.overloads), 3, "Should have 3 overloads after int32 launch")
            hash_after_i32 = multi_type_kernel.module.get_module_hash()
            self.assertNotEqual(hash_after_f64, hash_after_i32, "Module hash should change when new overload is added")

            # Re-launch with float32 (should reuse existing overload, no hash change)
            x_f32_2 = wp.array([10.0, 20.0, 30.0], dtype=wp.float32)
            y_f32_2 = wp.zeros(3, dtype=wp.float32)
            wp.launch(multi_type_kernel, dim=3, inputs=[x_f32_2, y_f32_2])
            assert_np_equal(y_f32_2.numpy(), [20.0, 40.0, 60.0])
            self.assertEqual(len(multi_type_kernel.overloads), 3, "Should still have 3 overloads (reused float32)")
            hash_after_reuse = multi_type_kernel.module.get_module_hash()
            self.assertEqual(
                hash_after_i32, hash_after_reuse, "Module hash should not change when reusing existing overload"
            )

            # Verify all overloads still work correctly after recompilation
            x_f64_2 = wp.array([5.0, 6.0, 7.0], dtype=wp.float64)
            y_f64_2 = wp.zeros(3, dtype=wp.float64)
            wp.launch(multi_type_kernel, dim=3, inputs=[x_f64_2, y_f64_2])
            assert_np_equal(y_f64_2.numpy(), [10.0, 12.0, 14.0])

            x_i32_2 = wp.array([5, 6, 7], dtype=wp.int32)
            y_i32_2 = wp.zeros(3, dtype=wp.int32)
            wp.launch(multi_type_kernel, dim=3, inputs=[x_i32_2, y_i32_2])
            assert_np_equal(y_i32_2.numpy(), [10, 12, 14])


devices = get_test_devices()

add_function_test(
    TestUniqueModule, "test_unique_module_kernel_object_reuse", test_unique_module_kernel_object_reuse, devices=devices
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
