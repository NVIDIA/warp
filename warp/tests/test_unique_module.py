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

import numpy as np

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


def test_unique_module_deferred_static_expressions(test, device):
    """Test that unique modules correctly hash deferred wp.static() expressions.

    Some wp.static() expressions cannot be evaluated at kernel declaration time
    and must be deferred until codegen (e.g., when they reference a loop variable).
    This test verifies that unique modules properly resolve and include these
    deferred expressions in the hash, ensuring kernels with different static
    values get different hashes.
    """

    def make_kernel(values):
        @wp.kernel(module="unique", enable_backward=False)
        def kernel_with_deferred_static(result: wp.array(dtype=int)):
            tid = wp.tid()
            if tid == 0:
                for i in range(wp.static(len(values))):
                    # wp.static(values[i]) references loop var 'i', so it's deferred
                    result[i] = wp.static(values[i])

        return kernel_with_deferred_static

    # Create two kernels with different values but same length
    kernel1 = make_kernel([100, 200])
    kernel2 = make_kernel([999, 888])

    # They should be different kernel objects (different hashes)
    test.assertIsNot(kernel1, kernel2, "Kernels with different static values should be different objects")
    test.assertNotEqual(
        kernel1.module.name,
        kernel2.module.name,
        "Kernels with different static values should have different module names",
    )

    # Verify they produce correct results
    result1 = wp.zeros(2, dtype=int, device=device)
    wp.launch(kernel1, dim=1, inputs=[], outputs=[result1], device=device)
    assert_np_equal(result1.numpy(), np.array([100, 200]))

    result2 = wp.zeros(2, dtype=int, device=device)
    wp.launch(kernel2, dim=1, inputs=[], outputs=[result2], device=device)
    assert_np_equal(result2.numpy(), np.array([999, 888]))

    # Test with same last element but different first element — the hash must
    # capture ALL loop iterations, not just the last one (GH-1211)
    kernel3 = make_kernel([100, 999])
    kernel4 = make_kernel([200, 999])
    test.assertIsNot(kernel3, kernel4, "Kernels differing only in non-last elements should be different")
    test.assertNotEqual(kernel3.module.name, kernel4.module.name)

    result3 = wp.zeros(2, dtype=int, device=device)
    wp.launch(kernel3, dim=1, inputs=[], outputs=[result3], device=device)
    assert_np_equal(result3.numpy(), np.array([100, 999]))

    result4 = wp.zeros(2, dtype=int, device=device)
    wp.launch(kernel4, dim=1, inputs=[], outputs=[result4], device=device)
    assert_np_equal(result4.numpy(), np.array([200, 999]))

    # Test with different length to ensure distinct hash from 2-element kernels
    kernel5 = make_kernel([1, 2, 3])
    test.assertIsNot(kernel5, kernel1, "Kernels with different lengths should be different objects")
    test.assertNotEqual(kernel5.module.name, kernel1.module.name)

    result5 = wp.zeros(3, dtype=int, device=device)
    wp.launch(kernel5, dim=1, inputs=[], outputs=[result5], device=device)
    assert_np_equal(result5.numpy(), np.array([1, 2, 3]))

    # Test that identical values reuse the same kernel (hash stability)
    kernel1_dup = make_kernel([100, 200])
    test.assertIs(kernel1_dup, kernel1, "Identical values should reuse the same kernel object")


devices = get_test_devices()

add_function_test(
    TestUniqueModule, "test_unique_module_kernel_object_reuse", test_unique_module_kernel_object_reuse, devices=devices
)
add_function_test(
    TestUniqueModule,
    "test_unique_module_deferred_static_expressions",
    test_unique_module_deferred_static_expressions,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
