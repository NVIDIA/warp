# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test executable variants without changing unique-kernel binding behavior."""

import unittest
from unittest import mock

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


def test_deferred_helper_variant_symbols(test, device):
    """Keep cached executable symbols usable after compiling another block size."""
    static_values = (11, 22)

    @wp.func
    def write_deferred_values(values_out: wp.array[int]):
        for index in range(wp.static(len(static_values))):
            values_out[index] = wp.static(static_values[index])

    @wp.kernel(module="unique", enable_backward=False, name=f"deferred_values_{device.alias.replace(':', '_')}")
    def deferred_values_kernel(values_out: wp.array[int]):
        write_deferred_values(values_out)

    values_out = wp.zeros(2, dtype=int, device=device)
    default_exec = deferred_values_kernel.module.load(device)
    if device.is_cuda:
        occupancy = wp.get_suggested_block_size(deferred_values_kernel, device)
        properties = wp.get_cuda_kernel_properties(deferred_values_kernel, device=device)

    for block_dim in (64, 256, 64, 256):
        with test.subTest(block_dim=block_dim):
            wp.launch(deferred_values_kernel, dim=1, inputs=[values_out], block_dim=block_dim, device=device)
            np.testing.assert_array_equal(values_out.numpy(), [11, 22])
            if device.is_cuda:
                test.assertEqual(wp.get_suggested_block_size(deferred_values_kernel, device), occupancy)
                test.assertEqual(wp.get_cuda_kernel_properties(deferred_values_kernel, device=device), properties)
            else:
                default_exec.get_kernel_hooks(deferred_values_kernel)
            test.assertIs(deferred_values_kernel.module.load(device), default_exec)


def test_block_dependent_helper_symbols(test, device):
    """Use each variant's symbols when helper statics depend on the block size."""

    @wp.func
    def write_tile_length(tile_length_out: wp.array[int]):
        tile = wp.tile(1)
        tile_length_out[0] = wp.static(len(tile))

    @wp.kernel(module="unique")
    def tile_length_kernel(tile_length_out: wp.array[int]):
        write_tile_length(tile_length_out)

    tile_length_out = wp.zeros(1, dtype=int, device=device)
    # Hash another variant before emitting the default one. The builder must
    # use the default hasher's symbols even if Kernel.hash has since changed.
    tile_length_kernel.module.get_module_hash(64)
    properties = wp.get_cuda_kernel_properties(tile_length_kernel, device=device)
    occupancy = wp.get_suggested_block_size(tile_length_kernel, device)
    for block_dim in (256, 64, 128, 256, 64):
        with test.subTest(block_dim=block_dim):
            wp.launch(tile_length_kernel, dim=1, inputs=[tile_length_out], block_dim=block_dim, device=device)
            np.testing.assert_array_equal(tile_length_out.numpy(), [block_dim])
            test.assertEqual(wp.get_suggested_block_size(tile_length_kernel, device), occupancy)
            test.assertEqual(wp.get_cuda_kernel_properties(tile_length_kernel, device=device), properties)


def test_variant_symbols_for_late_alias(test, device):
    """Resolve an equivalent kernel registered after an executable was loaded."""

    def make_tile_length_kernel(module):
        @wp.kernel(module=module)
        def tile_length_kernel(tile_length_out: wp.array[int]):
            tile = wp.tile(1)
            tile_length_out[0] = wp.static(len(tile))

        return tile_length_kernel

    module = wp.get_module(f"{__name__}.late_alias_{device.alias.replace(':', '_')}")
    original_kernel = make_tile_length_kernel(module)
    tile_length_out = wp.zeros(1, dtype=int, device=device)
    default_exec = module.load(device)
    equivalent_kernel = make_tile_length_kernel(module)
    test.assertIsNot(original_kernel, equivalent_kernel)
    test.assertIs(module.load(device), default_exec)
    wp.launch(original_kernel, dim=1, inputs=[tile_length_out], block_dim=64, device=device)
    np.testing.assert_array_equal(tile_length_out.numpy(), [64])
    wp.launch(equivalent_kernel, dim=1, inputs=[tile_length_out], block_dim=256, device=device)
    np.testing.assert_array_equal(tile_length_out.numpy(), [256])


def test_unique_helper_late_target_table(test, device):
    """Allow helper target tables to be populated between decoration and launch."""
    late_bound_targets = [None]

    @wp.func
    def call_late_bound_targets(value_out: wp.array[int]):
        for index in range(wp.static(len(late_bound_targets))):
            wp.static(late_bound_targets[index])(value_out)

    @wp.kernel(module="unique", enable_backward=False)
    def late_bound_targets_kernel(value_out: wp.array[int]):
        call_late_bound_targets(value_out)

    @wp.func
    def write_constant_value(value_out: wp.array[int]):
        value_out[0] = 42

    late_bound_targets[0] = write_constant_value
    value_out = wp.zeros(1, dtype=int, device=device)
    wp.launch(late_bound_targets_kernel, dim=1, inputs=[value_out], device=device)
    np.testing.assert_array_equal(value_out.numpy(), [42])


class TestModuleVariants(unittest.TestCase):
    """Test variant symbol ownership and unique-kernel compatibility."""

    def test_retained_executable_late_alias(self):
        """Resolve a late alias after registration invalidates cached hashers."""
        module = wp.get_module(f"{__name__}.retained_alias")

        def make_constant_value_kernel():
            @wp.kernel(module=module)
            def constant_value_kernel(value_out: wp.array[int]):
                value_out[0] = 1

            return constant_value_kernel

        original_kernel = make_constant_value_kernel()
        retained_exec = module.load("cpu", block_dim=256)
        equivalent_kernel = make_constant_value_kernel()
        module.get_module_hash(64)
        self.assertIs(
            retained_exec.get_kernel_hooks(equivalent_kernel), retained_exec.get_kernel_hooks(original_kernel)
        )

    def test_unique_custom_gradient_reuse(self):
        """Reuse identical unique kernels that call a helper with a custom gradient."""

        @wp.func
        def square(x: float):
            return x * x

        @wp.func_grad(square)
        def square_grad(x: float, adj_ret: float):
            wp.adjoint[x] += 2.0 * x * adj_ret

        def make_square_kernel():
            @wp.kernel(module="unique")
            def square_kernel(input_values: wp.array[float], squared_values: wp.array[float]):
                squared_values[0] = square(input_values[0])

            return square_kernel

        original_kernel = make_square_kernel()
        for _ in range(9):
            self.assertIs(make_square_kernel(), original_kernel)

    def test_unique_helper_static_evaluation_timing(self):
        """Leave deferred helper statics unevaluated until module compilation."""
        static_evaluation_events = []

        def record_static_evaluation(index):
            static_evaluation_events.append(index)
            return index + 1

        @wp.func
        def write_evaluated_static_values(values_out: wp.array[int]):
            for index in range(2):
                values_out[index] = wp.static(record_static_evaluation(index))

        @wp.kernel(module="unique", enable_backward=False)
        def static_evaluation_kernel(values_out: wp.array[int]):
            write_evaluated_static_values(values_out)

        self.assertEqual(static_evaluation_events, [])
        values_out = wp.zeros(2, dtype=int, device="cpu")
        # A cached binary bypasses the code-generation side effects checked here.
        # Force compilation without clearing the shared cache.
        with mock.patch.object(wp.config, "cache_kernels", False):
            wp.launch(static_evaluation_kernel, dim=1, inputs=[values_out], device="cpu")
        np.testing.assert_array_equal(values_out.numpy(), [1, 2])
        self.assertTrue(static_evaluation_events)


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

add_function_test(
    TestModuleVariants, "test_deferred_helper_variant_symbols", test_deferred_helper_variant_symbols, devices=devices
)
add_function_test(
    TestModuleVariants, "test_block_dependent_helper_symbols", test_block_dependent_helper_symbols, devices=cuda_devices
)
add_function_test(
    TestModuleVariants, "test_variant_symbols_for_late_alias", test_variant_symbols_for_late_alias, devices=cuda_devices
)
add_function_test(
    TestModuleVariants, "test_unique_helper_late_target_table", test_unique_helper_late_target_table, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
