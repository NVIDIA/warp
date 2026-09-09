# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the one-lane and cooperative-fiber CPU code-generation paths."""

import os
import unittest

import numpy as np

import warp as wp
from warp._src.context import Launch


@wp.kernel(module="unique")
def cpu_block_codegen_kernel(values: wp.array[float], output: wp.array[float]):
    i = wp.tid()
    output[i] = values[i] * float(wp.block_dim()) + 1.0


def _generate_source(block_dim):
    module = cpu_block_codegen_kernel.module
    module.get_module_hash(block_dim)
    options = module.resolved_options[block_dim]
    return module._run_codegen(options, is_cpu=True)[0]


def _launch_specialization(block_dim, values, output, *, adjoint=False, adj_values=None, adj_output=None):
    command = Launch(cpu_block_codegen_kernel, wp.get_device("cpu"), block_dim=block_dim, adjoint=adjoint)
    command.set_dim(len(values))
    command.set_params((values, output))
    if adjoint:
        command.set_param_at_index(0, adj_values, adjoint=True)
        command.set_param_at_index(1, adj_output, adjoint=True)
    command.launch()


class TestCpuBlockCodegen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_source_contains_separate_one_and_many_lane_paths(self):
        name = cpu_block_codegen_kernel.get_mangled_name()
        for block_dim in (1, 4):
            with self.subTest(block_dim=block_dim):
                source = _generate_source(block_dim)
                self.assertIn(f"#define WP_TILE_BLOCK_DIM {block_dim}", source)
                self.assertIn("#if WP_TILE_BLOCK_DIM == 1", source)
                self.assertIn("for (size_t task_index = 0; task_index < dim->size; ++task_index)", source)
                self.assertIn(f"{name}_cpu_block_thunk_forward", source)
                self.assertIn(f"{name}_cpu_block_thunk_backward", source)
                self.assertIn("const size_t remaining = total - block_first", source)
                self.assertIn("block_first += (size_t)active_count", source)
                self.assertIn("wp::tile_shared_storage_t::bind(payload->tile_mem)", source)

    def test_forward_and_backward_specializations(self):
        for block_dim in (1, 4):
            with self.subTest(block_dim=block_dim):
                host_values = np.arange(11, dtype=np.float32) - 3.0
                values = wp.array(host_values, dtype=float, device="cpu")
                output = wp.zeros(len(host_values), dtype=float, device="cpu")
                _launch_specialization(block_dim, values, output)
                np.testing.assert_allclose(output.numpy(), host_values * block_dim + 1.0)

                adj_values = wp.zeros_like(values)
                adj_output = wp.ones_like(output)
                _launch_specialization(
                    block_dim,
                    values,
                    output,
                    adjoint=True,
                    adj_values=adj_values,
                    adj_output=adj_output,
                )
                np.testing.assert_allclose(adj_values.numpy(), np.full_like(host_values, block_dim))

    def test_one_lane_object_has_no_fiber_runtime_symbols(self):
        values = wp.zeros(1, dtype=float, device="cpu")
        output = wp.zeros_like(values)
        _launch_specialization(1, values, output)

        module = cpu_block_codegen_kernel.module
        device = wp.get_device("cpu")
        module_dir = os.path.join(wp.config.kernel_cache_dir, module.get_module_identifier(block_dim=1))
        object_path = os.path.join(module_dir, module._get_compile_output_name(device, block_dim=1))
        with open(object_path, "rb") as object_file:
            object_bytes = object_file.read()

        for symbol in (
            b"wp_cpu_run_block",
            b"wp_cpu_get_thread_idx",
            b"wp_cpu_get_active_count",
            b"wp_cpu_tile_sync",
            b"wp_cpu_block_pool_size",
            b"wp_fiber_",
        ):
            with self.subTest(symbol=symbol.decode()):
                self.assertNotIn(symbol, object_bytes)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
