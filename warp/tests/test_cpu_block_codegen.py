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
    value = values[i] * float(wp.block_dim())
    for _ in range(17):
        value += 1.0
    output[i] = value


def _generate_source(block_dim):
    module = cpu_block_codegen_kernel.module
    module.get_module_hash(block_dim)
    options = module.resolved_options[block_dim]
    return module._run_codegen(options, is_cpu=True)[0]


def _launch_specialization(block_dim, values, output, *, adjoint=False, adj_values=None, adj_output=None):
    previous = wp.config.enable_cpu_blocks
    wp.config.enable_cpu_blocks = True
    try:
        command = Launch(
            cpu_block_codegen_kernel,
            wp.get_device("cpu"),
            block_dim=block_dim,
            adjoint=adjoint,
        )
    finally:
        wp.config.enable_cpu_blocks = previous
    command.set_dim(len(values))
    command.set_params((values, output))
    if adjoint:
        command.set_param_at_index(0, adj_values, adjoint=True)
        command.set_param_at_index(1, adj_output, adjoint=True)
    command.launch()


@wp.kernel(module="unique", enable_backward=False)
def cpu_scalar_tid_count_kernel(counts: wp.array[int]):
    i = wp.tid()
    wp.atomic_add(counts, i, 1)


class TestCpuBlockCodegen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_source_selects_cpu_template(self):
        name = cpu_block_codegen_kernel.get_mangled_name()
        one_lane_source = _generate_source(1)
        self.assertIn("#define WP_TILE_BLOCK_DIM 1", one_lane_source)
        self.assertIn("for (size_t task_index = 0; task_index < dim->size; ++task_index)", one_lane_source)
        self.assertIn("template <bool wp_linear_contiguous>", one_lane_source)
        self.assertIn(f"{name}_cpu_forward_dispatch(dim, _wp_args)", one_lane_source)
        self.assertIn(f"{name}_cpu_backward_dispatch(dim, _wp_args, _wp_adj_args)", one_lane_source)
        self.assertIn("const bool wp_linear_contiguous = dim->coord_mult == 1", one_lane_source)
        self.assertIn(f"{name}_cpu_kernel_forward<wp_linear_contiguous>", one_lane_source)
        self.assertIn(f"{name}_cpu_forward_tasks<false>", one_lane_source)
        self.assertIn("strides[0] == (int)sizeof(wp::float32)", one_lane_source)
        self.assertIn("wp_contiguous_args.values.strides[0] = sizeof(wp::float32)", one_lane_source)
        self.assertRegex(one_lane_source, r"if \(var_\d+ >= 17\) goto end_for_0;")
        self.assertRegex(one_lane_source, r"var_\d+ \+= 1;")
        self.assertRegex(one_lane_source, r"if \(var_\d+ < 0\) goto end_for_0;")
        self.assertRegex(one_lane_source, r"var_\d+ -= 1;")
        self.assertNotIn("iter_cmp(", one_lane_source)
        self.assertNotIn("iter_reverse(", one_lane_source)
        self.assertNotIn("#if WP_TILE_BLOCK_DIM == 1", one_lane_source)
        self.assertNotIn(f"{name}_cpu_block_thunk_forward", one_lane_source)
        self.assertNotIn(f"{name}_cpu_block_thunk_backward", one_lane_source)
        self.assertNotIn("wp_cpu_run_block", one_lane_source)

        cooperative_source = _generate_source(4)
        self.assertIn("#define WP_TILE_BLOCK_DIM 4", cooperative_source)
        self.assertIn(f"{name}_cpu_block_thunk_forward", cooperative_source)
        self.assertIn(f"{name}_cpu_block_thunk_backward", cooperative_source)
        self.assertIn("const size_t remaining = total - block_first", cooperative_source)
        self.assertIn("block_first += (size_t)active_count", cooperative_source)
        self.assertIn("wp::tile_shared_storage_t::bind(payload->tile_mem)", cooperative_source)
        self.assertIn(f"{name}_cpu_kernel_forward<false>", cooperative_source)
        self.assertNotIn("#if WP_TILE_BLOCK_DIM == 1", cooperative_source)
        self.assertNotIn("for (size_t task_index = 0; task_index < dim->size; ++task_index)", cooperative_source)
        self.assertNotIn(f"{name}_cpu_forward_dispatch", cooperative_source)
        self.assertNotIn(f"{name}_cpu_backward_dispatch", cooperative_source)
        self.assertNotIn("const bool wp_linear_contiguous", cooperative_source)

    def test_scalar_tid_preserves_dimensional_aliasing(self):
        counts = wp.zeros(3, dtype=int, device="cpu")
        wp.launch(cpu_scalar_tid_count_kernel, dim=(3, 4), inputs=[counts], device="cpu", block_dim=1)
        np.testing.assert_array_equal(counts.numpy(), np.full(3, 4, dtype=np.int32))

    def test_forward_and_backward_specializations(self):
        for block_dim in (1, 4):
            with self.subTest(block_dim=block_dim):
                host_values = np.arange(11, dtype=np.float32) - 3.0
                values = wp.array(host_values, dtype=float, device="cpu")
                output = wp.zeros(len(host_values), dtype=float, device="cpu")
                _launch_specialization(block_dim, values, output)
                np.testing.assert_allclose(output.numpy(), host_values * block_dim + 17.0)

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
            b"wp_fiber_",
        ):
            with self.subTest(symbol=symbol.decode()):
                self.assertNotIn(symbol, object_bytes)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
