# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test CPU loop vectorization and its code generation prerequisites."""

import unittest
from typing import Any

import numpy as np

import warp as wp

LOOP_COLUMNS = 64


@wp.kernel(module="unique", module_options={"enable_backward": True, "optimization_level": 2})
def cpu_simple_array_kernel(a: wp.array[float], output: wp.array[float]):
    i = wp.tid()
    output[i] = a[i] * 2.0


@wp.kernel(module="unique", module_options={"enable_backward": False, "optimization_level": 2})
def cpu_mixed_dimensional_array_kernel(a: wp.array[float], b: wp.array2d[float], output: wp.array[float]):
    i = wp.tid()
    output[i] = a[i] + b[i, 0]


@wp.kernel(module="unique", module_options={"enable_backward": False, "optimization_level": 2})
def cpu_linear_array_loop_kernel(a: wp.array[float], output: wp.array[float]):
    row = wp.tid()
    begin = row * LOOP_COLUMNS
    for column in range(LOOP_COLUMNS):
        i = begin + column
        output[i] = a[i] * 2.0 + 1.0


@wp.kernel(
    module="unique",
    module_options={
        "enable_backward": False,
        "optimization_level": 2,
        "cpu_compiler_flags": "-march=native -fno-vectorize",
    },
)
def cpu_linear_array_loop_opt_out_kernel(a: wp.array[float], output: wp.array[float]):
    row = wp.tid()
    begin = row * LOOP_COLUMNS
    for column in range(LOOP_COLUMNS):
        i = begin + column
        output[i] = a[i] * 2.0 + 1.0


@wp.kernel(module="unique", module_options={"max_unroll": 1})
def cpu_generic_constant_range_kernel(values: wp.array[Any]):
    row = wp.tid()
    for column in range(LOOP_COLUMNS):
        i = row * LOOP_COLUMNS + column
        values[i] *= 2.0


wp.overload(cpu_generic_constant_range_kernel, [wp.array[float]])


@wp.kernel(module="unique", module_options={"max_unroll": 1})
def cpu_constant_range_adjoint_kernel(values: wp.array[float], output: wp.array[float]):
    i = wp.tid()
    value = values[i]
    for _ in range(-20, 21, 3):
        value *= 1.01
    for _ in range(20, -21, -3):
        value *= 0.99
    output[i] = value


class TestCpuVectorization(unittest.TestCase):
    """Validate CPU loop lowering and contiguous-array specialization."""

    def test_linear_array_loop_contiguous_and_strided(self):
        rows = 5
        values_np = np.linspace(-2.0, 3.0, rows * LOOP_COLUMNS, dtype=np.float32)

        values = wp.array(values_np, dtype=float, device="cpu")
        output = wp.empty_like(values)
        wp.launch(cpu_linear_array_loop_kernel, dim=rows, inputs=[values], outputs=[output], device="cpu", block_dim=1)
        np.testing.assert_allclose(output.numpy(), values_np * 2.0 + 1.0)

        values_base = wp.zeros(len(values_np) * 2, dtype=float, device="cpu")
        output_base = wp.full(len(values_np) * 2, -100.0, dtype=float, device="cpu")
        strided_values = values_base[::2]
        strided_output = output_base[::2]
        wp.copy(strided_values, values)
        wp.launch(
            cpu_linear_array_loop_kernel,
            dim=rows,
            inputs=[strided_values],
            outputs=[strided_output],
            device="cpu",
            block_dim=1,
        )
        output_base_np = output_base.numpy()
        np.testing.assert_allclose(output_base_np[::2], values_np * 2.0 + 1.0)
        np.testing.assert_array_equal(output_base_np[1::2], np.full(len(values_np), -100.0, dtype=np.float32))

    def test_linear_array_loop_opt_out(self):
        """The embedded Clang frontend accepts the loop vectorizer opt-out."""
        values_np = np.linspace(-2.0, 3.0, LOOP_COLUMNS, dtype=np.float32)
        values = wp.array(values_np, dtype=float, device="cpu")
        output = wp.empty_like(values)

        self.assertIn("-fno-vectorize", cpu_linear_array_loop_opt_out_kernel.module.options["cpu_compiler_flags"])
        wp.launch(
            cpu_linear_array_loop_opt_out_kernel,
            dim=1,
            inputs=[values],
            outputs=[output],
            device="cpu",
            block_dim=1,
        )
        np.testing.assert_allclose(output.numpy(), values_np * 2.0 + 1.0)

    def test_contiguous_specialization_preserves_aliasing(self):
        rows = 2
        element_count = rows * LOOP_COLUMNS

        in_place_np = np.linspace(-1.0, 2.0, element_count, dtype=np.float32)
        in_place = wp.array(in_place_np, dtype=float, device="cpu")
        wp.launch(
            cpu_linear_array_loop_kernel,
            dim=rows,
            inputs=[in_place],
            outputs=[in_place],
            device="cpu",
            block_dim=1,
        )
        np.testing.assert_allclose(in_place.numpy(), in_place_np * 2.0 + 1.0)

        overlap_elements = LOOP_COLUMNS
        overlap_np = np.arange(overlap_elements + 1, dtype=np.float32)
        overlap = wp.array(overlap_np, dtype=float, device="cpu")
        wp.launch(
            cpu_linear_array_loop_kernel,
            dim=1,
            inputs=[overlap[:-1]],
            outputs=[overlap[1:]],
            device="cpu",
            block_dim=1,
        )
        for i in range(overlap_elements):
            overlap_np[i + 1] = overlap_np[i] * 2.0 + 1.0
        np.testing.assert_allclose(overlap.numpy(), overlap_np)

    def test_generic_loop_linear_tid_specialization(self):
        module = cpu_generic_constant_range_kernel.module
        block_dim = 1
        module.mark_modified()
        options = module.resolve_options(wp.config, block_dim=block_dim)

        self.assertNotIn(block_dim, module.hashers)
        source, _, _, _, _ = module._run_codegen(options, is_cpu=True)
        self.assertIn("template <bool wp_linear_contiguous>", source)
        self.assertIn("_cpu_forward_dispatch(dim, _wp_args)", source)
        self.assertIn("_cpu_backward_dispatch(dim, _wp_args, _wp_adj_args)", source)
        self.assertIn("const bool wp_linear_contiguous = dim->coord_mult == 1", source)
        self.assertIn("_wp_args->values.strides[0] == (int)sizeof(wp::float32)", source)
        self.assertIn("_wp_adj_args->values.strides[0] == (int)sizeof(wp::float32)", source)

    def test_cpu_template_capability_gating(self):
        module = cpu_simple_array_kernel.module
        name = cpu_simple_array_kernel.get_mangled_name()
        block_dim = 1
        module.mark_modified()
        options = module.resolve_options(wp.config, block_dim=block_dim)

        source, _, _, _, _ = module._run_codegen(options, is_cpu=True)
        self.assertIn("template <bool wp_linear_contiguous>", source)
        self.assertIn("_cpu_forward_dispatch(dim, _wp_args)", source)
        self.assertIn("_cpu_backward_dispatch(dim, _wp_args, _wp_adj_args)", source)
        self.assertIn(f"{name}_cpu_forward_tasks<true>", source)
        self.assertIn(f"{name}_cpu_backward_tasks<true>", source)
        self.assertIn("wp_contiguous_args", source)

        mixed_module = cpu_mixed_dimensional_array_kernel.module
        mixed_name = cpu_mixed_dimensional_array_kernel.get_mangled_name()
        mixed_module.mark_modified()
        mixed_options = mixed_module.resolve_options(wp.config, block_dim=block_dim)
        mixed_source, _, _, _, _ = mixed_module._run_codegen(mixed_options, is_cpu=True)
        self.assertIn(f"{mixed_name}_cpu_forward_tasks<false>", mixed_source)
        self.assertNotIn(f"{mixed_name}_cpu_forward_tasks<true>", mixed_source)
        self.assertNotIn("const bool wp_linear_contiguous", mixed_source)
        self.assertNotIn(f"{mixed_name}_cpu_kernel_forward<true", mixed_source)
        self.assertNotIn("wp_contiguous_args", mixed_source)

    def test_constant_range_adjoint(self):
        values_np = np.linspace(-2.0, 3.0, 11, dtype=np.float32)
        values = wp.array(values_np, dtype=float, device="cpu", requires_grad=True)
        output = wp.empty_like(values, requires_grad=True)

        with wp.Tape() as tape:
            wp.launch(
                cpu_constant_range_adjoint_kernel,
                dim=len(values),
                inputs=[values],
                outputs=[output],
                device="cpu",
                block_dim=1,
            )
        tape.backward(grads={output: wp.ones_like(output)})

        scale = 1.01 ** len(range(-20, 21, 3)) * 0.99 ** len(range(20, -21, -3))
        np.testing.assert_allclose(output.numpy(), values_np * scale, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(values.grad.numpy(), np.full_like(values_np, scale), rtol=1.0e-6, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
