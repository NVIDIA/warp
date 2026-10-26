# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test CPU loop and SLP vectorization and their code generation prerequisites."""

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

import warp as wp

LOOP_COLUMNS = 64


@wp.kernel(module="unique", module_options={"enable_backward": True, "optimization_level": 2})
def cpu_vec4_component_kernel(a: wp.array[wp.vec4f], b: wp.array[wp.vec4f], output: wp.array[wp.vec4f]):
    i = wp.tid()
    output[i] = wp.cw_mul(a[i], b[i]) + a[i]


@wp.kernel(module="unique", module_options={"enable_backward": False, "optimization_level": 1})
def cpu_vec4_component_o1_kernel(a: wp.array[wp.vec4f], b: wp.array[wp.vec4f], output: wp.array[wp.vec4f]):
    i = wp.tid()
    output[i] = wp.cw_mul(a[i], b[i]) + a[i]


@wp.kernel(
    module="unique",
    module_options={
        "enable_backward": False,
        "optimization_level": 2,
        "cpu_compiler_flags": "-march=native -fno-slp-vectorize",
    },
)
def cpu_vec4_component_slp_off_kernel(a: wp.array[wp.vec4f], b: wp.array[wp.vec4f], output: wp.array[wp.vec4f]):
    i = wp.tid()
    output[i] = wp.cw_mul(a[i], b[i]) + a[i]


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
    """Validate CPU vectorization and contiguous-array specialization."""

    def test_vec4_component_arithmetic(self):
        a_np = np.arange(44, dtype=np.float32).reshape(11, 4) - 7.0
        b_np = np.linspace(-2.0, 3.0, 44, dtype=np.float32).reshape(11, 4)
        expected = a_np * b_np + a_np

        for kernel in (cpu_vec4_component_o1_kernel, cpu_vec4_component_kernel):
            with self.subTest(optimization_level=kernel.module.options["optimization_level"]):
                a = wp.array(a_np, dtype=wp.vec4f, device="cpu")
                b = wp.array(b_np, dtype=wp.vec4f, device="cpu")
                output = wp.empty_like(a)
                wp.launch(kernel, dim=len(a), inputs=[a, b], outputs=[output], device="cpu", block_dim=1)
                np.testing.assert_allclose(output.numpy(), expected, rtol=1.0e-6, atol=1.0e-6)

    def test_vec4_component_slp_optimization_record(self):
        """Check that SLP runs at CPU O2 but not at O1 or with the opt-out."""
        if wp.config.mode == "debug":
            self.skipTest("SLP vectorization is disabled in CPU debug mode")

        a_np = np.arange(44, dtype=np.float32).reshape(11, 4) - 7.0
        b_np = np.linspace(-2.0, 3.0, 44, dtype=np.float32).reshape(11, 4)
        a = wp.array(a_np, dtype=wp.vec4f, device="cpu")
        b = wp.array(b_np, dtype=wp.vec4f, device="cpu")
        output = wp.empty_like(a)
        cases = (
            ("o2", cpu_vec4_component_kernel, True),
            ("o1", cpu_vec4_component_o1_kernel, False),
            ("slp_off", cpu_vec4_component_slp_off_kernel, False),
        )
        slp_record = r"(?m)^Pass:[ \t]+slp-vectorizer[ \t]*$"

        with tempfile.TemporaryDirectory(prefix="wp_slp_record_") as temp_dir:
            if any(character.isspace() for character in temp_dir):
                self.skipTest("CPU compiler flags cannot contain a path with whitespace")

            for name, kernel, should_run_slp in cases:
                with self.subTest(case=name):
                    module = kernel.module
                    original_flags = module.options["cpu_compiler_flags"]
                    compiler_flags = original_flags
                    if compiler_flags is None:
                        compiler_flags = wp.config.cpu_compiler_flags
                    if compiler_flags is None:
                        compiler_flags = "-march=native"
                    record_file = Path(temp_dir) / f"{name}.yaml"
                    flags = f"{compiler_flags} -opt-record-file {record_file} -opt-record-passes slp-vectorizer"

                    try:
                        wp.set_module_options({"cpu_compiler_flags": flags}, module)
                        with (
                            mock.patch.object(wp.config, "cache_kernels", False),
                            mock.patch.object(wp.config, "use_precompiled_headers", False),
                        ):
                            wp.launch(kernel, dim=len(a), inputs=[a, b], outputs=[output], device="cpu", block_dim=1)
                    finally:
                        wp.set_module_options({"cpu_compiler_flags": original_flags}, module)

                    records = record_file.read_text(encoding="utf-8") if record_file.exists() else ""
                    if should_run_slp:
                        self.assertRegex(records, slp_record)
                    else:
                        self.assertNotRegex(records, slp_record)
                    np.testing.assert_allclose(output.numpy(), a_np * b_np + a_np, rtol=1.0e-6, atol=1.0e-6)

    def test_vec4_component_adjoint(self):
        a_np = np.arange(28, dtype=np.float32).reshape(7, 4) - 4.0
        b_np = np.linspace(-1.0, 2.0, 28, dtype=np.float32).reshape(7, 4)
        a = wp.array(a_np, dtype=wp.vec4f, device="cpu", requires_grad=True)
        b = wp.array(b_np, dtype=wp.vec4f, device="cpu", requires_grad=True)
        output = wp.empty_like(a, requires_grad=True)

        with wp.Tape() as tape:
            wp.launch(cpu_vec4_component_kernel, dim=len(a), inputs=[a, b], outputs=[output], device="cpu", block_dim=1)
        tape.backward(grads={output: wp.ones_like(output)})

        np.testing.assert_allclose(a.grad.numpy(), b_np + 1.0, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(b.grad.numpy(), a_np, rtol=1.0e-6, atol=1.0e-6)

    def test_vec4_component_strided_arrays(self):
        values = np.arange(44, dtype=np.float32).reshape(11, 4) - 7.0
        factors = np.linspace(-2.0, 3.0, 44, dtype=np.float32).reshape(11, 4)
        values_base = np.zeros((22, 4), dtype=np.float32)
        factors_base = np.zeros((22, 4), dtype=np.float32)
        values_base[::2] = values
        factors_base[::2] = factors

        a = wp.array(values_base, dtype=wp.vec4f, device="cpu")[::2]
        b = wp.array(factors_base, dtype=wp.vec4f, device="cpu")[::2]
        output_base = wp.zeros(22, dtype=wp.vec4f, device="cpu")
        wp.launch(cpu_vec4_component_kernel, dim=11, inputs=[a, b], outputs=[output_base[::2]], device="cpu")

        result = output_base.numpy()
        np.testing.assert_allclose(result[::2], values * factors + values, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_array_equal(result[1::2], np.zeros((11, 4), dtype=np.float32))

    def test_vec4_component_slp_opt_out(self):
        """Warp accepts the SLP opt-out for a CPU module."""
        a_np = np.arange(44, dtype=np.float32).reshape(11, 4) - 7.0
        b_np = np.linspace(-2.0, 3.0, 44, dtype=np.float32).reshape(11, 4)
        a = wp.array(a_np, dtype=wp.vec4f, device="cpu")
        b = wp.array(b_np, dtype=wp.vec4f, device="cpu")
        output = wp.empty_like(a)

        wp.launch(
            cpu_vec4_component_slp_off_kernel,
            dim=len(a),
            inputs=[a, b],
            outputs=[output],
            device="cpu",
            block_dim=1,
        )
        np.testing.assert_allclose(output.numpy(), a_np * b_np + a_np, rtol=1.0e-6, atol=1.0e-6)

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
        """Warp accepts the loop vectorizer opt-out."""
        values_np = np.linspace(-2.0, 3.0, LOOP_COLUMNS, dtype=np.float32)
        values = wp.array(values_np, dtype=float, device="cpu")
        output = wp.empty_like(values)

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
