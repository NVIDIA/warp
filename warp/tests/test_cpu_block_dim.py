# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test opt-in resolution and replay of CPU kernel block dimensions."""

import contextlib
import unittest
from unittest import mock

import numpy as np

import warp as wp
import warp._src.context as wp_context
from warp.autograd import jacobian


@contextlib.contextmanager
def _cpu_blocks(enabled):
    previous = wp.config.enable_cpu_blocks
    wp.config.enable_cpu_blocks = enabled
    try:
        yield
    finally:
        wp.config.enable_cpu_blocks = previous


@wp.kernel
def _block_info_kernel(thread_indices: wp.array[wp.int32], block_dims: wp.array[wp.int32]):
    i = wp.tid()
    thread_indices[i] = i
    block_dims[i] = wp.block_dim()


@wp.kernel
def _block_scale_kernel(values: wp.array[float], output: wp.array[float]):
    i = wp.tid()
    output[i] = values[i] * float(wp.block_dim())


@wp.kernel
def _no_tid_counter(output: wp.array[wp.int32]):
    wp.atomic_add(output, 0, 1)


@wp.kernel(module="unique")
def _asan_gate_kernel(output: wp.array[wp.int32]):
    output[0] = 1


@wp.kernel(module="unique")
def _force_load_gate_kernel(output: wp.array[wp.int32]):
    output[0] = 1


@wp.kernel(module="unique")
def _force_load_preserve_kernel(output: wp.array[wp.int32]):
    output[0] = 1


@wp.kernel(module="unique")
def _force_load_default_kernel(output: wp.array[wp.int32]):
    output[0] = 1


@wp.func
def _mapped_block_dim(_value: float):
    return float(wp.block_dim())


@wp.func
def _nested_block_value(value: int):
    return wp.block_dim() * 100000 + value


@wp.kernel
def _tid_1d_kernel(output: wp.array[wp.int32]):
    i = wp.tid()
    output[i] = _nested_block_value(i)


@wp.kernel
def _tid_2d_kernel(output: wp.array2d[wp.int32]):
    i, j = wp.tid()
    output[i, j] = _nested_block_value(i * 1000 + j)


@wp.kernel
def _tid_3d_kernel(output: wp.array3d[wp.int32]):
    i, j, k = wp.tid()
    output[i, j, k] = _nested_block_value(i * 10000 + j * 100 + k)


@wp.kernel
def _tid_4d_kernel(output: wp.array4d[wp.int32]):
    i, j, k, ell = wp.tid()
    output[i, j, k, ell] = _nested_block_value(i * 10000000 + j * 10000 + k * 100 + ell)


class TestCpuBlockDim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.cpu = wp.get_device("cpu")

    def test_resolution_table(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), _cpu_blocks(enabled):
                for requested in (None, -7, 0, 1):
                    self.assertEqual(wp_context._resolve_launch_block_dim(self.cpu, requested), 1)

                self.assertEqual(wp_context._resolve_launch_block_dim(self.cpu, 2), 2 if enabled else 1)
                self.assertEqual(wp_context._resolve_launch_block_dim(self.cpu, 1024), 1024 if enabled else 1)
                with self.assertRaisesRegex(ValueError, "at most 1024 on CPU"):
                    wp_context._resolve_launch_block_dim(self.cpu, 1025)

    def test_cuda_resolution_is_unchanged(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA is not available")
        cuda = wp.get_device("cuda:0")
        for enabled in (False, True):
            with self.subTest(enabled=enabled), _cpu_blocks(enabled):
                for requested, expected in ((None, 256), (-1, 256), (0, 256), (1, 1), (2, 2), (1025, 1025)):
                    self.assertEqual(wp_context._resolve_launch_block_dim(cuda, requested), expected)

                output = wp.zeros(5, dtype=wp.int32, device=cuda)
                indices = wp.empty_like(output)
                wp.launch(
                    _block_info_kernel,
                    dim=5,
                    inputs=[indices, output],
                    device=cuda,
                    block_dim=2,
                )
                np.testing.assert_array_equal(output.numpy(), np.full(5, 2, dtype=np.int32))

    def test_config_toggle_selects_distinct_specializations(self):
        count = 7
        indices = wp.empty(count, dtype=wp.int32, device=self.cpu)
        block_dims = wp.empty_like(indices)

        with _cpu_blocks(False):
            wp.launch(_block_info_kernel, dim=count, inputs=[indices, block_dims], device=self.cpu, block_dim=4)
            np.testing.assert_array_equal(block_dims.numpy(), np.ones(count, dtype=np.int32))

        with _cpu_blocks(True):
            wp.launch(_block_info_kernel, dim=count, inputs=[indices, block_dims], device=self.cpu, block_dim=4)
            np.testing.assert_array_equal(block_dims.numpy(), np.full(count, 4, dtype=np.int32))

        with _cpu_blocks(False):
            wp.launch(_block_info_kernel, dim=count, inputs=[indices, block_dims], device=self.cpu, block_dim=4)
            np.testing.assert_array_equal(block_dims.numpy(), np.ones(count, dtype=np.int32))

        np.testing.assert_array_equal(indices.numpy(), np.arange(count, dtype=np.int32))
        self.assertIn((self.cpu.context, 1), _block_info_kernel.module.execs)
        self.assertIn((self.cpu.context, 4), _block_info_kernel.module.execs)

    def test_direct_launch_uses_cpu_block_resolution(self):
        with _cpu_blocks(False):
            default_command = wp.Launch(_block_info_kernel, self.cpu)
            disabled_command = wp.Launch(_block_info_kernel, self.cpu, block_dim=4)
        self.assertEqual(default_command.block_dim, 1)
        self.assertEqual(disabled_command.block_dim, 1)

        with _cpu_blocks(True):
            enabled_command = wp.Launch(_block_info_kernel, self.cpu, block_dim=4)
        self.assertEqual(enabled_command.block_dim, 4)

    def test_force_load_uses_cpu_block_resolution(self):
        module = _force_load_gate_kernel.module
        with _cpu_blocks(False):
            wp.force_load(device=self.cpu, modules=[module], block_dim=4, max_workers=0)
        self.assertIn((self.cpu.context, 1), module.execs)
        self.assertNotIn((self.cpu.context, 4), module.execs)

        with _cpu_blocks(True):
            wp.force_load(device=self.cpu, modules=[module], block_dim=4, max_workers=0)
        self.assertIn((self.cpu.context, 4), module.execs)

    def test_force_load_without_request_uses_cpu_launch_default(self):
        module = _force_load_default_kernel.module
        with _cpu_blocks(True):
            wp.force_load(device=self.cpu, modules=[module], max_workers=0)
        self.assertIn((self.cpu.context, 1), module.execs)
        self.assertNotIn((self.cpu.context, 256), module.execs)

    def test_force_load_preserves_existing_effective_dimension(self):
        module = _force_load_preserve_kernel.module
        with _cpu_blocks(True):
            wp.force_load(device=self.cpu, modules=[module], block_dim=4, max_workers=0)
        self.assertIn((self.cpu.context, 4), module.execs)
        self.assertNotIn((self.cpu.context, 1), module.execs)

        # With no new request, force_load() reuses already-effective variants
        # instead of reinterpreting them through the current configuration.
        with _cpu_blocks(False):
            wp.force_load(device=self.cpu, modules=[module], max_workers=0)
        self.assertIn((self.cpu.context, 4), module.execs)
        self.assertNotIn((self.cpu.context, 1), module.execs)

    def test_all_valid_block_dimensions_with_full_blocks_and_tails(self):
        block_dims = (1, 2, 8, 31, 32, 63, 64, 65, 255, 256, 257, 511, 512, 1023, 1024)
        with _cpu_blocks(True):
            for block_dim in block_dims:
                with self.subTest(block_dim=block_dim):
                    count = block_dim + 1
                    indices = wp.empty(count, dtype=wp.int32, device=self.cpu)
                    actual_block_dims = wp.empty_like(indices)

                    # Repeating the same full-block-plus-one-tail launch also
                    # exercises warm worker reuse for every supported size.
                    for _ in range(2):
                        wp.launch(
                            _block_info_kernel,
                            dim=count,
                            inputs=[indices, actual_block_dims],
                            device=self.cpu,
                            block_dim=block_dim,
                        )

                    np.testing.assert_array_equal(indices.numpy(), np.arange(count, dtype=np.int32))
                    np.testing.assert_array_equal(actual_block_dims.numpy(), np.full(count, block_dim, dtype=np.int32))

    def test_empty_launch_and_arbitrary_partial_prefix(self):
        output = wp.zeros(1, dtype=wp.int32, device=self.cpu)
        with _cpu_blocks(True):
            for block_dim in (1, 32, 1024):
                with self.subTest(empty_block_dim=block_dim):
                    wp.launch(_no_tid_counter, dim=0, inputs=[output], device=self.cpu, block_dim=block_dim)
            self.assertEqual(output.numpy()[0], 0)

            block_dim = 65
            count = block_dim + 17
            indices = wp.empty(count, dtype=wp.int32, device=self.cpu)
            actual_block_dims = wp.empty_like(indices)
            wp.launch(
                _block_info_kernel,
                dim=count,
                inputs=[indices, actual_block_dims],
                device=self.cpu,
                block_dim=block_dim,
            )
            np.testing.assert_array_equal(indices.numpy(), np.arange(count, dtype=np.int32))
            np.testing.assert_array_equal(actual_block_dims.numpy(), np.full(count, block_dim, dtype=np.int32))

    def test_multidimensional_tid_and_nested_function(self):
        block_dim = 8
        cases = (
            (_tid_1d_kernel, (5,), lambda i: i[0]),
            (_tid_2d_kernel, (2, 3), lambda i: i[0] * 1000 + i[1]),
            (_tid_3d_kernel, (2, 3, 4), lambda i: i[0] * 10000 + i[1] * 100 + i[2]),
            (
                _tid_4d_kernel,
                (2, 2, 3, 2),
                lambda i: i[0] * 10000000 + i[1] * 10000 + i[2] * 100 + i[3],
            ),
        )
        with _cpu_blocks(True):
            for kernel, shape, encode in cases:
                with self.subTest(shape=shape):
                    output = wp.empty(shape=shape, dtype=wp.int32, device=self.cpu)
                    wp.launch(kernel, dim=shape, outputs=[output], device=self.cpu, block_dim=block_dim)
                    expected = np.empty(shape, dtype=np.int32)
                    for index in np.ndindex(shape):
                        expected[index] = block_dim * 100000 + encode(index)
                    np.testing.assert_array_equal(output.numpy(), expected)

    def test_launch_tiled_uses_effective_dimension(self):
        for enabled, expected in ((False, 2), (True, 8)):
            with self.subTest(enabled=enabled), _cpu_blocks(enabled):
                output = wp.zeros(1, dtype=wp.int32, device=self.cpu)
                wp.launch_tiled(
                    _no_tid_counter,
                    dim=[2],
                    inputs=[output],
                    block_dim=4,
                    device=self.cpu,
                )
                self.assertEqual(output.numpy()[0], expected)

    def test_forward_and_tape_backward_retain_effective_dimension(self):
        values = wp.array(np.arange(7, dtype=np.float32), dtype=float, device=self.cpu, requires_grad=True)
        output = wp.zeros_like(values, requires_grad=True)

        with _cpu_blocks(True):
            with wp.Tape() as tape:
                wp.launch(
                    _block_scale_kernel,
                    dim=len(values),
                    inputs=[values],
                    outputs=[output],
                    device=self.cpu,
                    block_dim=4,
                )

        np.testing.assert_allclose(output.numpy(), values.numpy() * 4.0)
        self.assertEqual(tape.launches[0][6], 4)

        # The recorded effective value must not be re-resolved through the now
        # disabled option when the tape launches the backward specialization.
        with _cpu_blocks(False):
            tape.backward(grads={output: wp.ones_like(output)})
        np.testing.assert_allclose(values.grad.numpy(), np.full(len(values), 4.0, dtype=np.float32))

    def test_recorded_command_retains_effective_dimension(self):
        indices = wp.empty(5, dtype=wp.int32, device=self.cpu)
        block_dims = wp.empty_like(indices)
        with _cpu_blocks(True):
            command = wp.launch(
                _block_info_kernel,
                dim=5,
                inputs=[indices, block_dims],
                device=self.cpu,
                block_dim=4,
                record_cmd=True,
            )

        self.assertEqual(command.block_dim, 4)
        self.assertEqual(command.module_exec.block_dim, 4)
        with _cpu_blocks(False):
            command.launch()
        np.testing.assert_array_equal(block_dims.numpy(), np.full(5, 4, dtype=np.int32))

    def test_cpu_capture_retains_effective_dimension(self):
        indices = wp.empty(5, dtype=wp.int32, device=self.cpu)
        block_dims = wp.zeros_like(indices)
        with _cpu_blocks(True):
            _block_info_kernel.module.load(self.cpu, block_dim=4)
            with wp.ScopedCapture(device=self.cpu, force_module_load=False) as capture:
                wp.launch(
                    _block_info_kernel,
                    dim=5,
                    inputs=[indices, block_dims],
                    device=self.cpu,
                    block_dim=4,
                )

        with _cpu_blocks(False):
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(block_dims.numpy(), np.full(5, 4, dtype=np.int32))

    def test_cpu_block_dimension_cap_applies_with_both_config_states(self):
        output = wp.empty(1, dtype=wp.int32, device=self.cpu)
        for enabled in (False, True):
            with self.subTest(enabled=enabled), _cpu_blocks(enabled):
                with self.assertRaisesRegex(ValueError, "at most 1024 on CPU"):
                    wp.launch(_no_tid_counter, dim=0, inputs=[output], device=self.cpu, block_dim=1025)
                with self.assertRaisesRegex(ValueError, "at most 1024 on CPU"):
                    wp.launch_tiled(_no_tid_counter, dim=[0], inputs=[output], device=self.cpu, block_dim=1025)
                with self.assertRaisesRegex(ValueError, "at most 1024 on CPU"):
                    wp.Launch(_no_tid_counter, self.cpu, block_dim=1025)
                with self.assertRaisesRegex(ValueError, "at most 1024 on CPU"):
                    wp.force_load(
                        device=self.cpu,
                        modules=[_force_load_gate_kernel.module],
                        block_dim=1025,
                        max_workers=0,
                    )

    def test_address_sanitizer_gate_precedes_compilation(self):
        output = wp.empty(1, dtype=wp.int32, device=self.cpu)
        self.assertNotIn((self.cpu.context, 2), _asan_gate_kernel.module.execs)
        with (
            _cpu_blocks(True),
            mock.patch.object(
                type(wp_context.runtime), "clang_sanitizer", new_callable=mock.PropertyMock, return_value="address"
            ),
        ):
            message = "Cooperative CPU fibers do not support AddressSanitizer builds"
            with self.assertRaisesRegex(NotImplementedError, message):
                wp.launch(_asan_gate_kernel, dim=0, inputs=[output], device=self.cpu, block_dim=2)
            with self.assertRaisesRegex(NotImplementedError, message):
                wp.launch_tiled(_asan_gate_kernel, dim=[0], inputs=[output], device=self.cpu, block_dim=2)
            with self.assertRaisesRegex(NotImplementedError, message):
                wp.Launch(_asan_gate_kernel, self.cpu, block_dim=2)
            with self.assertRaisesRegex(NotImplementedError, message):
                wp.force_load(
                    device=self.cpu,
                    modules=[_asan_gate_kernel.module],
                    block_dim=2,
                    max_workers=0,
                )
        self.assertNotIn((self.cpu.context, 2), _asan_gate_kernel.module.execs)

    def test_map_and_jacobian_defaults(self):
        values = wp.ones(2, dtype=float, device=self.cpu, requires_grad=True)
        output = wp.zeros_like(values, requires_grad=True)
        with _cpu_blocks(True):
            mapped_default = wp.map(_mapped_block_dim, values)
            mapped_explicit = wp.map(_mapped_block_dim, values, block_dim=2)
            np.testing.assert_array_equal(mapped_default.numpy(), np.ones(2, dtype=np.float32))
            np.testing.assert_array_equal(mapped_explicit.numpy(), np.full(2, 2.0, dtype=np.float32))

            jacobian_default = jacobian(
                _block_scale_kernel,
                dim=2,
                inputs=[values],
                outputs=[output],
                device=self.cpu,
            )
            jacobian_explicit = jacobian(
                _block_scale_kernel,
                dim=2,
                inputs=[values],
                outputs=[output],
                device=self.cpu,
                block_dim=2,
            )
            np.testing.assert_allclose(jacobian_default[0, 0].numpy(), np.eye(2, dtype=np.float32))
            np.testing.assert_allclose(jacobian_explicit[0, 0].numpy(), np.eye(2, dtype=np.float32) * 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
