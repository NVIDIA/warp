# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp._src.codegen as codegen
from warp._src.context import ModuleBuilder
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


def kernel_source(kernel, device: str, *, llvm_cuda: bool = False) -> str:
    options = kernel.module.resolve_options(wp.config) | kernel.options
    options["llvm_cuda"] = llvm_cuda
    ModuleBuilder(kernel.module, options)
    return codegen.codegen_kernel(kernel, device=device, options=options)


@wp.kernel(enable_cuda_smem_spilling=True, launch_bounds=128, enable_backward=False, module="unique")
def fill_index(a: wp.array[int]):
    a[wp.tid()] = wp.tid()


def test_cuda_smem_spilling_kernel_runs(test, device):
    """Run a kernel with CUDA shared-memory spilling enabled."""
    values = wp.empty(128, dtype=int, device=device)
    wp.launch(fill_index, dim=values.shape, inputs=[values], block_dim=128, device=device)
    assert_np_equal(values.numpy(), np.arange(128, dtype=np.int32))


class TestCudaSmemSpilling(unittest.TestCase):
    def test_invalid_enable_cuda_smem_spilling_rejected(self):
        """Reject invalid CUDA shared-memory spilling values."""
        for value in (0, 1, 1.0, "true"):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "enable_cuda_smem_spilling"):

                @wp.kernel(enable_cuda_smem_spilling=value, module="unique")
                def invalid(a: wp.array[int]):
                    a[wp.tid()] = 0

    def test_false_matches_default_identity(self):
        """Preserve default module identity when spilling is disabled."""

        def make(value):
            @wp.kernel(enable_cuda_smem_spilling=value, module="unique")
            def kernel(a: wp.array[int]):
                a[wp.tid()] = 0

            return kernel

        default = make(None)
        disabled = make(False)
        enabled = make(True)

        self.assertEqual(default.options, disabled.options)
        self.assertEqual(default.module.hash_module(), disabled.module.hash_module())
        self.assertNotEqual(default.module.hash_module(), enabled.module.hash_module())

    def test_codegen_emits_smem_spilling(self):
        """Emit shared-memory spilling for every eligible CUDA entry point."""

        @wp.kernel(module="unique")
        def default_kernel(a: wp.array[int]):
            a[wp.tid()] = 0

        self.assertNotIn("WP_ENABLE_SMEM_SPILLING();", kernel_source(default_kernel, "cuda"))

        for grid_stride in (True, False):
            with self.subTest(grid_stride=grid_stride):

                @wp.kernel(
                    enable_cuda_smem_spilling=True,
                    grid_stride=grid_stride,
                    enable_backward=True,
                    module="unique",
                )
                def enabled(a: wp.array[float]):
                    a[wp.tid()] = 1.0

                source = kernel_source(enabled, "cuda")
                self.assertEqual(source.count("WP_ENABLE_SMEM_SPILLING();"), 2)
                self.assertNotIn("WP_ENABLE_SMEM_SPILLING();", kernel_source(enabled, "cpu"))
                self.assertNotIn("WP_ENABLE_SMEM_SPILLING();", kernel_source(enabled, "cuda", llvm_cuda=True))

    def test_dynamic_shared_memory_disables_smem_spilling(self):
        """Disable spilling when CUDA entry points use dynamic shared memory."""

        @wp.kernel(enable_cuda_smem_spilling=True, enable_backward=True, module="unique")
        def shared_kernel(a: wp.array[float]):
            tile = wp.tile_ones(shape=32, dtype=float, storage="shared")
            wp.tile_store(a, tile)

        source = kernel_source(shared_kernel, "cuda")
        self.assertGreater(shared_kernel.adj.get_total_required_shared(), 0)
        self.assertGreater(shared_kernel.adj.get_total_required_shared_backward(), 0)
        self.assertNotIn("WP_ENABLE_SMEM_SPILLING();", source)

    def test_backward_dynamic_shared_memory_disables_only_backward(self):
        """Disable spilling only for entry points that use dynamic shared memory."""
        tile_size = 4
        scratch_size = 8

        @wp.func
        def copy_tile(x: wp.array2d[float], y: wp.array2d[float], i: int):
            tile = wp.tile_load(x, shape=(tile_size, tile_size), offset=(i * tile_size, 0))
            wp.tile_store(y, tile, offset=(i * tile_size, 0))

        @wp.func_grad(copy_tile)
        def adj_copy_tile(x: wp.array2d[float], y: wp.array2d[float], i: int):
            grad = wp.tile_load(wp.adjoint[y], shape=(tile_size, tile_size), offset=(i * tile_size, 0))
            scratch = wp.tile_ones(shape=(scratch_size, scratch_size), dtype=float, storage="shared")
            pad = wp.tile_broadcast(wp.tile_sum(scratch), shape=(tile_size, tile_size))
            wp.tile_atomic_add(wp.adjoint[x], grad + pad * 0.0, offset=(i * tile_size, 0))

        @wp.kernel(enable_cuda_smem_spilling=True, module="unique")
        def kernel(x: wp.array2d[float], y: wp.array2d[float]):
            copy_tile(x, y, wp.tid())

        source = kernel_source(kernel, "cuda")
        name = kernel.get_mangled_name()
        forward_name = f"{name}_cuda_kernel_forward"
        backward_name = f"{name}_cuda_kernel_backward"
        forward_start = source.index(forward_name)
        backward_start = source.index(backward_name)
        forward_source = source[forward_start:backward_start]
        backward_source = source[backward_start:]

        self.assertEqual(kernel.adj.get_total_required_shared(), 0)
        self.assertGreater(kernel.adj.get_total_required_shared_backward(), 0)
        self.assertIn("WP_ENABLE_SMEM_SPILLING();", forward_source)
        self.assertNotIn("WP_ENABLE_SMEM_SPILLING();", backward_source)

    def test_cuda_version_and_debug_guards(self):
        """Guard shared-memory spilling by toolkit version and debug mode."""
        header = codegen.cuda_module_header
        self.assertIn("__CUDACC_VER_MAJOR__ >= 13", header)
        self.assertIn("!defined(_DEBUG)", header)
        self.assertNotIn("!defined(_DEBUG) || defined(_WIN32)", header)
        self.assertIn('#define WP_ENABLE_SMEM_SPILLING() asm volatile(".pragma \\"enable_smem_spilling\\";");', header)
        self.assertIn("#define WP_ENABLE_SMEM_SPILLING()\n#endif", header)


devices = get_test_devices()
add_function_test(
    TestCudaSmemSpilling,
    "test_cuda_smem_spilling_kernel_runs",
    test_cuda_smem_spilling_kernel_runs,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
