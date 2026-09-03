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


@wp.kernel(cuda_max_registers=64, enable_backward=False, module="unique")
def fill_index(a: wp.array[int]):
    a[wp.tid()] = wp.tid()


def test_cuda_max_registers_kernel_runs(test, device):
    """Run a kernel with a CUDA register limit."""
    values = wp.empty(128, dtype=int, device=device)
    wp.launch(fill_index, dim=values.shape, inputs=[values], device=device)
    assert_np_equal(values.numpy(), np.arange(128, dtype=np.int32))


class TestCudaMaxRegisters(unittest.TestCase):
    def test_invalid_cuda_max_registers_rejected(self):
        """Reject invalid CUDA maximum-register values."""
        for value, error in ((True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)):
            with self.subTest(value=value), self.assertRaisesRegex(error, "cuda_max_registers"):

                @wp.kernel(cuda_max_registers=value, module="unique")
                def invalid(a: wp.array[int]):
                    a[wp.tid()] = 0

    def test_launch_bounds_conflict_rejected(self):
        """Reject simultaneous CUDA register limits and launch bounds."""
        with self.assertRaisesRegex(ValueError, "cuda_max_registers and launch_bounds"):

            @wp.kernel(launch_bounds=128, cuda_max_registers=64, module="unique")
            def invalid(a: wp.array[int]):
                a[wp.tid()] = 0

    def test_codegen_emits_cuda_max_registers(self):
        """Emit CUDA register limits for every CUDA entry point."""

        @wp.kernel(module="unique")
        def default_kernel(a: wp.array[int]):
            a[wp.tid()] = 0

        self.assertNotIn("WP_MAXNREG(", kernel_source(default_kernel, "cuda"))

        for grid_stride in (True, False):
            with self.subTest(grid_stride=grid_stride):

                @wp.kernel(cuda_max_registers=64, grid_stride=grid_stride, enable_backward=True, module="unique")
                def limited(a: wp.array[float]):
                    a[wp.tid()] = 1.0

                source = kernel_source(limited, "cuda")
                self.assertEqual(source.count("WP_MAXNREG(64)"), 2)
                self.assertEqual(source.count("__global__ void WP_MAXNREG(64)"), 2)
                self.assertNotIn("WP_MAXNREG(", kernel_source(limited, "cpu"))
                self.assertNotIn("WP_MAXNREG(", kernel_source(limited, "cuda", llvm_cuda=True))

    def test_cuda_version_guard(self):
        """Guard CUDA register limits by CUDA Toolkit version."""
        header = codegen.cuda_module_header
        self.assertIn("__CUDACC_VER_MAJOR__", header)
        self.assertIn("__CUDACC_VER_MINOR__ >= 4", header)
        self.assertIn("#define WP_MAXNREG(n) __maxnreg__(n)", header)
        self.assertIn("#define WP_MAXNREG(n)\n#endif", header)


devices = get_test_devices()
add_function_test(
    TestCudaMaxRegisters,
    "test_cuda_max_registers_kernel_runs",
    test_cuda_max_registers_kernel_runs,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
