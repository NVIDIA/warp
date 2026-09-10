# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for kernel compilation and linking configuration."""

import ctypes
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import warp as wp
from warp._src import build as _build_module


def _make_arange_kernel():
    """Create a fresh unique-module kernel so each call gets its own Module."""

    @wp.kernel(module="unique")
    def arange(a: wp.array[float]):
        tid = wp.tid()
        a[tid] = float(tid) * 2.0

    return arange


def _run_and_check(test):
    """Compile, launch, and verify a fresh arange kernel on the CPU."""
    kernel = _make_arange_kernel()
    a = wp.zeros(10, dtype=float, device="cpu")
    wp.launch(kernel, dim=10, inputs=[a], device="cpu")
    expected = np.arange(10, dtype=np.float32) * 2.0
    np.testing.assert_allclose(a.numpy(), expected)


@wp.struct
class _LargeKernelArgument:
    v1: wp.vec3
    v2: wp.vec3
    v3: wp.vec3
    m1: wp.mat22
    m2: wp.mat22
    m3: wp.mat22
    m4: wp.mat22
    m5: wp.mat22
    m6: wp.mat22


def _make_debug_warmup_kernel():
    """Create a fresh CUDA kernel compiled in debug mode."""

    @wp.kernel(module="unique", module_options={"mode": "debug"})
    def debug_warmup(output: wp.array[int]):
        output[0] = 1

    return debug_warmup


def _make_large_argument_kernel():
    """Create a fresh release kernel with a large by-value argument."""

    @wp.kernel(module="unique", module_options={"mode": "release"})
    def read_large_argument(value: _LargeKernelArgument, output: wp.array[float]):
        output[0] = value.v1[0]
        output[1] = value.v2[0]
        output[2] = value.v3[0]
        output[3] = value.m1[0, 0]
        output[4] = value.m2[0, 0]
        output[5] = value.m3[0, 0]
        output[6] = value.m4[0, 0]
        output[7] = value.m5[0, 0]
        output[8] = value.m6[0, 0]

    return read_large_argument


class TestCompilation(unittest.TestCase):
    """Tests for kernel compilation and linking behavior."""

    def test_default_linker(self):
        """Verify that the default JITLink linker compiles and runs a CPU kernel."""
        _run_and_check(self)

    def test_legacy_cpu_linker(self):
        """Verify that the legacy RTDyld linker compiles and runs a CPU kernel."""
        old_val = wp.config.legacy_cpu_linker
        try:
            wp.config.legacy_cpu_linker = True
            _run_and_check(self)
        finally:
            wp.config.legacy_cpu_linker = old_val

    def test_linker_roundtrip(self):
        """Verify that switching back to the default linker after using legacy works."""
        old_val = wp.config.legacy_cpu_linker
        try:
            wp.config.legacy_cpu_linker = True
            _run_and_check(self)
            wp.config.legacy_cpu_linker = False
            _run_and_check(self)
        finally:
            wp.config.legacy_cpu_linker = old_val

    def test_cpu_build_passes_extra_include_dirs_to_clang(self):
        original_runtime = _build_module.warp._src.context.runtime
        captured = {}

        def compile_cpp(*args):
            flags = []
            flag = args[8][0]
            index = 0
            while flag is not None:
                flags.append(flag.decode("utf-8"))
                index += 1
                flag = args[8][index]
            captured["flags"] = flags
            return 0

        try:
            _build_module.warp._src.context.runtime = SimpleNamespace(llvm=SimpleNamespace(wp_compile_cpp=compile_cpp))
            with tempfile.TemporaryDirectory() as tmpdir:
                include_dir = Path(tmpdir) / "include"
                include_dir.mkdir()
                cpp_path = Path(tmpdir) / "kernel.cpp"
                obj_path = Path(tmpdir) / "kernel.o"
                cpp_path.write_text("// empty test kernel\n")

                _build_module.build_cpu(str(obj_path), str(cpp_path), extra_include_dirs=[include_dir])
        finally:
            _build_module.warp._src.context.runtime = original_runtime

        self.assertIn("-I", captured["flags"])
        self.assertIn(str(include_dir.resolve()), captured["flags"])

    def test_llvm_cuda_build_passes_extra_include_dirs_to_clang(self):
        original_runtime = _build_module.warp._src.context.runtime
        captured = {}

        def compile_cuda(*args):
            num_include_dirs = args[3]
            include_dirs = args[4]
            captured["include_dirs"] = [include_dirs[i].decode("utf-8") for i in range(num_include_dirs)]
            return 0

        try:
            _build_module.warp._src.context.runtime = SimpleNamespace(
                llvm=SimpleNamespace(wp_compile_cuda=compile_cuda)
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                include_dir = Path(tmpdir) / "include"
                include_dir.mkdir()
                cu_path = Path(tmpdir) / "kernel.cu"
                output_path = Path(tmpdir) / "kernel.ptx"
                cu_path.write_text("// empty test kernel\n")

                _build_module.build_cuda(
                    str(cu_path),
                    80,
                    str(output_path),
                    pch_dir=None,
                    llvm_cuda=True,
                    extra_include_dirs=[include_dir],
                )
        finally:
            _build_module.warp._src.context.runtime = original_runtime

        self.assertEqual(captured["include_dirs"], [str(include_dir.resolve())])

    def test_cuda_debug_compile_does_not_corrupt_release(self):
        """Compile a CUDA release kernel safely after a debug kernel."""

        if wp.config.llvm_cuda:
            self.skipTest("NVRTC is not the configured CUDA compiler")

        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("No CUDA devices available")

        cache_kernels = wp.config.cache_kernels
        try:
            wp.config.cache_kernels = False

            device = devices[0]
            warmup_output = wp.zeros(1, dtype=int, device=device)
            wp.launch(_make_debug_warmup_kernel(), dim=1, outputs=[warmup_output], device=device)
            np.testing.assert_array_equal(warmup_output.numpy(), [1])

            value = _LargeKernelArgument()
            self.assertGreater(ctypes.sizeof(_LargeKernelArgument.ctype), 128)
            value.v1 = wp.vec3(1.0)
            value.v2 = wp.vec3(2.0)
            value.v3 = wp.vec3(3.0)
            value.m1 = wp.mat22(4.0)
            value.m2 = wp.mat22(5.0)
            value.m3 = wp.mat22(6.0)
            value.m4 = wp.mat22(7.0)
            value.m5 = wp.mat22(8.0)
            value.m6 = wp.mat22(9.0)

            output = wp.zeros(9, dtype=float, device=device)
            wp.launch(_make_large_argument_kernel(), dim=1, inputs=[value], outputs=[output], device=device)
            np.testing.assert_allclose(output.numpy(), np.arange(1.0, 10.0, dtype=np.float32))
        finally:
            wp.config.cache_kernels = cache_kernels


if __name__ == "__main__":
    unittest.main()
