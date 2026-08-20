# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for kernel compilation and linking configuration."""

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


if __name__ == "__main__":
    unittest.main()
