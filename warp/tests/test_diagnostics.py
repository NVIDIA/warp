# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.metadata
import io
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import warp as wp
from warp._src.context import (
    get_host_compiler_version,
    get_llvm_version,
    get_nanovdb_version,
    get_nvrtc_version,
)


class TestDiagnostics(unittest.TestCase):
    """Tests for wp.print_diagnostics() and related version query functions."""

    def get_print_diagnostics_output(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            info = wp.print_diagnostics()
        return info, output.getvalue()

    def test_print_diagnostics_returns_dict(self):
        info, _ = self.get_print_diagnostics_output()
        self.assertIsInstance(info, dict)

    def test_print_diagnostics_has_required_keys(self):
        info, _ = self.get_print_diagnostics_output()
        required_keys = [
            "warp_python",
            "warp_native",
            "warp_clang",
            "llvm",
            "numpy",
            "python",
            "platform",
            "cuda_enabled",
            "cuda_toolkit",
            "cuda_driver",
            "nvrtc",
            "cuda_compatibility",
            "mathdx_enabled",
            "libmathdx",
            "cubql_enabled",
            "nanovdb",
            "host_compiler",
            "debug",
            "verify_fp",
            "fast_math",
            "devices",
        ]
        for key in required_keys:
            self.assertIn(key, info, f"Missing key: {key}")

    def test_print_diagnostics_version_strings(self):
        info, _ = self.get_print_diagnostics_output()
        self.assertIsInstance(info["warp_python"], str)
        self.assertRegex(info["warp_python"], r"^\d+\.\d+\.\d+")
        self.assertIsInstance(info["warp_native"], str)
        self.assertRegex(info["warp_native"], r"^\d+\.\d+\.\d+")
        self.assertIsInstance(info["numpy"], str)
        self.assertIsInstance(info["python"], str)
        self.assertIsInstance(info["platform"], str)

    def test_print_diagnostics_build_flags(self):
        info, _ = self.get_print_diagnostics_output()
        self.assertIsInstance(info["debug"], bool)
        self.assertIsInstance(info["verify_fp"], bool)
        self.assertIsInstance(info["fast_math"], bool)

    def test_print_diagnostics_cubql_enabled(self):
        info, _ = self.get_print_diagnostics_output()
        self.assertIsInstance(info["cubql_enabled"], bool)

    def test_print_diagnostics_devices(self):
        info, _ = self.get_print_diagnostics_output()
        devices = info["devices"]
        self.assertIsInstance(devices, list)
        self.assertGreaterEqual(len(devices), 1)
        # CPU device should always be present
        self.assertEqual(devices[0]["alias"], "cpu")

    def test_print_diagnostics_optional_frameworks(self):
        """Verify that optional framework keys match installed packages."""
        info, _ = self.get_print_diagnostics_output()
        for pkg in ("torch", "jax", "jaxlib"):
            try:
                expected = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                self.assertNotIn(pkg, info, f"{pkg} not installed but key present")
            else:
                self.assertIn(pkg, info, f"{pkg} installed but key missing")
                self.assertEqual(info[pkg], expected)

    def test_print_diagnostics_cuda_device_structure(self):
        info, diagnostics_output = self.get_print_diagnostics_output()
        devices = info["devices"]
        if wp.is_cuda_available():
            self.assertGreaterEqual(len(devices), 2, "CUDA available but no CUDA device in list")
            cuda_dev = devices[1]
            warp_dev = wp.get_device(cuda_dev["alias"])
            self.assertRegex(cuda_dev["arch"], r"^sm_\d+$")
            self.assertIsInstance(cuda_dev["sm_count"], int)
            self.assertGreater(cuda_dev["sm_count"], 0)
            self.assertIsInstance(cuda_dev["memory_gb"], float)
            self.assertIsInstance(cuda_dev["mempool_enabled"], bool)
            self.assertRegex(cuda_dev["pci_bus_id"], r"^[0-9A-F]+:[0-9A-F]+:[0-9A-F]+$")
            for key in (
                "is_cpu_memory_access_from_gpu_supported",
                "is_gpu_memory_access_from_cpu_supported",
                "is_cpu_gpu_atomic_supported",
            ):
                self.assertIn(key, cuda_dev)
                self.assertIsInstance(cuda_dev[key], bool)
                self.assertEqual(cuda_dev[key], getattr(warp_dev, key))

            self.assertIn("GPU->CPU mem:", diagnostics_output)
            self.assertIn("CPU->GPU mem:", diagnostics_output)
            self.assertIn("CPU/GPU atomics:", diagnostics_output)

    def test_print_diagnostics_suppresses_init_banner_at_debug_log_level(self):
        script = textwrap.dedent(
            """
            import contextlib
            import io
            import warp as wp

            wp.config.log_level = wp.LOG_DEBUG
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                wp.print_diagnostics()
            assert wp.config.log_level == wp.LOG_DEBUG, wp.config.log_level
            print(output.getvalue().splitlines()[0])
            """
        )

        result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
        first_line = result.stdout.strip()
        self.assertEqual(first_line, "Software")

    def test_nanovdb_version(self):
        version = get_nanovdb_version()
        self.assertIsInstance(version, str)
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")

    def test_host_compiler_version(self):
        version = get_host_compiler_version()
        self.assertIsInstance(version, str)
        self.assertNotEqual(version, "unknown")

    def test_llvm_version(self):
        version = get_llvm_version()
        self.assertIsInstance(version, str)
        # LLVM should be available since warp-clang is loaded
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")

    def test_nvrtc_version(self):
        version = get_nvrtc_version()
        # NVRTC may be statically linked even without a GPU present at runtime,
        # so we only validate the shape when a version is returned.
        if version is not None:
            self.assertIsInstance(version, tuple)
            self.assertEqual(len(version), 2)
            self.assertGreater(version[0], 0)

    def test_libmathdx_version(self):
        """Verify the public libmathdx version tuple and its consistency with diagnostics."""
        version = wp.get_libmathdx_version()
        if version is not None:
            self.assertIsInstance(version, tuple)
            self.assertEqual(len(version), 3)
            for component in version:
                self.assertIsInstance(component, int)
                self.assertGreaterEqual(component, 0)
        info, _ = self.get_print_diagnostics_output()
        self.assertEqual(info["libmathdx"], ".".join(map(str, version)) if version is not None else None)

    def test_libmathdx_version_conversion(self):
        """Verify conversion of libmathdx version strings to an integer tuple or None."""
        for native_version, expected in (("0.3.10", (0, 3, 10)), ("", None)):
            with self.subTest(native_version=native_version):
                with patch("warp._src.context.Runtime.get_libmathdx_version", return_value=native_version):
                    self.assertEqual(wp.get_libmathdx_version(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
