# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.metadata
import unittest

import warp as wp
from warp._src.context import (
    get_host_compiler_version,
    get_libmathdx_version,
    get_llvm_version,
    get_nanovdb_version,
    get_nvrtc_version,
)


class TestDiagnostics(unittest.TestCase):
    """Tests for wp.print_diagnostics() and related version query functions."""

    def test_print_diagnostics_returns_dict(self):
        info = wp.print_diagnostics()
        self.assertIsInstance(info, dict)

    def test_print_diagnostics_has_required_keys(self):
        info = wp.print_diagnostics()
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
        info = wp.print_diagnostics()
        self.assertIsInstance(info["warp_python"], str)
        self.assertRegex(info["warp_python"], r"^\d+\.\d+\.\d+")
        self.assertIsInstance(info["warp_native"], str)
        self.assertRegex(info["warp_native"], r"^\d+\.\d+\.\d+")
        self.assertIsInstance(info["numpy"], str)
        self.assertIsInstance(info["python"], str)
        self.assertIsInstance(info["platform"], str)

    def test_print_diagnostics_build_flags(self):
        info = wp.print_diagnostics()
        self.assertIsInstance(info["debug"], bool)
        self.assertIsInstance(info["verify_fp"], bool)
        self.assertIsInstance(info["fast_math"], bool)

    def test_print_diagnostics_devices(self):
        info = wp.print_diagnostics()
        devices = info["devices"]
        self.assertIsInstance(devices, list)
        self.assertGreaterEqual(len(devices), 1)
        # CPU device should always be present
        self.assertEqual(devices[0]["alias"], "cpu")

    def test_print_diagnostics_optional_frameworks(self):
        """Optional framework keys match installed packages."""
        info = wp.print_diagnostics()
        for pkg in ("torch", "jax", "jaxlib"):
            try:
                expected = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                self.assertNotIn(pkg, info, f"{pkg} not installed but key present")
            else:
                self.assertIn(pkg, info, f"{pkg} installed but key missing")
                self.assertEqual(info[pkg], expected)

    def test_print_diagnostics_cuda_device_structure(self):
        info = wp.print_diagnostics()
        devices = info["devices"]
        if wp.is_cuda_available():
            self.assertGreaterEqual(len(devices), 2, "CUDA available but no CUDA device in list")
            cuda_dev = devices[1]
            self.assertRegex(cuda_dev["arch"], r"^sm_\d+$")
            self.assertIsInstance(cuda_dev["sm_count"], int)
            self.assertGreater(cuda_dev["sm_count"], 0)
            self.assertIsInstance(cuda_dev["memory_gb"], float)
            self.assertIsInstance(cuda_dev["mempool_enabled"], bool)
            self.assertRegex(cuda_dev["pci_bus_id"], r"^[0-9A-F]+:[0-9A-F]+:[0-9A-F]+$")

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
        version = get_libmathdx_version()
        self.assertIsInstance(version, str)
        # If MathDx is enabled, version should be a valid version string
        if version:
            self.assertRegex(version, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
