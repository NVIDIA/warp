# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for kernel compilation and linking configuration."""

import unittest

import numpy as np

import warp as wp


def _make_arange_kernel():
    """Create a fresh unique-module kernel so each call gets its own Module."""

    @wp.kernel(module="unique")
    def arange(a: wp.array(dtype=float)):
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


if __name__ == "__main__":
    unittest.main()
