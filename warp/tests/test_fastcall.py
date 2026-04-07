# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
import unittest

import warp as wp

wp.init()


class TestFastcallAvailable(unittest.TestCase):
    """Verify the _warp_fastcall module loaded successfully.

    The fastcall module is expected to be available on all supported platforms.
    A ctypes fallback exists for unusual setups, but a failure here should be
    investigated -- it likely indicates a build or packaging issue.
    """

    def test_module_loads(self):
        self.assertIsNotNone(wp._src.context.runtime.fastcall, "_warp_fastcall module failed to load")


# Skip remaining tests if the module didn't load -- a ctypes fallback keeps
# Warp functional, but these tests exercise the fastcall path specifically.
@unittest.skipIf(wp._src.context.runtime.fastcall is None, "_warp_fastcall not available")
class TestFastcall(unittest.TestCase):
    """Tests for the _warp_fastcall METH_FASTCALL extension module.

    These tests verify the call chain (importlib loading, METH_FASTCALL dispatch,
    argument marshalling, return value wrapping) and that the fastcall methods
    override the ctypes ones on runtime.core. The underlying native functions
    are already exercised by test_fp16, test_types, test_scalar_ops, etc.
    """

    def test_core_methods_are_overridden(self):
        """Verify that runtime.core.wp_* methods are the fastcall versions, not ctypes."""
        core = wp._src.context.runtime.core
        # Fastcall methods are builtin_function_or_method, ctypes ones are _FuncPtr instances.
        self.assertIsInstance(core.wp_float_to_half_bits, types.BuiltinFunctionType)
        self.assertIsInstance(core.wp_half_bits_to_float, types.BuiltinFunctionType)

    def test_float_to_half_bits(self):
        result = wp._src.context.runtime.core.wp_float_to_half_bits(1.0)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0x3C00)

    def test_half_bits_to_float(self):
        result = wp._src.context.runtime.core.wp_half_bits_to_float(0x3C00)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 1.0)

    def test_round_trip(self):
        core = wp._src.context.runtime.core
        bits = core.wp_float_to_half_bits(1.0)
        self.assertEqual(core.wp_half_bits_to_float(bits), 1.0)

    def test_consistency_with_ctypes(self):
        """Verify all METH_FASTCALL results match the original ctypes path."""
        core = wp._src.context.runtime.core
        for v in [0.0, 1.0, -1.0, 3.14]:
            bits = core.wp_float_to_half_bits(v)
            self.assertEqual(bits, core.ctypes.wp_float_to_half_bits(v))
            self.assertEqual(core.wp_half_bits_to_float(bits), core.ctypes.wp_half_bits_to_float(bits))

    def test_wrong_arg_count(self):
        core = wp._src.context.runtime.core
        with self.assertRaises(TypeError):
            core.wp_float_to_half_bits()
        with self.assertRaises(TypeError):
            core.wp_float_to_half_bits(1.0, 2.0)
        with self.assertRaises(TypeError):
            core.wp_half_bits_to_float()

    def test_wrong_arg_type(self):
        core = wp._src.context.runtime.core
        with self.assertRaises(TypeError):
            core.wp_float_to_half_bits("not a float")

    def test_vec3h_construction(self):
        """Verify the fastcall path works through the public API (wp.vec3h)."""
        v = wp.vec3h(1.0, 2.0, 3.0)
        self.assertEqual(float(v[0]), 1.0)
        self.assertEqual(float(v[1]), 2.0)
        self.assertEqual(float(v[2]), 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
