# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp


class TestContext(unittest.TestCase):
    def test_context_type_str(self):
        self.assertEqual(wp._src.context.type_str(list[int]), "list[int]")
        self.assertEqual(wp._src.context.type_str(list[float]), "list[float]")

        self.assertEqual(wp._src.context.type_str(tuple[int]), "tuple[int]")
        self.assertEqual(wp._src.context.type_str(tuple[float]), "tuple[float]")
        self.assertEqual(wp._src.context.type_str(tuple[int, float]), "tuple[int, float]")
        self.assertEqual(wp._src.context.type_str(tuple[int, ...]), "tuple[int, ...]")

    def test_kernel_name_override(self):
        """@wp.kernel(name=...) should override the kernel's key."""

        @wp.kernel(name="custom_kernel_name")
        def some_kernel(x: wp.array(dtype=float)):
            i = wp.tid()
            x[i] = x[i]

        self.assertEqual(some_kernel.key, "custom_kernel_name")

    def test_kernel_name_default(self):
        """Omitting name should fall back to the qualified name."""

        @wp.kernel
        def default_named_kernel(x: wp.array(dtype=float)):
            i = wp.tid()
            x[i] = x[i]

        self.assertTrue(default_named_kernel.key.endswith("default_named_kernel"))

    def test_kernel_name_invalid(self):
        """A name that is not a valid C identifier should raise ValueError."""
        with self.assertRaises(ValueError) as cm:

            @wp.kernel(name="has space")
            def bad_kernel(x: wp.array(dtype=float)):
                i = wp.tid()
                x[i] = x[i]

        self.assertIn("not a valid C identifier", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
