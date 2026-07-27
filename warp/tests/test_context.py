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

    def test_kernel_mangled_name_cache(self):
        """Verify that mangled kernel names are cached and invalidated when their inputs change."""

        def mangled_name_kernel():
            pass

        module = wp.Module("test_kernel_mangled_name_cache", None)
        kernel = wp.Kernel(mangled_name_kernel, module=module)

        kernel.hash = bytes.fromhex("01234567" + "00" * 28)
        first_name = kernel.get_mangled_name()
        self.assertEqual(first_name, f"{kernel.key}_01234567")
        self.assertIs(kernel.get_mangled_name(), first_name)

        kernel.hash = bytes.fromhex("89abcdef" + "00" * 28)
        second_name = kernel.get_mangled_name()
        self.assertEqual(second_name, f"{kernel.key}_89abcdef")
        self.assertIsNot(second_name, first_name)
        self.assertIs(kernel.get_mangled_name(), second_name)

        # Hash changes invalidate names individually, so mark_modified() must remain O(1).
        module.mark_modified()
        self.assertIs(kernel._mangled_name, second_name)

        module.execs["sentinel"] = object()
        module._set_strip_hash(True)
        self.assertEqual(module.execs, {})
        self.assertIs(kernel.get_mangled_name(), kernel.key)

        module.execs["sentinel"] = object()
        module._set_strip_hash(False)
        self.assertEqual(module.execs, {})
        self.assertEqual(kernel.get_mangled_name(), second_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
