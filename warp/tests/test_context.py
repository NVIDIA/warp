# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import unittest

import warp as wp
from warp._src.codegen import _codegen_lock


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

    def test_kernel_mangled_name_cache_is_synchronized(self):
        """Prevent a hash update from leaving a stale mangled-name cache."""
        reader_has_hash = threading.Event()
        resume_reader = threading.Event()
        reader_errors = []

        # CPython may switch threads after the real ``bytes.hex()`` call.
        # Waiting inside an override widens that legal handoff point deterministically.
        class PausingHash(bytes):
            def hex(self, *args, **kwargs):
                reader_has_hash.set()
                if not resume_reader.wait(timeout=5.0):
                    raise TimeoutError("Timed out waiting to resume the mangled-name reader.")
                return super().hex(*args, **kwargs)

        def mangled_name_kernel():
            pass

        module = wp.Module("test_kernel_mangled_name_cache_is_synchronized", None)
        kernel = wp.Kernel(mangled_name_kernel, module=module)
        old_hash = PausingHash(bytes.fromhex("01234567" + "00" * 28))
        new_hash = bytes.fromhex("89abcdef" + "00" * 28)
        kernel.hash = old_hash

        def read_mangled_name():
            try:
                kernel.get_mangled_name()
            except Exception as error:
                reader_errors.append(error)

        reader = threading.Thread(target=read_mangled_name)
        reader.start()
        self.assertTrue(reader_has_hash.wait(timeout=5.0), "Mangled-name reader did not capture the old hash.")

        # Reproduce ModuleBuilder's critical section. If the reader is not
        # synchronized on this lock, update the hash while it is paused and it
        # will subsequently cache the old name over the new hash.
        acquired = _codegen_lock.acquire(blocking=False)
        if acquired:
            try:
                kernel.hash = new_hash
            finally:
                _codegen_lock.release()

        resume_reader.set()
        reader.join(timeout=5.0)
        self.assertFalse(reader.is_alive(), "Mangled-name reader did not finish.")

        if not acquired:
            kernel.hash = new_hash

        if reader_errors:
            raise reader_errors[0]

        self.assertEqual(kernel.hash, new_hash)
        self.assertEqual(kernel.get_mangled_name(), f"{kernel.key}_89abcdef")


if __name__ == "__main__":
    unittest.main(verbosity=2)
