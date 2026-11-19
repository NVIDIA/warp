# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import unittest

import warp as wp
from warp._src.codegen import _codegen_lock


class TestContext(unittest.TestCase):
    def test_cuda_13_rejects_unsupported_device_arch(self):
        for arch in (61, 74):
            with self.subTest(arch=arch):
                with self.assertRaises(RuntimeError) as error:
                    wp._src.context._validate_cuda_device_arch(arch, (13, 0), "cuda:0")

                message = str(error.exception)
                for detail in ("CUDA 13", f"sm_{arch}", "sm_75", "CUDA 12 build", "cuda:0"):
                    self.assertIn(detail, message)

    def test_cuda_13_accepts_minimum_device_arch(self):
        wp._src.context._validate_cuda_device_arch(75, (13, 0))

    def test_cuda_12_accepts_older_device_arch(self):
        wp._src.context._validate_cuda_device_arch(52, (12, 9))

    def test_unknown_cuda_toolkit_accepts_device_arch(self):
        wp._src.context._validate_cuda_device_arch(52, None)

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

    def test_static_kernel_argument_signature_and_sharing(self):
        def scale(values: wp.array[wp.float32], output: wp.array[wp.float32], factor: wp.float32):
            i = wp.tid()
            output[i] = values[i] * factor

        def offset_vec(values: wp.array[wp.vec3], offset: wp.vec3, scale_f64: wp.float64):
            i = wp.tid()
            values[i] = values[i] + offset * float(scale_f64)

        def generic_add(values: wp.array[wp.Scalar], output: wp.array[wp.Scalar], offset: float):
            i = wp.tid()
            output[i] = values[i] + offset

        module = wp.Module("test_static_kernel_argument_signature_and_sharing", None)
        static_args = {"factor": 0.5}
        k_half = wp.Kernel(scale, module=module, static_args=static_args)
        static_args["factor"] = 2.0
        k_two = wp.Kernel(scale, module=module, static_args={"factor": 2.0})

        self.assertEqual(list(k_half.adj.arg_types), ["values", "output"])
        self.assertEqual([arg.label for arg in k_half.adj.args], ["values", "output"])
        self.assertEqual(k_half.arg_indices, {"values": 0, "output": 1})
        self.assertEqual(k_half.adj.static_args, {"factor": 0.5})
        self.assertEqual(k_two.adj.static_args, {"factor": 2.0})
        self.assertIsNotNone(k_half.adj._shared_source)
        self.assertIs(k_half.adj._shared_source, k_two.adj._shared_source)

        vec = wp.vec3(1.0, 2.0, 3.0)
        f64 = wp.float64(0.25)
        k_val = wp.Kernel(offset_vec, module=module, static_args={"offset": vec, "scale_f64": f64})
        vec[0] = 99.0
        f64.value = 99.0
        stored = k_val.adj.static_args
        self.assertEqual(tuple(stored["offset"]), (1.0, 2.0, 3.0))
        self.assertEqual(stored["scale_f64"].value, 0.25)

        k_gen = wp.Kernel(generic_add, module=module, static_args={"offset": 2.0})
        overload = k_gen.add_overload([wp.array[wp.float32], wp.array[wp.float32]])
        self.assertEqual(overload.adj.static_args, {"offset": 2.0})
        self.assertEqual([arg.label for arg in overload.adj.args], ["values", "output"])

    def test_static_kernel_argument_validation(self):
        def scale(values: wp.array[wp.float32], factor: wp.float32):
            i = wp.tid()
            values[i] = values[i] * factor

        module = wp.Module("test_static_kernel_argument_validation", None)

        with self.assertRaisesRegex(ValueError, "no arguments.*missing"):
            wp.Kernel(scale, module=module, static_args={"missing": 1.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
