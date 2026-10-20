# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for loading APIC captures from in-memory bytes."""

import contextlib
import ctypes
import gc
import os
import re
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np

import warp as wp
import warp._src.context as wp_context
from warp._src.apic.types import APICModuleBytes
from warp.tests.unittest_utils import (
    add_function_test,
    get_test_devices,
    get_test_devices_with_cuda_graph_module_load,
    run_python_subprocess,
)

wp.init()

N = 16


@wp.kernel
def scale_kernel(input: wp.array[float], output: wp.array[float], s: float):
    i = wp.tid()
    output[i] = input[i] * s


def _save_scale_capture(tmpdir, device, kernel=scale_kernel):
    """Save a one-launch capture of ``kernel`` and return its path."""
    a = wp.array(np.arange(N, dtype=np.float32), device=device)
    b = wp.zeros(N, dtype=float, device=device)
    wp.load_module(kernel.module, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(kernel, dim=N, inputs=[a, b, 2.0], device=device)
    path = os.path.join(tmpdir, "scale")
    wp.capture_save(capture.graph, path, inputs={"a": a}, outputs={"b": b})
    return path


def _read_capture(path):
    """Return a saved capture's ``.wrp`` bytes and its module files keyed by filename."""
    with open(path + ".wrp", "rb") as wrp_file:
        wrp = wrp_file.read()
    modules = {}
    modules_dir = path + "_modules"
    if os.path.isdir(modules_dir):
        for filename in sorted(os.listdir(modules_dir)):
            with open(os.path.join(modules_dir, filename), "rb") as module_file:
                modules[filename] = module_file.read()
    return wrp, modules


@contextlib.contextmanager
def _native_errors_silenced():
    """Keep expected native loader errors out of the test output."""
    core = wp_context.runtime.core
    saved = core.wp_is_error_output_enabled()
    core.wp_set_error_output_enabled(False)
    try:
        yield
    finally:
        core.wp_set_error_output_enabled(saved)


def _native_load(wrp, entries, device):
    """Load ``(filename, data)`` entries, allowing ``None`` fields for invalid inputs.

    Returns the native graph pointer, or ``None`` on failure.
    """
    device = wp.get_device(device)
    keep_alive = []
    native_entries = (APICModuleBytes * max(len(entries), 1))()
    for native_entry, (filename, data) in zip(native_entries, entries, strict=False):
        if filename is not None:
            native_entry.filename = filename.encode("utf-8")
        if data is None:
            native_entry.size = 1
        elif data:
            buffer = ctypes.create_string_buffer(data, len(data))
            keep_alive.append(buffer)
            native_entry.data = ctypes.addressof(buffer)
            native_entry.size = len(data)
    wrp_buffer = ctypes.create_string_buffer(wrp, len(wrp))
    return wp_context.runtime.core.wp_apic_load_graph_from_bytes(
        device.context if device.is_cuda else None,
        ctypes.addressof(wrp_buffer),
        len(wrp),
        native_entries,
        len(entries),
        1 if device.is_cpu else 0,
    )


@wp.kernel(module="unique", module_options={"cuda_output": "ptx"})
def scale_kernel_ptx(input: wp.array[float], output: wp.array[float], s: float):
    i = wp.tid()
    output[i] = input[i] * s


@wp.kernel(module="unique")
def offset_kernel(values: wp.array[float], offset: float):
    i = wp.tid()
    values[i] = values[i] + offset


def _run_scale_graph(graph, values, device):
    """Set the ``a`` input of a loaded scale graph, launch it, and return its ``b`` output."""
    graph.set_param("a", wp.array(values, dtype=float, device=device))
    wp.capture_launch(graph)
    out = wp.zeros(len(values), dtype=float, device=device)
    graph.get_param("b", out)
    return out.numpy()


def test_load_from_bytes_matches_disk_load(test, device):
    """Compare disk and bytes loads and verify independent CPU module lifetimes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
        from_disk = wp.capture_load(path, device=device)
        from_bytes = wp.capture_load_from_bytes(wrp, modules, device=device)

    test.assertTrue(from_bytes.is_loaded)
    test.assertEqual(from_bytes.params, from_disk.params)

    values = np.full(N, 3.0, dtype=np.float32)
    disk_result = _run_scale_graph(from_disk, values, device)
    del from_disk
    gc.collect()
    bytes_result = _run_scale_graph(from_bytes, values, device)

    np.testing.assert_allclose(disk_result, values * 2.0)
    np.testing.assert_allclose(bytes_result, values * 2.0)


def test_load_from_bytes_releases_buffers(test, device):
    """Load after deleting capture files and replay after releasing source buffers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
    test.assertFalse(os.path.exists(path + ".wrp"))
    test.assertFalse(os.path.exists(path + "_modules"))

    wrp_buffer = bytearray(wrp)
    module_buffers = {name: bytearray(data) for name, data in modules.items()}
    graph = wp.capture_load_from_bytes(wrp_buffer, module_buffers, device=device)

    wrp_buffer[:] = bytes(len(wrp_buffer))
    for buffer in module_buffers.values():
        buffer[:] = bytes(len(buffer))
    del wrp_buffer, module_buffers, wrp, modules
    gc.collect()

    values = np.arange(N, dtype=np.float32)
    np.testing.assert_allclose(_run_scale_graph(graph, values, device), values * 2.0)


def test_load_from_bytes_buffer_types(test, device):
    """Load and replay from ``memoryview`` and NumPy buffers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)

    converters = {
        "memoryview": memoryview,
        "numpy": lambda data: np.frombuffer(data, dtype=np.uint8),
    }
    values = np.arange(N, dtype=np.float32)
    for kind, convert in converters.items():
        with test.subTest(kind=kind):
            graph = wp.capture_load_from_bytes(
                convert(wrp), {name: convert(data) for name, data in modules.items()}, device=device
            )
            np.testing.assert_allclose(_run_scale_graph(graph, values, device), values * 2.0)


def test_load_from_bytes_ptx_module(test, device):
    """Load PTX file bytes without a trailing NUL; the loader must append it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_scale_capture(tmpdir, device, kernel=scale_kernel_ptx)
        wrp, modules = _read_capture(path)
    test.assertTrue(modules)
    test.assertTrue(all(name.endswith(".ptx") for name in modules), list(modules))

    graph = wp.capture_load_from_bytes(wrp, modules, device=device)
    values = np.arange(N, dtype=np.float32)
    np.testing.assert_allclose(_run_scale_graph(graph, values, device), values * 2.0)


def test_load_from_bytes_multiple_modules(test, device):
    """Resolve each recorded module by filename and ignore extra entries."""
    a = wp.array(np.arange(N, dtype=np.float32), device=device)
    b = wp.zeros(N, dtype=float, device=device)
    wp.load_module(scale_kernel.module, device=device)
    wp.load_module(offset_kernel.module, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=N, inputs=[a, b, 2.0], device=device)
        wp.launch(offset_kernel, dim=N, inputs=[b, 1.0], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "two_modules")
        wp.capture_save(capture.graph, path, inputs={"a": a}, outputs={"b": b})
        wrp, modules = _read_capture(path)
    test.assertEqual(len(modules), 2)
    modules["wp_unused_module.o"] = b"not a module"

    graph = wp.capture_load_from_bytes(wrp, modules, device=device)
    values = np.arange(N, dtype=np.float32)
    np.testing.assert_allclose(_run_scale_graph(graph, values, device), values * 2.0 + 1.0)


def test_load_from_bytes_missing_module(test, device):
    """Fail with the missing filename when a recorded module is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
    ((filename, data),) = modules.items()

    cases = {
        "omitted": None,
        "empty": {},
        "directory prefix": {f"scale_modules/{filename}": data},
    }
    for case, provided in cases.items():
        with test.subTest(case=case):
            with _native_errors_silenced():
                with test.assertRaisesRegex(RuntimeError, re.escape(filename)):
                    wp.capture_load_from_bytes(wrp, provided, device=device)


def test_load_from_bytes_bad_module(test, device):
    """Reject corrupt or empty modules, then load the capture with intact bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
    for case in ("corrupt", "empty"):
        invalid = {name: b"\xff" * len(data) if case == "corrupt" else b"" for name, data in modules.items()}
        with test.subTest(case=case):
            with _native_errors_silenced():
                with test.assertRaises(RuntimeError):
                    wp.capture_load_from_bytes(wrp, invalid, device=device)

    graph = wp.capture_load_from_bytes(wrp, modules, device=device)
    values = np.arange(N, dtype=np.float32)
    np.testing.assert_allclose(_run_scale_graph(graph, values, device), values * 2.0)


def test_load_from_bytes_empty_native_module(test, device):
    """Reject an empty CUDA module supplied through a null native data pointer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
    ((filename, _),) = modules.items()

    with _native_errors_silenced():
        native_graph = _native_load(wrp, [(filename, b"")], device)
    try:
        test.assertFalse(native_graph)
    finally:
        if native_graph:
            wp_context.runtime.core.wp_apic_destroy_graph(native_graph)


_WRP_HEADER = struct.Struct("<4sIIIQ")  # magic, version, flags, section count, section-table offset
_WRP_SECTION_ENTRY = struct.Struct("<IIQQQ")  # type, flags, offset, stored size, uncompressed size
_WRP_HEADER_FIELDS = ("magic", "version", "flags", "num_sections", "section_table_offset")


def _find_section(wrp, section_type):
    """Return the offset and size of one section in a ``.wrp`` image."""
    _, _, _, section_count, table_offset = _WRP_HEADER.unpack_from(wrp)
    for index in range(section_count):
        entry_type, _, offset, size, _ = _WRP_SECTION_ENTRY.unpack_from(
            wrp, table_offset + index * _WRP_SECTION_ENTRY.size
        )
        if entry_type == section_type:
            return offset, size
    raise ValueError(f"No WRP section of type {section_type}")


def _patch_header(wrp, **fields):
    """Return a copy of a ``.wrp`` image with selected header fields replaced."""
    header = dict(zip(_WRP_HEADER_FIELDS, _WRP_HEADER.unpack_from(wrp), strict=True))
    header.update(fields)
    patched = bytearray(wrp)
    _WRP_HEADER.pack_into(patched, 0, *(header[name] for name in _WRP_HEADER_FIELDS))
    return bytes(patched)


def _patch_section_size(wrp, section_type, size):
    """Return a copy of a ``.wrp`` image with one section's stored size replaced."""
    _, _, _, section_count, table_offset = _WRP_HEADER.unpack_from(wrp)
    patched = bytearray(wrp)
    for index in range(section_count):
        entry_offset = table_offset + index * _WRP_SECTION_ENTRY.size
        entry = list(_WRP_SECTION_ENTRY.unpack_from(wrp, entry_offset))
        if entry[0] == section_type:
            entry[3] = size
            _WRP_SECTION_ENTRY.pack_into(patched, entry_offset, *entry)
            return bytes(patched)
    raise ValueError(f"No WRP section of type {section_type}")


def test_load_from_bytes_without_kernels(test, device):
    """Load a capture that launches no kernels without passing modules."""
    src = wp.array(np.arange(N, dtype=np.float32), device=device)
    dst = wp.zeros(N, dtype=float, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.copy(dst, src)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "copy_only")
        wp.capture_save(capture.graph, path, inputs={"src": src}, outputs={"dst": dst})
        wrp, _ = _read_capture(path)

    graph = wp.capture_load_from_bytes(wrp, device=device)
    graph.set_param("src", wp.array(np.full(N, 5.0, dtype=np.float32), device=device))
    wp.capture_launch(graph)
    out = wp.zeros(N, dtype=float, device=device)
    graph.get_param("dst", out)
    np.testing.assert_allclose(out.numpy(), np.full(N, 5.0, dtype=np.float32))


class TestApicLoadFromBytes(unittest.TestCase):
    @unittest.skipUnless(wp.is_cpu_available(), "Requires CPU support")
    def test_load_missing_cpu_forward_symbol(self):
        """Reject unresolved CPU kernels in both loaders and allow a later valid load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_scale_capture(tmpdir, "cpu")
            wrp, modules = _read_capture(path)
            graph = wp.capture_load(path, device="cpu")
            core = wp_context.runtime.core
            forward_name = core.wp_apic_get_kernel_forward_name(graph._native_graph, 0)
            kernel_key = core.wp_apic_get_kernel_key(graph._native_graph, 0).decode("utf-8")
            filename = core.wp_apic_get_kernel_module_binary_filename(graph._native_graph, 0).decode("utf-8")
            invalid = wrp.replace(forward_name, b"x" * len(forward_name))
            with open(path + ".wrp", "wb") as wrp_file:
                wrp_file.write(invalid)

            for source in ("file", "bytes"):
                with self.subTest(source=source):
                    with self.assertRaisesRegex(RuntimeError, re.escape(kernel_key) + ".*" + re.escape(filename)):
                        if source == "file":
                            wp.capture_load(path, device="cpu")
                        else:
                            wp.capture_load_from_bytes(invalid, modules, device="cpu")

        graph = wp.capture_load_from_bytes(wrp, modules, device="cpu")
        values = np.arange(N, dtype=np.float32)
        np.testing.assert_allclose(_run_scale_graph(graph, values, "cpu"), values * 2.0)

    @unittest.skipUnless(wp.is_cpu_available(), "Requires CPU support")
    def test_load_missing_cpu_backward_symbol(self):
        """Allow forward replay when a CPU module has no recorded backward symbol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_scale_capture(tmpdir, "cpu")
            wrp, modules = _read_capture(path)
        graph = wp.capture_load_from_bytes(wrp, modules, device="cpu")
        backward_name = wp_context.runtime.core.wp_apic_get_kernel_backward_name(graph._native_graph, 0)
        self.assertTrue(backward_name)
        wrp = wrp.replace(backward_name, b"x" * len(backward_name))

        graph = wp.capture_load_from_bytes(wrp, modules, device="cpu")
        values = np.arange(N, dtype=np.float32)
        np.testing.assert_allclose(_run_scale_graph(graph, values, "cpu"), values * 2.0)

    @unittest.skipUnless(wp.is_cpu_available(), "Requires CPU support")
    def test_load_from_bytes_invalid_module_entries(self):
        """Reject native module entries that repeat a filename or have null fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_scale_capture(tmpdir, "cpu")
            wrp, modules = _read_capture(path)
        ((filename, data),) = modules.items()

        cases = {
            "repeated filename": (
                [(filename, data), (filename, data)],
                re.escape(filename) + " is listed more than once",
            ),
            "null filename": ([(None, data)], "has a null filename"),
            "null data": ([(filename, None)], "has null data"),
        }
        for case, (entries, message_pattern) in cases.items():
            with self.subTest(case=case):
                with _native_errors_silenced():
                    native_graph = _native_load(wrp, entries, "cpu")
                try:
                    self.assertFalse(native_graph)
                    self.assertRegex(wp_context.runtime.get_error_string(), message_pattern)
                finally:
                    if native_graph:
                        wp_context.runtime.core.wp_apic_destroy_graph(native_graph)

    @unittest.skipUnless(wp.is_cpu_available(), "Requires CPU support")
    def test_load_from_bytes_truncated_image(self):
        """Reject truncation within the header, section table, and each data section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_scale_capture(tmpdir, "cpu")
            wrp, modules = _read_capture(path)

        cut_points = {"empty": 0, "header": 32, "section table": 64 + 16}
        for section_type, label in ((1, "metadata"), (2, "memory"), (3, "operations")):
            offset, size = _find_section(wrp, section_type)
            cut_points[label] = offset + size // 2
        for label, length in cut_points.items():
            with self.subTest(cut=label):
                with _native_errors_silenced():
                    with self.assertRaises(RuntimeError):
                        wp.capture_load_from_bytes(wrp[:length], modules, device="cpu")

    @unittest.skipUnless(wp.is_cpu_available(), "Requires CPU support")
    def test_load_from_bytes_bad_section_table(self):
        """Reject section tables or sections that extend past the provided bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_scale_capture(tmpdir, "cpu")
            wrp, modules = _read_capture(path)

        cases = {
            "section table offset": _patch_header(wrp, section_table_offset=len(wrp)),
            "section count": _patch_header(wrp, num_sections=0xFFFFFFFF),
            "section size": _patch_section_size(wrp, 3, 1 << 40),
        }
        for case, patched in cases.items():
            with self.subTest(case=case):
                with _native_errors_silenced():
                    with self.assertRaises(RuntimeError):
                        wp.capture_load_from_bytes(patched, modules, device="cpu")

    @unittest.skipUnless(wp.is_cpu_available(), "Requires CPU support")
    def test_load_from_bytes_invalid_arguments(self):
        """Raise ``TypeError`` or ``ValueError`` for invalid data and module arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_scale_capture(tmpdir, "cpu")
            wrp, modules = _read_capture(path)
        ((filename, data),) = modules.items()

        with self.assertRaises(TypeError):
            wp.capture_load_from_bytes("not bytes", modules, device="cpu")
        with self.assertRaises(ValueError):
            wp.capture_load_from_bytes(np.frombuffer(wrp, dtype=np.uint8)[::2], modules, device="cpu")
        with self.assertRaises(TypeError):
            wp.capture_load_from_bytes(wrp, [(filename, data)], device="cpu")
        with self.assertRaises(TypeError):
            wp.capture_load_from_bytes(wrp, {filename.encode("utf-8"): data}, device="cpu")
        with self.assertRaises(TypeError):
            wp.capture_load_from_bytes(wrp, {filename: 123}, device="cpu")

        with self.assertRaises(ValueError):
            wp.capture_load_from_bytes(wrp, {filename: np.frombuffer(data, dtype=np.uint8)[::2]}, device="cpu")

    @unittest.skipUnless(wp.is_cpu_available(), "Requires CPU support")
    def test_load_from_bytes_initializes_warp(self):
        """Load and replay in a fresh process without calling wp.init()."""
        script = textwrap.dedent(
            """
            import sys
            from pathlib import Path

            import numpy as np
            import warp as wp

            path = Path(sys.argv[1])
            count = int(sys.argv[2])
            wrp = path.with_suffix(".wrp").read_bytes()
            modules = {
                entry.name: entry.read_bytes()
                for entry in path.with_name(path.name + "_modules").iterdir()
            }
            graph = wp.capture_load_from_bytes(wrp, modules, device="cpu")
            values = np.arange(count, dtype=np.float32) + 1.0
            graph.set_param("a", wp.array(values, dtype=float, device="cpu"))
            wp.capture_launch(graph)
            output = wp.empty(count, dtype=float, device="cpu")
            graph.get_param("b", output)
            np.testing.assert_allclose(output.numpy(), values * 2.0)
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_scale_capture(tmpdir, "cpu")
            result = subprocess.run(
                [sys.executable, "-c", script, path, str(N)], capture_output=True, text=True, check=False
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_load_obj_from_bytes(self):
        """Resolve a lazily linked CPU kernel after overwriting its source buffer."""
        if wp_context.runtime.llvm is None:
            self.skipTest("Requires the warp-clang backend")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_scale_capture(tmpdir, "cpu")
            _, modules = _read_capture(path)
            loaded = wp.capture_load(path, device="cpu")

        core = wp_context.runtime.core
        forward_name = core.wp_apic_get_kernel_forward_name(loaded._native_graph, 0)
        filename = core.wp_apic_get_kernel_module_binary_filename(loaded._native_graph, 0).decode("utf-8")
        buffer = bytearray(modules[filename])
        address = ctypes.addressof((ctypes.c_char * len(buffer)).from_buffer(buffer))
        handle = b"wp_apic_test_load_obj_from_bytes"
        result = wp_context.runtime.llvm.wp_load_obj_from_bytes(
            address, len(buffer), handle, wp.config.legacy_cpu_linker
        )
        self.assertEqual(result, 0)
        try:
            buffer[:] = bytes(len(buffer))
            self.assertNotEqual(wp_context.runtime.llvm.wp_lookup(handle, forward_name), 0)
        finally:
            wp_context.runtime.llvm.wp_unload_obj(handle)

    @unittest.skipUnless(wp.is_cpu_available(), "Requires CPU support")
    def test_load_obj_null_arguments(self):
        """Return an error for null object paths or module names without crashing."""
        script = textwrap.dedent(
            """
            import sys

            import warp as wp
            import warp._src.context as wp_context

            wp.init()
            object_file = sys.argv[1].encode("utf-8") if sys.argv[2] != "path" else None
            module_name = b"wp_apic_test_null_arguments" if sys.argv[2] != "module" else None
            result = wp_context.runtime.llvm.wp_load_obj(object_file, module_name, wp.config.legacy_cpu_linker)
            assert result == -1, result
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_scale_capture(tmpdir, "cpu")
            _, modules = _read_capture(path)
            object_file = os.path.join(path + "_modules", next(iter(modules)))
            for field in ("path", "module"):
                with self.subTest(field=field):
                    result = run_python_subprocess(script, object_file, field, hide_gpu=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()
devices = get_test_devices()
cuda_devices_with_cuda_graph_module_load = [d for d in devices_with_cuda_graph_module_load if d.is_cuda]

add_function_test(
    TestApicLoadFromBytes,
    "test_load_from_bytes_matches_disk_load",
    test_load_from_bytes_matches_disk_load,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicLoadFromBytes,
    "test_load_from_bytes_releases_buffers",
    test_load_from_bytes_releases_buffers,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicLoadFromBytes,
    "test_load_from_bytes_buffer_types",
    test_load_from_bytes_buffer_types,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicLoadFromBytes,
    "test_load_from_bytes_multiple_modules",
    test_load_from_bytes_multiple_modules,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicLoadFromBytes,
    "test_load_from_bytes_missing_module",
    test_load_from_bytes_missing_module,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicLoadFromBytes,
    "test_load_from_bytes_bad_module",
    test_load_from_bytes_bad_module,
    devices=devices_with_cuda_graph_module_load,
)

add_function_test(
    TestApicLoadFromBytes,
    "test_load_from_bytes_ptx_module",
    test_load_from_bytes_ptx_module,
    devices=cuda_devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicLoadFromBytes,
    "test_load_from_bytes_empty_native_module",
    test_load_from_bytes_empty_native_module,
    devices=cuda_devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicLoadFromBytes,
    "test_load_from_bytes_without_kernels",
    test_load_from_bytes_without_kernels,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
