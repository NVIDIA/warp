# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for loading APIC captures from in-memory bytes."""

import contextlib
import ctypes
import gc
import os
import re
import struct
import tempfile
import unittest

import numpy as np

import warp as wp
import warp._src.context as wp_context
from warp._src.apic.types import APICModuleBytes
from warp.tests.unittest_utils import (
    add_function_test,
    get_test_devices,
    get_test_devices_with_cuda_graph_module_load,
)

wp.init()

N = 16


@wp.kernel
def scale_kernel(input: wp.array[float], output: wp.array[float], s: float):
    i = wp.tid()
    output[i] = input[i] * s


def _save_scale_capture(tmpdir, device, kernel=scale_kernel):
    """Save a one-launch capture of ``kernel`` and return its path and expected output."""
    a = wp.array(np.arange(N, dtype=np.float32), device=device)
    b = wp.zeros(N, dtype=float, device=device)
    wp.load_module(kernel.module, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(kernel, dim=N, inputs=[a, b, 2.0], device=device)
    path = os.path.join(tmpdir, "scale")
    wp.capture_save(capture.graph, path, inputs={"a": a}, outputs={"b": b})
    return path, np.arange(N, dtype=np.float32) * 2.0


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
    """Call ``wp_apic_load_graph_from_bytes`` with explicit module entries.

    ``entries`` is a list of ``(filename, data)`` pairs where either item may
    be ``None``, so tests can pass entries a dict cannot express. Returns the
    native graph pointer, or ``None`` on failure.
    """
    device = wp.get_device(device)
    keep_alive = []
    native_entries = (APICModuleBytes * max(len(entries), 1))()
    for native_entry, (filename, data) in zip(native_entries, entries, strict=False):
        if filename is not None:
            native_entry.filename = filename.encode("utf-8")
        if data is None:
            native_entry.size = 1
        else:
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


def _destroy_native_graph(native_graph, device):
    """Destroy a native graph returned by ``_native_load``."""
    with wp.get_device(device).context_guard:
        wp_context.runtime.core.wp_apic_destroy_graph(native_graph)


def test_native_load_from_bytes(test, device):
    """Load a capture through the native bytes entry point and query its bindings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)

    native_graph = _native_load(wrp, list(modules.items()), device)
    test.assertTrue(native_graph, wp_context.runtime.get_error_string())
    core = wp_context.runtime.core
    try:
        names = {
            core.wp_apic_get_param_name(native_graph, i).decode("utf-8")
            for i in range(core.wp_apic_get_num_params(native_graph))
        }
        test.assertEqual(names, {"a", "b"})
        test.assertEqual(core.wp_apic_get_param_size(native_graph, b"b"), N * 4)
        if wp.get_device(device).is_cuda:
            test.assertTrue(core.wp_apic_get_cuda_graph_exec(native_graph), wp_context.runtime.get_error_string())
    finally:
        _destroy_native_graph(native_graph, device)


def test_load_from_bytes_invalid_module_entries(test, device):
    """Reject native module entries that repeat a filename or have null fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
    ((filename, data),) = modules.items()

    cases = {
        "repeated filename": ([(filename, data), (filename, data)], re.escape(filename) + " is listed more than once"),
        "null filename": ([(None, data)], "has a null filename"),
        "null data": ([(filename, None)], "has null data"),
    }
    for case, (entries, message_pattern) in cases.items():
        with test.subTest(case=case):
            with _native_errors_silenced():
                native_graph = _native_load(wrp, entries, device)
            test.assertFalse(native_graph)
            test.assertRegex(wp_context.runtime.get_error_string(), message_pattern)


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
    """Load one capture from disk and from bytes and compare params and results.

    The disk-loaded graph is released before the bytes-loaded graph launches,
    so on CPU the bytes-loaded graph must be running its own objects.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
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


def test_load_from_bytes_without_files(test, device):
    """Load and launch a capture after deleting every file it was saved to."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
    test.assertFalse(os.path.exists(path + ".wrp"))

    graph = wp.capture_load_from_bytes(wrp, modules, device=device)
    values = np.arange(N, dtype=np.float32)
    np.testing.assert_allclose(_run_scale_graph(graph, values, device), values * 2.0)


def test_load_from_bytes_releases_buffers(test, device):
    """Overwrite and drop the source buffers after loading, then launch.

    A graph that kept a pointer into the caller's ``.wrp`` or module buffers
    would read zeros or freed memory here instead of its own copies.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
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
    """Accept ``bytes``, ``bytearray``, ``memoryview``, and NumPy inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)

    converters = {
        "bytes": bytes,
        "bytearray": bytearray,
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
    """Load a capture whose module was compiled to PTX.

    PTX is the one module format whose bytes need a NUL terminator, and
    ``_read_capture`` returns file contents without one.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device, kernel=scale_kernel_ptx)
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
        path, _ = _save_scale_capture(tmpdir, device)
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
                with test.assertRaisesRegex(RuntimeError, re.escape(filename) + " was not provided"):
                    wp.capture_load_from_bytes(wrp, provided, device=device)


def test_load_from_bytes_bad_module(test, device):
    """Fail on corrupt module bytes, then load the same capture with intact bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
    corrupt = {name: b"\xff" * len(data) for name, data in modules.items()}

    with _native_errors_silenced():
        with test.assertRaisesRegex(RuntimeError, "Failed to load (CPU )?module"):
            wp.capture_load_from_bytes(wrp, corrupt, device=device)

    graph = wp.capture_load_from_bytes(wrp, modules, device=device)
    values = np.arange(N, dtype=np.float32)
    np.testing.assert_allclose(_run_scale_graph(graph, values, device), values * 2.0)


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


def test_load_from_bytes_truncated_image(test, device):
    """Reject images truncated at each section without reading past the buffer.

    Each prefix ends inside a different part of a valid image: nothing at all,
    the file header, the section table, and each section. Loading a prefix
    must fail with a ``Warp APIC error`` instead of reading beyond it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)

    cut_points = {"empty": 0, "header": 32, "section table": 64 + 16}
    for section_type, label in ((1, "metadata"), (2, "memory"), (3, "operations")):
        offset, size = _find_section(wrp, section_type)
        cut_points[label] = offset + size // 2
    for label, length in cut_points.items():
        with test.subTest(cut=label):
            with _native_errors_silenced():
                with test.assertRaisesRegex(RuntimeError, "Warp APIC error"):
                    wp.capture_load_from_bytes(wrp[:length], modules, device=device)


def test_load_from_bytes_bad_section_table(test, device):
    """Reject a section table or section that points past the image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)

    cases = {
        "section table offset": (_patch_header(wrp, section_table_offset=len(wrp)), "section table is out of bounds"),
        "section count": (_patch_header(wrp, num_sections=0xFFFFFFFF), "section table is out of bounds"),
        "section size": (_patch_section_size(wrp, 3, 1 << 40), r"section \d+ is out of bounds"),
    }
    for case, (patched, message_pattern) in cases.items():
        with test.subTest(case=case):
            with _native_errors_silenced():
                with test.assertRaisesRegex(RuntimeError, message_pattern):
                    wp.capture_load_from_bytes(patched, modules, device=device)


def test_load_from_bytes_invalid_arguments(test, device):
    """Raise ``TypeError`` or ``ValueError`` for invalid data and module arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _save_scale_capture(tmpdir, device)
        wrp, modules = _read_capture(path)
    ((filename, data),) = modules.items()

    with test.assertRaisesRegex(TypeError, "data must be a bytes-like object"):
        wp.capture_load_from_bytes("not bytes", modules, device=device)
    with test.assertRaisesRegex(ValueError, "data must be a C-contiguous buffer"):
        wp.capture_load_from_bytes(np.frombuffer(wrp, dtype=np.uint8)[::2], modules, device=device)
    with test.assertRaisesRegex(TypeError, "modules must be a mapping"):
        wp.capture_load_from_bytes(wrp, [(filename, data)], device=device)
    with test.assertRaisesRegex(TypeError, "Module names must be strings"):
        wp.capture_load_from_bytes(wrp, {filename.encode("utf-8"): data}, device=device)
    with test.assertRaisesRegex(TypeError, "must be a bytes-like object"):
        wp.capture_load_from_bytes(wrp, {filename: 123}, device=device)


def test_load_from_bytes_without_kernels(test, device):
    """Load a capture that launches no kernels without passing modules."""
    src = wp.array(np.arange(N, dtype=np.float32), device=device)
    dst = wp.zeros(N, dtype=float, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.copy(dst, src)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "copy_only")
        wp.capture_save(capture.graph, path, inputs={"src": src}, outputs={"dst": dst})
        wrp, modules = _read_capture(path)
    test.assertEqual(modules, {})

    graph = wp.capture_load_from_bytes(wrp, device=device)
    graph.set_param("src", wp.array(np.full(N, 5.0, dtype=np.float32), device=device))
    wp.capture_launch(graph)
    out = wp.zeros(N, dtype=float, device=device)
    graph.get_param("dst", out)
    np.testing.assert_allclose(out.numpy(), np.full(N, 5.0, dtype=np.float32))


class TestApicFromBytes(unittest.TestCase):
    def test_load_obj_from_bytes(self):
        """Load a CPU object from bytes and resolve a kernel after the source buffer is zeroed.

        LLVM links an object lazily on the first symbol lookup, so the lookup
        runs after the caller's buffer is overwritten. A loader that kept a
        pointer into that buffer would fail to resolve the symbol.
        """
        if wp_context.runtime.llvm is None:
            self.skipTest("Requires the warp-clang backend")

        with tempfile.TemporaryDirectory() as tmpdir:
            path, _ = _save_scale_capture(tmpdir, "cpu")
            _, modules = _read_capture(path)
            loaded = wp.capture_load(path, device="cpu")

        forward_name = wp_context.runtime.core.wp_apic_get_kernel_forward_name(loaded._native_graph, 0)
        self.assertEqual(len(modules), 1)
        (data,) = modules.values()

        buffer = bytearray(data)
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


devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()

add_function_test(
    TestApicFromBytes,
    "test_native_load_from_bytes",
    test_native_load_from_bytes,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicFromBytes,
    "test_load_from_bytes_invalid_module_entries",
    test_load_from_bytes_invalid_module_entries,
    devices=devices_with_cuda_graph_module_load,
)

devices = get_test_devices()
cuda_devices_with_cuda_graph_module_load = [d for d in devices_with_cuda_graph_module_load if d.is_cuda]

for _test in (
    test_load_from_bytes_matches_disk_load,
    test_load_from_bytes_without_files,
    test_load_from_bytes_releases_buffers,
    test_load_from_bytes_buffer_types,
    test_load_from_bytes_multiple_modules,
    test_load_from_bytes_missing_module,
    test_load_from_bytes_bad_module,
    test_load_from_bytes_truncated_image,
    test_load_from_bytes_bad_section_table,
    test_load_from_bytes_invalid_arguments,
):
    add_function_test(TestApicFromBytes, _test.__name__, _test, devices=devices_with_cuda_graph_module_load)

add_function_test(
    TestApicFromBytes,
    "test_load_from_bytes_ptx_module",
    test_load_from_bytes_ptx_module,
    devices=cuda_devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApicFromBytes,
    "test_load_from_bytes_without_kernels",
    test_load_from_bytes_without_kernels,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
