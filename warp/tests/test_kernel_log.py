# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the kernel logging system (wp.log / wp.LOG_*)."""

import contextlib
import logging
import unittest

import warp as wp
import warp._src.context as ctx
import warp.config
from warp.tests.unittest_utils import add_function_test, get_test_devices

# ---------------------------------------------------------------------------
# Kernel definitions at module scope (required for inspect.getsourcelines)
# ---------------------------------------------------------------------------


@wp.kernel
def _kernel_log_basic(out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    wp.log(wp.LOG_INFO, "basic test", i)
    out[i] = i


@wp.kernel
def _kernel_log_no_payload(out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    wp.log(wp.LOG_WARN, "no payload here")
    out[i] = i


@wp.kernel
def _kernel_log_int64(out: wp.array(dtype=wp.int64)):
    i = wp.tid()
    v = wp.int64(i) * wp.int64(1000)
    wp.log(wp.LOG_DEBUG, "int64 payload", v)
    out[i] = v


@wp.kernel
def _kernel_log_float32(out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    x = wp.float32(i) * wp.float32(0.5)
    wp.log(wp.LOG_ERROR, "float32 payload", x)
    out[i] = x


@wp.kernel
def _kernel_log_overflow(n: int, count: wp.array(dtype=wp.int32)):
    i = wp.tid()
    wp.log(wp.LOG_WARN, "overflow test", i)
    count[i] = i


@wp.kernel
def _kernel_log_all_levels(out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    wp.log(wp.LOG_DEBUG, "debug msg", i)
    wp.log(wp.LOG_INFO, "info msg", i)
    wp.log(wp.LOG_WARN, "warn msg", i)
    wp.log(wp.LOG_ERROR, "error msg", i)
    out[i] = i


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_kernel_log(level=logging.DEBUG):
    """Attach a temporary handler to the ``warp.kernel`` logger and yield the record list."""
    records = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record)

    cap = _Cap()
    logger = logging.getLogger("warp.kernel")
    original_level = logger.level
    logger.addHandler(cap)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(cap)
        logger.setLevel(original_level)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestKernelLog(unittest.TestCase):
    # ------------------------------------------------------------------
    # Public constants
    # ------------------------------------------------------------------

    def test_level_constants(self):
        """LOG_* constants have the expected Python logging level values."""
        self.assertEqual(wp.LOG_DEBUG, logging.DEBUG)  # 10
        self.assertEqual(wp.LOG_INFO, logging.INFO)  # 20
        self.assertEqual(wp.LOG_WARNING, logging.WARNING)  # 30
        self.assertEqual(wp.LOG_WARN, logging.WARNING)  # deprecated alias — same value
        self.assertEqual(wp.LOG_ERROR, logging.ERROR)  # 40

    # ------------------------------------------------------------------
    # Codegen / static validation
    # ------------------------------------------------------------------

    def test_codegen_uses_logging_flag(self):
        """Kernels with wp.log() set uses_logging=True on the Adjoint."""
        _kernel_log_basic.adj.build(_kernel_log_basic.module)
        self.assertTrue(_kernel_log_basic.adj.uses_logging)

    def test_codegen_no_payload_kernel(self):
        """Kernels with no-payload wp.log() also set uses_logging=True."""
        _kernel_log_no_payload.adj.build(_kernel_log_no_payload.module)
        self.assertTrue(_kernel_log_no_payload.adj.uses_logging)

    def test_codegen_call_site_registered(self):
        """Each wp.log() call registers a (level, msg, file, line) entry."""
        original = dict(ctx.runtime.log_call_sites)
        ctx.runtime.log_call_sites.clear()
        try:
            _kernel_log_basic.adj.build(_kernel_log_basic.module)
            sites = ctx.runtime.log_call_sites
            self.assertGreater(len(sites), 0)
            level, msg, filename, lineno = next(iter(sites.values()))
            self.assertEqual(level, wp.LOG_INFO)
            self.assertEqual(msg, "basic test")
            self.assertIn("test_kernel_log", filename)
            self.assertIsInstance(lineno, int)
        finally:
            ctx.runtime.log_call_sites.clear()
            ctx.runtime.log_call_sites.update(original)

    # ------------------------------------------------------------------
    # Runtime tests (require device execution)
    # ------------------------------------------------------------------


def test_log_basic(test, device):
    """wp.log(LOG_INFO, msg, i) records arrive on the warp.kernel logger."""
    n = 4
    out = wp.zeros(n, dtype=wp.int32, device=device)
    with _capture_kernel_log() as records:
        wp.launch(_kernel_log_basic, dim=n, inputs=[out], device=device)
        wp.synchronize_device(device)

    test.assertGreater(len(records), 0, "No log records received after synchronize")
    r = records[0]
    test.assertEqual(r.levelno, wp.LOG_INFO)
    test.assertIn("basic test", r.getMessage())
    # Standard LogRecord fields point to the kernel source, not to context.py
    test.assertIn("test_kernel_log", r.filename)
    test.assertIsInstance(r.lineno, int)
    test.assertIsInstance(r.warp_payload, int)


def test_log_no_payload(test, device):
    """wp.log(level, msg) without a payload produces records without warp_payload."""
    n = 2
    out = wp.zeros(n, dtype=wp.int32, device=device)
    with _capture_kernel_log() as records:
        wp.launch(_kernel_log_no_payload, dim=n, inputs=[out], device=device)
        wp.synchronize_device(device)

    test.assertGreater(len(records), 0)
    r = records[0]
    test.assertIn("no payload here", r.getMessage())
    test.assertFalse(hasattr(r, "warp_payload"))


def test_log_payload_int64(test, device):
    """wp.log with an int64 payload produces correct integer records."""
    n = 3
    out = wp.zeros(n, dtype=wp.int64, device=device)
    with _capture_kernel_log() as records:
        wp.launch(_kernel_log_int64, dim=n, inputs=[out], device=device)
        wp.synchronize_device(device)

    test.assertGreater(len(records), 0)
    for r in records:
        test.assertIn("int64 payload", r.getMessage())
        test.assertIsInstance(r.warp_payload, int)


def test_log_payload_float32(test, device):
    """wp.log with a float32 payload produces correct float records."""
    n = 4
    out = wp.zeros(n, dtype=wp.float32, device=device)
    with _capture_kernel_log() as records:
        wp.launch(_kernel_log_float32, dim=n, inputs=[out], device=device)
        wp.synchronize_device(device)

    test.assertGreater(len(records), 0)
    for r in records:
        test.assertIsInstance(r.warp_payload, float)


def test_log_not_drained_before_sync(test, device):
    """Records must NOT be visible until synchronize_device() is called."""
    n = 2
    out = wp.zeros(n, dtype=wp.int32, device=device)
    with _capture_kernel_log() as records:
        wp.launch(_kernel_log_no_payload, dim=n, inputs=[out], device=device)
        test.assertEqual(len(records), 0, "Handler must not fire before synchronize")
        wp.synchronize_device(device)
        test.assertGreater(len(records), 0, "Handler must fire after synchronize_device")


def test_log_all_levels(test, device):
    """All four log levels produce records with the correct level value."""
    n = 1
    out = wp.zeros(n, dtype=wp.int32, device=device)
    with _capture_kernel_log() as records:
        wp.launch(_kernel_log_all_levels, dim=n, inputs=[out], device=device)
        wp.synchronize_device(device)

    test.assertGreaterEqual(len(records), 4)
    levels = {r.levelno for r in records}
    test.assertIn(wp.LOG_DEBUG, levels)
    test.assertIn(wp.LOG_INFO, levels)
    test.assertIn(wp.LOG_WARN, levels)
    test.assertIn(wp.LOG_ERROR, levels)


def test_log_overflow(test, device):
    """When buffer is full, records are dropped and an overflow warning is logged."""
    original_capacity = warp.config.kernel_log_capacity
    warp.config.kernel_log_capacity = 4

    dev_obj = wp.get_device(device)
    dev_obj._discard_kernel_log_buffer()

    try:
        n = 100
        count = wp.zeros(n, dtype=wp.int32, device=device)
        with _capture_kernel_log() as records:
            wp.launch(_kernel_log_overflow, dim=n, inputs=[n, count], device=device)
            wp.synchronize_device(device)

        overflow_msgs = [r for r in records if "dropped" in r.getMessage()]
        test.assertGreater(len(overflow_msgs), 0, "Expected overflow warning in warp.kernel logger")
    finally:
        warp.config.kernel_log_capacity = original_capacity
        dev_obj._discard_kernel_log_buffer()


def test_log_stdlib_logger(test, device):
    """Records route to logging.getLogger("warp.kernel") with standard filename/lineno fields."""
    n = 2
    out = wp.zeros(n, dtype=wp.int32, device=device)
    with _capture_kernel_log() as records:
        wp.launch(_kernel_log_all_levels, dim=n, inputs=[out], device=device)
        wp.synchronize_device(device)

    test.assertGreater(len(records), 0, "Expected log records on warp.kernel logger")
    r = records[0]
    # Standard LogRecord fields must point to the kernel source so that
    # %(filename)s and %(lineno)d in formatter strings work without customisation.
    test.assertIn("test_kernel_log", r.filename)
    test.assertIsInstance(r.lineno, int)
    test.assertGreater(r.lineno, 0)


def test_reset_kernel_log(test, device):
    """reset_kernel_log() clears the buffer so records are not drained."""
    n = 4
    out = wp.zeros(n, dtype=wp.int32, device=device)

    # Warm up — ensure the kernel is compiled and one drain cycle completes
    with _capture_kernel_log():
        wp.launch(_kernel_log_basic, dim=n, inputs=[out], device=device)
        wp.synchronize_device(device)

    # Launch, reset before drain, then sync — should produce no records
    with _capture_kernel_log() as records:
        wp.launch(_kernel_log_basic, dim=n, inputs=[out], device=device)
        wp.reset_kernel_log(device)
        wp.synchronize_device(device)

    test.assertEqual(len(records), 0, "Expected no records after reset_kernel_log")


def test_get_overflow_count(test, device):
    """get_kernel_log_overflow_count returns 0 when no overflow has occurred."""
    dev_obj = wp.get_device(device)
    buf = dev_obj.get_kernel_log_buffer()
    buf.reset()
    count = wp.get_kernel_log_overflow_count(device)
    test.assertEqual(count, 0)


def test_log_synchronize_stream(test, device):
    """synchronize_stream() drains the synchronized stream's own log buffer."""
    if not wp.get_device(device).is_cuda:
        return  # synchronize_stream is CUDA-only

    n = 4
    out = wp.zeros(n, dtype=wp.int32, device=device)
    stream = wp.Stream(device)
    with _capture_kernel_log() as records:
        wp.launch(_kernel_log_basic, dim=n, inputs=[out], device=device, stream=stream)
        test.assertEqual(len(records), 0, "No records before synchronize_stream")
        wp.synchronize_stream(stream)
        test.assertGreater(len(records), 0, "Records visible after synchronize_stream")


# ---------------------------------------------------------------------------
# Wire up multi-device tests
# ---------------------------------------------------------------------------

devices = get_test_devices()

add_function_test(TestKernelLog, "test_log_basic", test_log_basic, devices=devices)
add_function_test(TestKernelLog, "test_log_no_payload", test_log_no_payload, devices=devices)
add_function_test(TestKernelLog, "test_log_payload_int64", test_log_payload_int64, devices=devices)
add_function_test(TestKernelLog, "test_log_payload_float32", test_log_payload_float32, devices=devices)
add_function_test(TestKernelLog, "test_log_not_drained_before_sync", test_log_not_drained_before_sync, devices=devices)
add_function_test(TestKernelLog, "test_log_all_levels", test_log_all_levels, devices=devices)
add_function_test(TestKernelLog, "test_log_overflow", test_log_overflow, devices=devices)
add_function_test(TestKernelLog, "test_log_stdlib_logger", test_log_stdlib_logger, devices=devices)
add_function_test(TestKernelLog, "test_reset_kernel_log", test_reset_kernel_log, devices=devices)
add_function_test(TestKernelLog, "test_get_overflow_count", test_get_overflow_count, devices=devices)
add_function_test(TestKernelLog, "test_log_synchronize_stream", test_log_synchronize_stream, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
