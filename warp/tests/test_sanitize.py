# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AddressSanitizer (ASan) coverage for JIT-compiled CPU kernels.

These tests only run when ``warp-clang`` was built with ASan
(``build_lib.py --sanitize=address``); otherwise the whole class self-skips.
A correct in-bounds kernel must run clean, and an out-of-bounds access into a
``wp.array`` must be reported by ASan as a ``heap-buffer-overflow``.

The out-of-bounds kernels run in Warp ``"release"`` mode on purpose: Warp's
``"debug"`` mode bounds-checks array indexing with ``assert`` and would abort
before ASan ever sees the raw access. ASan instrumentation itself is independent
of Warp's mode (it is auto-enabled whenever ``warp-clang`` is an ASan build).

Because an ASan report aborts the process, the triggers run in subprocesses.
"""

import os
import signal
import subprocess
import sys
import unittest

import warp as wp
import warp._src.context as wp_context


def _asan_subprocess_env() -> dict:
    """Environment for a child process that loads an ASan-instrumented Warp.

    The ASan runtime must initialize before any instrumented code runs, and on
    Linux it must be in the process global symbol scope so the JIT can resolve a
    kernel's ``__asan_*`` references (Warp loads ``warp-clang`` ``RTLD_LOCAL``).
    """
    env = os.environ.copy()

    # Force the options this test depends on by appending them after any inherited
    # values. ASan applies the last value for duplicate keys, and preserving the raw
    # string avoids corrupting option values that contain ':' characters.
    # detect_leaks=0 keeps the signal scoped to heap-buffer-overflow: LeakSanitizer (on by
    # default with ASan on Linux) would otherwise flag unrelated process-shutdown leaks from
    # Python/LLVM/Warp/etc. and fail the clean-exit case. Leak coverage belongs in a
    # dedicated job with its own suppressions.
    asan_options = env.get("ASAN_OPTIONS", "")
    forced_asan_options = "verify_asan_link_order=0:abort_on_error=1:detect_leaks=0"
    env["ASAN_OPTIONS"] = f"{asan_options}:{forced_asan_options}" if asan_options else forced_asan_options

    if sys.platform not in ("win32", "darwin") and "LD_PRELOAD" not in env:
        # Preload the ASan runtime so its __asan_* symbols are globally visible
        # (RTLD_GLOBAL), which DynamicLibrarySearchGenerator::GetForCurrentProcess
        # relies on. Try GCC's libasan first, then Clang's compiler-rt runtime.
        for query in (
            ["gcc", "-print-file-name=libasan.so"],
            ["clang", f"-print-file-name=libclang_rt.asan-{os.uname().machine}.so"],
        ):
            try:
                path = subprocess.run(query, capture_output=True, text=True, check=False).stdout.strip()
            except (OSError, ValueError):
                path = ""
            if path and os.path.isabs(path) and os.path.exists(path):
                env["LD_PRELOAD"] = path
                break

    return env


def _run_in_subprocess(func_name: str, timeout: int = 120):
    """Run a module-level function of this module in a subprocess.

    Returns ``(returncode, stdout, stderr)``. Mirrors ``test_assert`` but targets
    this module and injects the ASan environment.
    """
    setup_code = ""
    if sys.platform == "win32":
        # These tests intentionally abort subprocesses. Suppress Windows Error
        # Reporting so systems with LocalDumps enabled do not collect the
        # expected aborts as false-positive application crash dumps.
        setup_code = (
            "import ctypes;"
            "SEM_FAILCRITICALERRORS=0x0001;"
            "SEM_NOGPFAULTERRORBOX=0x0002;"
            "ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);"
        )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"{setup_code}import warp.tests.test_sanitize as m; m.{func_name}()",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_asan_subprocess_env(),
    )
    return result.returncode, result.stdout, result.stderr


def _assert_aborted(test_case, returncode):
    """Assert the process was killed by an abort-like signal (or nonzero on Windows)."""
    if sys.platform == "win32":
        test_case.assertNotEqual(returncode, 0)
    else:
        test_case.assertIn(returncode, [-signal.SIGABRT, -signal.SIGILL, -signal.SIGSEGV, -signal.SIGTRAP])


@wp.kernel
def _write_at_offset(a: wp.array[int], offset: int):
    i = wp.tid()
    a[i + offset] = 42


# Functions invoked in subprocesses (must be module-level for Warp codegen).
def _trigger_heap_overflow():
    wp.config.cache_kernels = False
    wp.config.mode = "release"  # avoid Warp's debug bounds-assert masking the ASan report
    with wp.ScopedDevice("cpu"):
        a = wp.zeros(16, dtype=int)
        # tid 0 writes a[16] — one element past the end, into ASan's redzone.
        wp.launch(_write_at_offset, dim=1, inputs=[a, 16])
        wp.synchronize_device()


def _trigger_inbounds_ok():
    wp.config.cache_kernels = False
    wp.config.mode = "release"
    with wp.ScopedDevice("cpu"):
        a = wp.zeros(16, dtype=int)
        wp.launch(_write_at_offset, dim=16, inputs=[a, 0])  # writes a[0..15], all in bounds
        wp.synchronize_device()
        assert a.numpy().tolist() == [42] * 16
        print("INBOUNDS_OK", flush=True)


def _trigger_logical_overflow():
    """Write one element past a sub-alignment array's logical end to exercise ASan's exact-size redzone.

    A single-element array is logically 4 bytes, well under the host allocator's 64-byte alignment. Writing ``a[1]``
    is one element past the logical end but lands inside that alignment padding. ASan only catches this when the
    allocation is the exact requested size: ``wp_alloc_host()`` must use an exact-size aligned allocator
    (``posix_memalign()`` on POSIX, ``_aligned_malloc()`` on Windows) rather than rounding the request up to a
    multiple of the alignment, which would bury the redzone past the padding and hide this access.
    """
    wp.config.cache_kernels = False
    wp.config.mode = "release"
    with wp.ScopedDevice("cpu"):
        a = wp.zeros(1, dtype=int)
        wp.launch(_write_at_offset, dim=1, inputs=[a, 1])  # writes a[1] — past the logical end
        wp.synchronize_device()


class TestSanitize(unittest.TestCase):
    """ASan coverage for JIT-compiled CPU kernels (runs only on ASan builds)."""

    def _skip_unless_asan(self):
        """Skip the current test unless Warp is running on an ASan build.

        Skip only after a successful init that reports a non-ASan build. An
        init/load failure (e.g. a misconfigured ASan build where the runtime is not
        on PATH/LD_PRELOAD) propagates as a loud error rather than a silent skip
        that would hide lost coverage.

        Skip per test via ``self.skipTest()`` rather than in ``setUpClass``: the
        parallel JUnit runner records a ``setUpClass`` skip via ``addSkip`` without
        a preceding ``startTest`` and then trips over its own timing state. Skipping
        inside the running test ensures the skip follows ``startTest``.
        """
        wp.init()
        if getattr(wp_context.runtime, "clang_sanitizer", "") != "address":
            self.skipTest("requires a warp-clang built with --sanitize=address")

    def test_heap_buffer_overflow_detected(self):
        self._skip_unless_asan()
        returncode, _stdout, stderr = _run_in_subprocess("_trigger_heap_overflow")
        _assert_aborted(self, returncode)
        self.assertIn("AddressSanitizer", stderr)
        self.assertIn("heap-buffer-overflow", stderr)

    def test_logical_bound_overflow_detected(self):
        """Verify a write just past a sub-alignment array's logical end is still reported.

        Guards against the host allocator rounding requests up to the alignment: the write must still be reported,
        proving the redzone tracks the requested size rather than the padded allocation.
        """
        self._skip_unless_asan()
        returncode, _stdout, stderr = _run_in_subprocess("_trigger_logical_overflow")
        _assert_aborted(self, returncode)
        self.assertIn("AddressSanitizer", stderr)
        self.assertIn("heap-buffer-overflow", stderr)

    def test_inbounds_runs_clean(self):
        self._skip_unless_asan()
        returncode, stdout, stderr = _run_in_subprocess("_trigger_inbounds_ok")
        self.assertEqual(returncode, 0, f"expected clean exit, got {returncode}; stderr:\n{stderr}")
        self.assertIn("INBOUNDS_OK", stdout)
        self.assertNotIn("AddressSanitizer", stderr)


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
