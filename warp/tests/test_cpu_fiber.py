# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the CPU fiber library (warp/native/cpu_fiber.{h,cpp}).

Tests the C-level fiber primitive directly via ctypes.

Each scenario is run in a subprocess so that any fiber-related crash
(stack overflow on a destroyed fiber, register-save bug, etc.) only
takes down the child, not the whole test runner. This pattern is
identical to warp/tests/test_assert.py.
"""

import ctypes
import os
import subprocess
import sys
import unittest

import warp as wp

# Module-level "scenario" entry points ? each is run in its own subprocess via
# `python -c "import warp.tests.test_cpu_fiber as m; m._run_<name>()"`.


def _warp_lib_path():
    bindir = os.path.join(os.path.dirname(wp.__file__), "bin")
    if sys.platform == "win32":
        return os.path.join(bindir, "warp.dll")
    if sys.platform == "darwin":
        return os.path.join(bindir, "libwarp.dylib")
    return os.path.join(bindir, "warp.so")


def _setup():
    wp.init()

    lib = ctypes.CDLL(_warp_lib_path())
    lib.wp_fiber_create.restype = ctypes.c_void_p
    lib.wp_fiber_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.wp_fiber_destroy.argtypes = [ctypes.c_void_p]
    lib.wp_fiber_switch.argtypes = [ctypes.c_void_p]
    lib.wp_fiber_active.restype = ctypes.c_void_p
    lib.wp_fiber_finished.argtypes = [ctypes.c_void_p]
    lib.wp_fiber_finished.restype = ctypes.c_int
    lib.wp_fiber_test_abi.restype = ctypes.c_int
    lib.wp_fiber_test_deep_calls.argtypes = [ctypes.c_size_t, ctypes.c_int]
    lib.wp_fiber_test_deep_calls.restype = ctypes.c_int
    lib.wp_fiber_test_overflow.argtypes = [ctypes.c_size_t]
    return lib, ctypes


def _run_active_returns_main():
    lib, _ = _setup()
    h1 = lib.wp_fiber_active()
    h2 = lib.wp_fiber_active()
    assert h1 == h2 and h1 != 0, f"unstable main fiber handle: {h1} vs {h2}"
    print("ok")


def _run_two_fiber_pingpong():
    lib, ctypes = _setup()
    N_SWITCHES = 1000

    main = lib.wp_fiber_active()
    native_switch_entry = ctypes.cast(lib.wp_fiber_switch, ctypes.c_void_p)

    # A ctypes callback must not be suspended across a fiber switch: CPython's
    # execution state is thread-local, not fiber-local. Use the native switch
    # function itself as each entry so this stress test stays entirely in C
    # while fibers are away from the main Python stack.
    for i in range(N_SWITCHES):
        b = lib.wp_fiber_create(native_switch_entry, main, 64 * 1024)
        assert b, f"wp_fiber_create returned a null handle for fiber 'b' at iteration {i}"
        a = lib.wp_fiber_create(native_switch_entry, b, 64 * 1024)
        assert a, f"wp_fiber_create returned a null handle for fiber 'a' at iteration {i}"

        # main -> a -> b -> main, leaving both entries suspended inside their
        # native wp_fiber_switch calls.
        lib.wp_fiber_switch(a)
        assert lib.wp_fiber_active() == main
        assert lib.wp_fiber_finished(a) == 0
        assert lib.wp_fiber_finished(b) == 0

        # Resume the suspended entries in reverse order and let each return.
        lib.wp_fiber_switch(b)
        assert lib.wp_fiber_finished(b) == 1
        lib.wp_fiber_switch(a)
        assert lib.wp_fiber_finished(a) == 1

        lib.wp_fiber_destroy(a)
        lib.wp_fiber_destroy(b)
    print("ok")


def _run_round_robin_1024_fibers():
    lib, ctypes = _setup()
    N = 1024
    GEN = 2

    main = lib.wp_fiber_active()
    native_switch_entry = ctypes.cast(lib.wp_fiber_switch, ctypes.c_void_p)

    for generation in range(GEN):
        fibers = [None] * N
        target = main
        for i in range(N - 1, -1, -1):
            fibers[i] = lib.wp_fiber_create(native_switch_entry, target, 64 * 1024)
            assert fibers[i], f"wp_fiber_create returned a null handle for fiber {i} in generation {generation}"
            target = fibers[i]

        # Traverse all 64 native entries before returning to main.
        lib.wp_fiber_switch(fibers[0])
        assert lib.wp_fiber_active() == main
        assert all(lib.wp_fiber_finished(fiber) == 0 for fiber in fibers)

        # Resume the chain from the tail so every entry can return cleanly.
        for i in range(N - 1, -1, -1):
            lib.wp_fiber_switch(fibers[i])
            assert lib.wp_fiber_finished(fibers[i]) == 1

        for fiber in fibers:
            lib.wp_fiber_destroy(fiber)
    print("ok")


def _run_abi_state():
    lib, _ = _setup()
    assert lib.wp_fiber_test_abi() == 1, "native ABI state probe failed"
    print("ok")


def _run_deep_calls():
    lib, _ = _setup()
    assert lib.wp_fiber_test_deep_calls(1024 * 1024, 512) == 1, "deep native calls corrupted the fiber stack"
    print("ok")


def _run_guarded_stack_overflow():
    lib, _ = _setup()
    lib.wp_fiber_test_overflow(64 * 1024)
    raise AssertionError("guarded fiber stack overflow returned normally")


def _run_finished_after_entry_returns():
    lib, ctypes = _setup()
    FT = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    ran = [False]

    def entry(_):
        ran[0] = True

    cb = FT(entry)
    f = lib.wp_fiber_create(ctypes.cast(cb, ctypes.c_void_p), None, 64 * 1024)
    assert f, "wp_fiber_create returned a null handle"
    assert lib.wp_fiber_finished(f) == 0
    lib.wp_fiber_switch(f)
    assert ran[0]
    assert lib.wp_fiber_finished(f) == 1
    lib.wp_fiber_destroy(f)
    print("ok")


def _run_return_resumes_main_fiber():
    """Verify that a returning fiber resumes main, not its last caller."""
    lib, ctypes = _setup()
    FT = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    log = []
    main = lib.wp_fiber_active()

    def second_entry(_):
        log.append("second-return")

    second_cb = FT(second_entry)
    second = lib.wp_fiber_create(ctypes.cast(second_cb, ctypes.c_void_p), None, 64 * 1024)
    native_switch_entry = ctypes.cast(lib.wp_fiber_switch, ctypes.c_void_p)
    first = lib.wp_fiber_create(native_switch_entry, second, 64 * 1024)
    assert first and second, "wp_fiber_create returned a null handle"

    lib.wp_fiber_switch(first)

    assert log == ["second-return"], f"return resumed the wrong fiber: {log}"
    assert lib.wp_fiber_active() == main
    assert lib.wp_fiber_finished(first) == 0
    assert lib.wp_fiber_finished(second) == 1
    lib.wp_fiber_destroy(first)
    lib.wp_fiber_destroy(second)
    print("ok")


def _run_destroy_rejects_active_and_main():
    """Invalid destruction is harmless even when assertions are disabled."""
    lib, ctypes = _setup()
    FT = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    main = lib.wp_fiber_active()
    fiber = ctypes.c_void_p()
    ran = [False]

    def entry(_):
        active = lib.wp_fiber_active()
        assert active == fiber.value
        lib.wp_fiber_destroy(active)
        lib.wp_fiber_destroy(main)
        ran[0] = True

    cb = FT(entry)
    fiber.value = lib.wp_fiber_create(ctypes.cast(cb, ctypes.c_void_p), None, 64 * 1024)
    assert fiber.value, "wp_fiber_create returned a null handle"
    lib.wp_fiber_switch(fiber)
    assert ran[0]
    assert lib.wp_fiber_finished(fiber) == 1
    assert lib.wp_fiber_active() == main
    lib.wp_fiber_destroy(main)
    lib.wp_fiber_destroy(fiber)
    print("ok")


def _run_windows_preconverted_thread():
    """Cooperate with a Windows host that converted the thread first."""
    lib, ctypes = _setup()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.ConvertThreadToFiber.argtypes = [ctypes.c_void_p]
    kernel32.ConvertThreadToFiber.restype = ctypes.c_void_p

    host_main = kernel32.ConvertThreadToFiber(None)
    assert host_main, f"ConvertThreadToFiber failed with error {ctypes.get_last_error()}"

    main = lib.wp_fiber_active()
    assert main, "wp_fiber_active rejected a thread already converted by its host"

    FT = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    ran = [False]

    def entry(_):
        ran[0] = True

    cb = FT(entry)
    fiber = lib.wp_fiber_create(ctypes.cast(cb, ctypes.c_void_p), None, 64 * 1024)
    assert fiber, "wp_fiber_create failed on a thread converted by its host"
    lib.wp_fiber_switch(fiber)
    assert ran[0]
    assert lib.wp_fiber_active() == main
    assert lib.wp_fiber_finished(fiber) == 1
    lib.wp_fiber_destroy(fiber)
    print("ok")


def _run_in_subprocess(scenario: str, timeout: int = 60):
    result = subprocess.run(
        [sys.executable, "-c", f"import warp.tests.test_cpu_fiber as m; m._run_{scenario}()"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


class TestCpuFiber(unittest.TestCase):
    """Run each fiber scenario in an isolated subprocess.

    Isolation keeps fiber stacks from interacting with the test runner's C stack.
    """

    def _check(self, scenario, timeout=60):
        rc, stdout, stderr = _run_in_subprocess(scenario, timeout=timeout)
        self.assertEqual(rc, 0, f"scenario {scenario!r} failed (rc={rc}):\nstdout:\n{stdout}\nstderr:\n{stderr}")
        self.assertIn("ok", stdout)

    def test_active_returns_main(self):
        self._check("active_returns_main")

    def test_two_fiber_pingpong(self):
        self._check("two_fiber_pingpong")

    def test_round_robin_1024_fibers(self):
        self._check("round_robin_1024_fibers", timeout=120)

    def test_abi_state(self):
        self._check("abi_state")

    def test_deep_calls(self):
        self._check("deep_calls")

    def test_guarded_stack_overflow(self):
        rc, stdout, _ = _run_in_subprocess("guarded_stack_overflow")
        self.assertNotEqual(rc, 0, f"guarded stack overflow unexpectedly survived:\n{stdout}")

    def test_finished_after_entry_returns(self):
        self._check("finished_after_entry_returns")

    def test_return_resumes_main_fiber(self):
        self._check("return_resumes_main_fiber")

    def test_destroy_rejects_active_and_main(self):
        self._check("destroy_rejects_active_and_main")

    @unittest.skipUnless(sys.platform == "win32", "requires Windows fibers")
    def test_windows_preconverted_thread(self):
        self._check("windows_preconverted_thread")


if __name__ == "__main__":
    unittest.main()
