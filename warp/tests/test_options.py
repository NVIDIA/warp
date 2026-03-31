# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
import io
import runpy
import sys
import types
import unittest
from unittest.mock import patch

import warp as wp
from warp._src.context import _get_caller_module_name
from warp.tests.unittest_utils import *


@wp.kernel
def scale(
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
):
    y[0] = x[0] ** 2.0


@wp.kernel(enable_backward=True)
def scale_1(
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
):
    y[0] = x[0] ** 2.0


@wp.kernel(enable_backward=False)
def scale_2(
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
):
    y[0] = x[0] ** 2.0


def test_options_backward_1(test, device):
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.set_module_options({"enable_backward": False})

    tape = wp.Tape()
    with tape:
        wp.launch(scale, dim=1, inputs=[x, y], device=device)

    with contextlib.redirect_stdout(io.StringIO()) as f:
        tape.backward(y)

    expected = f"Warp UserWarning: Running the tape backwards may produce incorrect gradients because recorded kernel {scale.key} is defined in a module with the option 'enable_backward=False' set.\n"

    assert f.getvalue() == expected
    assert_np_equal(tape.gradients[x].numpy(), np.array(0.0))


def test_options_backward_2(test, device):
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.set_module_options({"enable_backward": True})

    tape = wp.Tape()
    with tape:
        wp.launch(scale, dim=1, inputs=[x, y], device=device)

    tape.backward(y)
    assert_np_equal(tape.gradients[x].numpy(), np.array(6.0))


def test_options_backward_3(test, device):
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.set_module_options({"enable_backward": False})

    tape = wp.Tape()
    with tape:
        wp.launch(scale_1, dim=1, inputs=[x, y], device=device)

    tape.backward(y)
    assert_np_equal(tape.gradients[x].numpy(), np.array(6.0))


def test_options_backward_4(test, device):
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.set_module_options({"enable_backward": True})

    tape = wp.Tape()
    with tape:
        wp.launch(scale_2, dim=1, inputs=[x, y], device=device)

    with contextlib.redirect_stdout(io.StringIO()) as f:
        tape.backward(y)

    expected = f"Warp UserWarning: Running the tape backwards may produce incorrect gradients because recorded kernel {scale_2.key} is configured with the option 'enable_backward=False'.\n"

    assert f.getvalue() == expected
    assert_np_equal(tape.gradients[x].numpy(), np.array(0.0))


def test_options_opt_level(test, device):
    assert wp.config.optimization_level is None, "Default global `optimization_level` should be None"
    assert wp.get_module_options()["optimization_level"] is None, "Default module `optimization_level` should be None"

    wp.set_module_options({"optimization_level": 2})

    x = wp.array([4.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.launch(scale, dim=1, inputs=[x, y], device=device)
    assert y.numpy()[0] == 16.0

    # Reset to default for the next device
    wp.set_module_options({"optimization_level": None})


def test_options_cpu_compiler_flags_generic(test, device):
    """Compiling with cpu_compiler_flags="" (generic target) should not crash."""
    if device.is_cuda:
        return

    old_flags = wp.config.cpu_compiler_flags
    try:
        wp.set_module_options({"cpu_compiler_flags": ""})

        x = wp.array([3.0], dtype=float, device=device)
        y = wp.zeros_like(x)
        wp.launch(scale, dim=1, inputs=[x, y], device=device)
        assert y.numpy()[0] == 9.0
    finally:
        wp.config.cpu_compiler_flags = old_flags
        wp.set_module_options({"cpu_compiler_flags": None})


def test_options_cpu_compiler_flags_native(test, device):
    """Compiling with cpu_compiler_flags="-march=native" should not crash."""
    if device.is_cuda:
        return

    old_flags = wp.config.cpu_compiler_flags
    try:
        wp.set_module_options({"cpu_compiler_flags": "-march=native"})

        x = wp.array([4.0], dtype=float, device=device)
        y = wp.zeros_like(x)
        wp.launch(scale, dim=1, inputs=[x, y], device=device)
        assert y.numpy()[0] == 16.0
    finally:
        wp.config.cpu_compiler_flags = old_flags
        wp.set_module_options({"cpu_compiler_flags": None})


devices = get_test_devices()


class TestOptions(unittest.TestCase):
    def test_set_module_options_via_runpy(self):
        """set_module_options/get_module_options should work when the calling module is run via runpy."""
        namespace = runpy.run_module("warp.tests.aux_test_options_runpy", run_name="__main__")
        self.assertTrue(namespace["_result"]["success"])
        self.assertFalse(namespace["_result"]["enable_backward"])

    def test_set_module_options_via_runpy_preimported(self):
        """set_module_options should target __main__ even when the module is already in sys.modules.

        When a launcher does ``runpy.run_module(mod, run_name="__main__")``,
        the module may already be imported under its qualified name.
        ``set_module_options`` must still target the ``__main__`` module
        (matching ``@wp.kernel``'s use of ``f.__module__``), not the
        pre-imported module.
        """
        mod_name = "warp.tests.aux_test_options_runpy"

        # Pre-import the module so it exists in sys.modules under its real name,
        # simulating what happens with ``python -m pkg.examples example_name``.
        pre_imported = importlib.import_module(mod_name)
        self.assertIn(mod_name, sys.modules)

        # Now run it via runpy with run_name="__main__", same as the launcher.
        namespace = runpy.run_module(mod_name, run_name="__main__")

        # The options must be set on "__main__", not on the pre-imported module.
        self.assertTrue(namespace["_result"]["success"])
        self.assertFalse(namespace["_result"]["enable_backward"])

        main_module = wp.get_module("__main__")
        self.assertFalse(main_module.options["enable_backward"])

    def test_get_caller_module_name_error_message(self):
        """_get_caller_module_name should raise RuntimeError with a helpful message when all fallbacks fail."""
        # Build a fake frame where all fallback steps fail:
        # - __name__ is None (not a normal module or __main__)
        # - __spec__ is None
        # - inspect.getmodule() returns None (patched)
        # - filename doesn't match any sys.modules entry
        fake_code = types.SimpleNamespace(co_filename="<nonexistent>")
        fake_frame = types.SimpleNamespace(
            f_globals={"__spec__": None, "__name__": None},
            f_code=fake_code,
        )

        with (
            patch("warp._src.context.inspect.getmodule", return_value=None),
            patch("warp._src.context.sys._getframe", return_value=fake_frame),
        ):
            with self.assertRaises(RuntimeError) as cm:
                _get_caller_module_name(stack_level=1)
            self.assertIn("Could not determine the calling module", str(cm.exception))


add_function_test(TestOptions, "test_options_backward_1", test_options_backward_1, devices=devices)
add_function_test(TestOptions, "test_options_backward_2", test_options_backward_2, devices=devices)
add_function_test(TestOptions, "test_options_backward_3", test_options_backward_3, devices=devices)
add_function_test(TestOptions, "test_options_backward_4", test_options_backward_4, devices=devices)
add_function_test(TestOptions, "test_options_opt_level", test_options_opt_level, devices=devices, check_output=False)
add_function_test(
    TestOptions, "test_options_cpu_compiler_flags_generic", test_options_cpu_compiler_flags_generic, devices=devices
)
add_function_test(
    TestOptions, "test_options_cpu_compiler_flags_native", test_options_cpu_compiler_flags_native, devices=devices
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
