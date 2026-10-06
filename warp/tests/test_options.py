# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
import io
import runpy
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import warp as wp
from warp._src.context import _get_caller_module_name
from warp.tests.unittest_utils import *


@wp.kernel
def scale(
    x: wp.array[float],
    y: wp.array[float],
):
    y[0] = x[0] ** 2.0


@wp.kernel(enable_backward=True)
def scale_1(
    x: wp.array[float],
    y: wp.array[float],
):
    y[0] = x[0] ** 2.0


@wp.kernel(enable_backward=False)
def scale_2(
    x: wp.array[float],
    y: wp.array[float],
):
    y[0] = x[0] ** 2.0


@wp.func
def square(x: float):
    return x * x


@wp.kernel(enable_backward=True)
def scale_through_function(
    x: wp.array[float],
    y: wp.array[float],
):
    y[0] = square(x[0])


def test_options_backward_1(test, device):
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.set_module_options({"enable_backward": False})

    tape = wp.Tape()
    with tape:
        wp.launch(scale, dim=1, inputs=[x, y], device=device)

    with contextlib.redirect_stderr(io.StringIO()) as f:
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

    with contextlib.redirect_stderr(io.StringIO()) as f:
        tape.backward(y)

    expected = f"Warp UserWarning: Running the tape backwards may produce incorrect gradients because recorded kernel {scale_2.key} is configured with the option 'enable_backward=False'.\n"

    assert f.getvalue() == expected
    assert_np_equal(tape.gradients[x].numpy(), np.array(0.0))


def test_kernel_enable_backward_propagates_to_function(test, device):
    """Verify that a kernel-level backward override includes called Warp functions."""
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    old_enable_backward = wp.get_module_options()["enable_backward"]
    try:
        wp.set_module_options({"enable_backward": False})

        with wp.Tape() as tape:
            wp.launch(scale_through_function, dim=1, inputs=[x, y], device=device)

        tape.backward(y)
        assert_np_equal(tape.gradients[x].numpy(), np.array(6.0))
    finally:
        wp.set_module_options({"enable_backward": old_enable_backward})


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
    """Verify that compiling with cpu_compiler_flags="" (generic target) should not crash."""
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
    """Verify that compiling with cpu_compiler_flags="-march=native" should not crash."""
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


def test_options_opt_level_hash(test, device):
    """Verify that changing warp.config.optimization_level must change the module hash."""
    module = wp.get_module(__name__)

    # Ensure module option is None so the config value is used
    old_opt = module.options["optimization_level"]
    module.options["optimization_level"] = None
    module.hashers.clear()

    old_config = wp.config.optimization_level
    try:
        wp.config.optimization_level = None
        module.hashers.clear()
        hash_default = module.get_module_hash()

        wp.config.optimization_level = 2
        module.hashers.clear()
        hash_o2 = module.get_module_hash()

        wp.config.optimization_level = 3
        module.hashers.clear()
        hash_o3 = module.get_module_hash()

        # None is a distinct sentinel meaning "use target-specific default"
        # (O2 for CPU, O3 for CUDA), so it must differ from both explicit values.
        test.assertNotEqual(hash_default, hash_o2, "Hash must differ between None and explicit 2")
        test.assertNotEqual(hash_default, hash_o3, "Hash must differ between None and explicit 3")
        test.assertNotEqual(hash_o2, hash_o3, "Hash must differ between optimization levels 2 and 3")
    finally:
        wp.config.optimization_level = old_config
        module.options["optimization_level"] = old_opt
        module.hashers.clear()


def _make_cpu_llvm_options_kernel(flags: str | None, *, name: str | None = None):
    @wp.kernel(name=name, module="unique", module_options={"cpu_compiler_flags": flags})
    def exp_result(values: wp.array[wp.float32]):
        values[0] = wp.exp(values[0]) * 2.0 + 1.0

    return exp_result


def _run_cpu_llvm_options_kernel(kernel, x0: float = 0.5):
    values = wp.array([x0], dtype=wp.float32, device="cpu")
    wp.launch(kernel, dim=1, inputs=[values], device="cpu", block_dim=1)
    return values.numpy()[0]


devices = get_test_devices()


class TestOptions(unittest.TestCase):
    def test_cpu_llvm_option_forms_and_reset(self):
        wp.init()
        with patch.object(wp.config, "cache_kernels", False), patch.object(wp.config, "cpu_compiler_flags", None):
            baseline_kernel = _make_cpu_llvm_options_kernel(None)
            baseline = _run_cpu_llvm_options_kernel(baseline_kernel)
            tuned = [
                _make_cpu_llvm_options_kernel("-march=native -mllvm -limit-float-precision=6"),
                _make_cpu_llvm_options_kernel("-march=native -mllvm=-limit-float-precision=6"),
            ]
            for kernel in tuned:
                self.assertNotEqual(_run_cpu_llvm_options_kernel(kernel), baseline)
            self.assertNotEqual(tuned[0].module.get_module_hash(), tuned[1].module.get_module_hash())
            reset_kernel = _make_cpu_llvm_options_kernel(None, name="exp_result_reset")
            self.assertIsNot(reset_kernel.module, baseline_kernel.module)
            self.assertEqual(_run_cpu_llvm_options_kernel(reset_kernel), baseline)

    def test_cpu_llvm_invalid_option_recovers(self):
        wp.init()
        with patch.object(wp.config, "cache_kernels", False), patch.object(wp.config, "cpu_compiler_flags", None):
            baseline = _run_cpu_llvm_options_kernel(_make_cpu_llvm_options_kernel(None))
            invalid = _make_cpu_llvm_options_kernel("-mllvm=-warp-unknown-llvm-option")
            with self.assertRaisesRegex(Exception, "CPU kernel build failed"):
                invalid.module.load("cpu", block_dim=1)
            valid = _make_cpu_llvm_options_kernel("-march=native -mllvm=-inline-threshold=307")
            self.assertEqual(_run_cpu_llvm_options_kernel(valid), baseline)

    def test_cpu_llvm_concurrent_options(self):
        wp.init()
        with patch.object(wp.config, "cache_kernels", False), patch.object(wp.config, "cpu_compiler_flags", None):
            baseline_kernel = _make_cpu_llvm_options_kernel(None)
            baseline = _run_cpu_llvm_options_kernel(baseline_kernel)
            tuned = [
                _make_cpu_llvm_options_kernel("-mllvm=-limit-float-precision=5"),
                _make_cpu_llvm_options_kernel("-march=native -mllvm=-limit-float-precision=5"),
            ]
            exact = [
                _make_cpu_llvm_options_kernel("-march=native -mllvm=-inline-threshold=383"),
                _make_cpu_llvm_options_kernel(None, name="exp_result_concurrent_default"),
            ]
            self.assertIsNot(exact[1].module, baseline_kernel.module)
            with ThreadPoolExecutor(max_workers=len(tuned) + len(exact)) as executor:
                list(executor.map(lambda kernel: kernel.module.load("cpu", block_dim=1), tuned + exact))
            for kernel in tuned:
                self.assertNotEqual(_run_cpu_llvm_options_kernel(kernel), baseline)
            for kernel in exact:
                self.assertEqual(_run_cpu_llvm_options_kernel(kernel), baseline)

    def test_set_module_options_via_runpy(self):
        """Verify that set_module_options/get_module_options should work when the calling module is run via runpy."""
        namespace = runpy.run_module("warp.tests.aux_test_options_runpy", run_name="__main__")
        self.assertTrue(namespace["_result"]["success"])
        self.assertFalse(namespace["_result"]["enable_backward"])

    def test_set_module_options_via_runpy_preimported(self):
        """Verify that set_module_options should target __main__ even when the module is already in sys.modules.

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

    def test_cpu_target_output_name_differentiation(self):
        """Verify that CPU output filenames must distinguish LLVM and native ISA targets."""
        module = wp.get_module(__name__)
        device = wp.get_device("cpu")

        old_flags = module.options["cpu_compiler_flags"]
        try:
            with (
                patch("warp._src.context._get_cpu_feature_set", return_value=frozenset({"sse2"})),
                patch("warp._src.context._get_cpu_toolchain_version", return_value="22.1.8"),
            ):
                module.options["cpu_compiler_flags"] = ""
                name_llvm_22_1_8_portable = module._get_compile_output_name(device)

                module.options["cpu_compiler_flags"] = "-march=native"
                name_llvm_22_1_8_native = module._get_compile_output_name(device)

            with patch("warp._src.context._get_cpu_toolchain_version", return_value="22.1.9"):
                module.options["cpu_compiler_flags"] = ""
                name_llvm_22_1_9_portable = module._get_compile_output_name(device)

            self.assertNotEqual(name_llvm_22_1_8_portable, name_llvm_22_1_8_native)
            self.assertNotEqual(name_llvm_22_1_8_portable, name_llvm_22_1_9_portable)

            for output_name in (
                name_llvm_22_1_8_portable,
                name_llvm_22_1_8_native,
                name_llvm_22_1_9_portable,
            ):
                self.assertEqual(output_name.count(".cpu"), 1)
                self.assertRegex(output_name, r"\.cpu[0-9a-f]{8}\.o$")
        finally:
            module.options["cpu_compiler_flags"] = old_flags

    def test_cpu_isa_aot_warning(self):
        """Verify that compile_aot_module for CPU with -march=native must emit a portability warning."""
        module = wp.get_module(__name__)
        old_flags = wp.config.cpu_compiler_flags

        # Clear once-per-session warning dedup so the warning fires in this test
        saved_warnings = wp._src.logger._warnings_seen.copy()
        wp._src.logger._warnings_seen.clear()

        try:
            wp.config.cpu_compiler_flags = None  # resolves to -march=native
            module.hashers.clear()

            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                with tempfile.TemporaryDirectory() as tmpdir:
                    wp.compile_aot_module(module, device="cpu", module_dir=tmpdir)

            output = stderr_capture.getvalue()
            self.assertIn("-march=native", output)
            self.assertIn("cpu_compiler_flags=''", output)
        finally:
            wp.config.cpu_compiler_flags = old_flags
            module.hashers.clear()
            wp._src.logger._warnings_seen.update(saved_warnings)

    def test_cpu_isa_aot_warning_metadata(self):
        """Verify that compile_aot_module CPU -march=native warning must carry warning metadata."""
        module = wp.get_module(__name__)
        old_flags = wp.config.cpu_compiler_flags

        try:
            wp.config.cpu_compiler_flags = None  # resolves to -march=native
            module.hashers.clear()

            with (
                tempfile.TemporaryDirectory() as tmpdir,
                patch("warp._src.context.log_warning") as mock_log_warning,
                patch.object(module, "_compile"),
            ):
                wp.compile_aot_module(module, device="cpu", module_dir=tmpdir)

            warning_call = next(
                call
                for call in mock_log_warning.call_args_list
                if call.args
                and call.args[0].startswith("compile_aot_module: CPU module is being compiled with -march=native.")
            )
            self.assertIs(warning_call.kwargs.get("category"), UserWarning)
            self.assertEqual(warning_call.kwargs.get("stacklevel"), 2)
            self.assertIs(warning_call.kwargs.get("once"), True)
        finally:
            wp.config.cpu_compiler_flags = old_flags
            module.hashers.clear()

    def test_get_caller_module_name_error_message(self):
        """Verify that _get_caller_module_name should raise RuntimeError with a helpful message when all fallbacks fail."""
        # Build a fake frame where all fallback steps fail:
        # - __name__ is None (not a normal module or __main__)
        # - __spec__ is None
        # - inspect.getmodule() returns None (patched)
        # - filename doesn't match any sys.modules entry
        fake_code = types.SimpleNamespace(co_filename="<nonexistent>")
        fake_frame = types.SimpleNamespace(
            f_globals={"__spec__": None, "__name__": None},
            f_code=fake_code,
            f_back=None,
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
add_function_test(
    TestOptions,
    "test_kernel_enable_backward_propagates_to_function",
    test_kernel_enable_backward_propagates_to_function,
    devices=devices,
)
add_function_test(TestOptions, "test_options_opt_level", test_options_opt_level, devices=devices, check_output=False)
add_function_test(
    TestOptions, "test_options_cpu_compiler_flags_generic", test_options_cpu_compiler_flags_generic, devices=devices
)
add_function_test(
    TestOptions, "test_options_cpu_compiler_flags_native", test_options_cpu_compiler_flags_native, devices=devices
)
add_function_test(TestOptions, "test_options_opt_level_hash", test_options_opt_level_hash, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
