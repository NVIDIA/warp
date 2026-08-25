# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp._src.codegen as codegen
from warp._src.context import ModuleBuilder
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


def module_source(kernel, device: str) -> str:
    """Generate the full module source, including @wp.func bodies, for a kernel's module."""
    options = kernel.module.resolve_options(wp.config) | kernel.options
    return ModuleBuilder(kernel.module, options).codegen(device)


@wp.func(noinline=True)
def scale_out_of_line(x: float) -> float:
    return x * 2.0


@wp.kernel(enable_backward=False, module="unique")
def apply_scale(a: wp.array[float]):
    tid = wp.tid()
    a[tid] = scale_out_of_line(float(tid))


def test_noinline_func_runs(test, device):
    """Run a kernel calling a function marked noinline."""
    values = wp.empty(128, dtype=float, device=device)
    wp.launch(apply_scale, dim=values.shape, inputs=[values], device=device)
    assert_np_equal(values.numpy(), np.arange(128, dtype=np.float32) * 2.0)


class TestFuncInline(unittest.TestCase):
    def test_invalid_inline_hint_rejected(self):
        """Reject non-bool inline hints."""
        for hint in ("noinline", "forceinline"):
            for value in (1, 1.5, "yes"):
                with self.subTest(hint=hint, value=value), self.assertRaisesRegex(TypeError, hint):

                    @wp.func(**{hint: value}, module="unique")
                    def invalid(x: float) -> float:
                        return x

    def test_conflicting_inline_hints_rejected(self):
        """Reject noinline and forceinline on the same function."""
        with self.assertRaisesRegex(ValueError, "noinline and forceinline"):

            @wp.func(noinline=True, forceinline=True, module="unique")
            def invalid(x: float) -> float:
                return x

    def test_codegen_emits_inline_attrs(self):
        """Emit the requested inline attribute for CPU and CUDA, and nothing when none is requested."""
        for hint, macro in ((None, None), ("noinline", "WP_NOINLINE"), ("forceinline", "WP_FORCEINLINE")):

            @wp.func(**({hint: True} if hint else {}), module="unique")
            def helper(x: float) -> float:
                return x + 1.0

            @wp.kernel(enable_backward=False, module=helper.module)
            def uses_helper(a: wp.array[float]):
                a[wp.tid()] = helper(1.0)

            for device in ("cpu", "cuda"):
                with self.subTest(device=device, hint=hint):
                    # NOTE: match on each macro's use, not its name -- the module header
                    # #defines both regardless of whether any function requested them.
                    source = module_source(uses_helper, device)
                    emitted = [m for m in ("WP_NOINLINE", "WP_FORCEINLINE") if f"static {m} " in source]
                    self.assertEqual(emitted, [macro] if macro else [])

    def test_inline_attr_applied_to_adjoint(self):
        """Apply the inline attribute to the generated adjoint as well as the forward function."""

        @wp.func(noinline=True)
        def differentiable(x: float) -> float:
            return x * x

        @wp.kernel(enable_backward=True, module="unique")
        def uses_differentiable(a: wp.array[float]):
            a[wp.tid()] = differentiable(2.0)

        source = module_source(uses_differentiable, "cuda")
        self.assertIn("static WP_NOINLINE CUDA_CALLABLE ", source)
        self.assertIn("static WP_NOINLINE CUDA_CALLABLE void adj_", source)

    def test_inline_attr_applied_to_custom_grad_and_replay(self):
        """Apply the inline attribute to custom gradient and replay hooks, which are separate Functions."""

        @wp.func(noinline=True)
        def squared(x: float) -> float:
            return x * x

        @wp.func_grad(squared)
        def adj_squared(x: float, adj_ret: float):
            wp.adjoint[x] += 2.0 * x * adj_ret

        @wp.func_replay(squared)
        def replay_squared(x: float):  # no return annotation: it is matched against the forward args
            return x * x

        @wp.kernel(enable_backward=True, module="unique")
        def uses_squared(a: wp.array[float]):
            a[wp.tid()] = squared(3.0)

        self.assertEqual(squared.custom_grad_func.inline_hint, "noinline")
        self.assertEqual(squared.custom_replay_func.inline_hint, "noinline")

        source = module_source(uses_squared, "cuda")
        # Every definition generated for this function, custom hooks included, carries the hint.
        defs = [line for line in source.splitlines() if line.startswith("static") and "squared" in line]
        self.assertTrue(defs, "expected generated definitions for the hinted function")
        for line in defs:
            self.assertIn("WP_NOINLINE", line, f"missing inline attribute: {line}")

    def test_native_snippet_codegen_unaffected(self):
        """Keep generating native snippets, which share the function templates but take no hint."""
        snippet = "out[tid] = a * x[tid];"

        @wp.func_native(snippet)
        def scale(a: wp.float32, x: wp.array[wp.float32], out: wp.array[wp.float32], tid: int):  # fmt: skip
            ...

        @wp.kernel(module="unique")
        def uses_snippet(a: wp.float32, x: wp.array[wp.float32], out: wp.array[wp.float32]):
            tid = wp.tid()
            scale(a, x, out, tid)

        # Regression: the {inline_attr} slot must be filled here too, or codegen raises KeyError.
        self.assertIn(snippet, module_source(uses_snippet, "cuda"))

    def test_inline_macros_defined_in_headers(self):
        """Define the inline macros for both backends."""
        self.assertIn("#define WP_NOINLINE __attribute__((noinline))", codegen.cuda_module_header)
        self.assertIn("#define WP_FORCEINLINE __attribute__((always_inline))", codegen.cuda_module_header)
        self.assertIn("#define WP_NOINLINE __declspec(noinline)", codegen.cpu_module_header)
        self.assertIn("#define WP_NOINLINE __attribute__((noinline))", codegen.cpu_module_header)


devices = get_test_devices()
add_function_test(TestFuncInline, "test_noinline_func_runs", test_noinline_func_runs, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
