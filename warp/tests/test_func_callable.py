# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Callable`` parameters in user-defined Warp functions.

These tests live outside ``test_func.py`` because ``Callable`` parameter support
needs a dedicated set of helper functions, kernels, module dependency checks,
hashing checks, and rejection-path coverage. Keep future ``Callable``
parameter tests in this module so ``test_func.py`` remains focused on general
``@wp.func`` behavior.
"""

import unittest
from collections.abc import Callable as CollectionsCallable
from typing import Any
from typing import Callable as TypingCallable  # noqa: UP035

import numpy as np

import warp as wp
import warp.tests.aux_test_callable_double as callable_double_module
import warp.tests.aux_test_callable_triple as callable_triple_module
from warp.tests.unittest_utils import *


# User-facing helpers used by the runtime tests below. Keep these examples close
# to patterns a Warp user would write in kernels or user functions.
@wp.func
def callable_double_it(x: float):
    return x * 2.0


@wp.func
def callable_triple_it(x: float):
    return x * 3.0


@wp.func
def callable_apply_typing(g: TypingCallable, x: float):
    return g(x)


@wp.func
def callable_apply_collections(g: CollectionsCallable, x: float):
    return g(x)


@wp.func
def callable_apply_typing_parameterized(g: TypingCallable[[float], float], x: float):
    return g(x)


@wp.func
def callable_apply_collections_parameterized(g: CollectionsCallable[[float], float], x: float):
    return g(x)


@wp.func
def callable_apply_generic(g: TypingCallable, x: Any):
    return g(x)


@wp.func
def callable_apply_default(g: TypingCallable = callable_double_it, x: float = 3.0):
    return g(x)


@wp.func
def callable_apply_builtin_default(g: TypingCallable = wp.sin, x: float = 0.5):
    return g(x)


@wp.func
def callable_apply_nested(x: float):
    return callable_apply_typing(callable_double_it, x)


@wp.func
def callable_apply_binary(g: TypingCallable, x: float, y: float):
    return g(x, y)


@wp.func
def callable_apply_bound_parameter(g: TypingCallable, x: float):
    f = g
    return f(x)


@wp.func
def callable_square_it(x: float):
    return x * x


@wp.func
def callable_apply_for_grad(g: TypingCallable, x: float):
    return g(x)


@wp.func
def callable_apply_builtin_for_grad(g: TypingCallable, x: float):
    return g(x)


@wp.kernel
def callable_func_parameter_kernel(out: wp.array[float]):
    out[0] = callable_apply_typing(callable_double_it, 3.0)
    out[1] = callable_apply_typing(callable_triple_it, 4.0)
    out[2] = callable_apply_collections(callable_double_it, 5.0)
    out[3] = callable_apply_collections(callable_triple_it, 6.0)
    out[4] = callable_apply_nested(7.0)
    out[5] = callable_apply_default()
    out[6] = callable_apply_typing(g=callable_triple_it, x=8.0)
    out[7] = callable_apply_typing_parameterized(callable_double_it, 9.0)
    out[8] = callable_apply_collections_parameterized(callable_triple_it, 10.0)
    out[9] = callable_apply_generic(callable_double_it, 11.0)


def test_callable_func_parameter(test, device):
    """Verify Callable parameters accept common user-facing annotation forms."""
    out = wp.empty(10, dtype=float, device=device)

    wp.launch(callable_func_parameter_kernel, dim=1, outputs=[out], device=device)

    assert_np_equal(
        out.numpy(),
        np.array([6.0, 12.0, 10.0, 18.0, 14.0, 6.0, 24.0, 18.0, 30.0, 22.0], dtype=np.float32),
    )


# ``wp.static()`` needs a global container so the kernel can resolve a function
# target statically while exercising the local-binding composition path.
CALLABLE_STATIC_TARGETS = {"double": callable_double_it}


@wp.kernel
def callable_func_parameter_local_binding_kernel(out: wp.array[float]):
    f = callable_double_it
    out[0] = callable_apply_typing(f, 3.0)

    out[1] = callable_apply_bound_parameter(callable_triple_it, 4.0)

    static_f = wp.static(CALLABLE_STATIC_TARGETS["double"])
    out[2] = callable_apply_typing(static_f, 5.0)


def test_callable_func_parameter_local_binding(test, device):
    """Verify Callable parameters compose with function-valued local aliases."""
    out = wp.empty(3, dtype=float, device=device)

    wp.launch(callable_func_parameter_local_binding_kernel, dim=1, outputs=[out], device=device)

    assert_np_equal(out.numpy(), np.array([6.0, 12.0, 10.0], dtype=np.float32))


@wp.kernel
def callable_func_parameter_external_module_kernel(cond: wp.array(dtype=bool), out: wp.array(dtype=float)):
    i = wp.tid()
    if cond[i]:
        out[i] = callable_apply_typing(callable_double_module.callable_external_module_double_it, 3.0)
    else:
        out[i] = callable_apply_typing(callable_triple_module.callable_external_module_triple_it, 3.0)


def test_callable_func_parameter_external_module(test, device):
    """Verify Callable parameters accept targets imported from Python modules."""
    cond = wp.array([True, False], dtype=bool, device=device)
    out = wp.empty(2, dtype=float, device=device)

    wp.launch(callable_func_parameter_external_module_kernel, dim=2, inputs=[cond], outputs=[out], device=device)

    assert_np_equal(out.numpy(), np.array([6.0, 9.0], dtype=np.float32))


@wp.kernel
def callable_builtin_func_parameter_kernel(out: wp.array(dtype=float)):
    out[0] = callable_apply_typing(wp.sin, 0.5)
    out[1] = callable_apply_typing(wp.sqrt, 9.0)
    out[2] = callable_apply_binary(wp.add, 2.0, 3.0)
    out[3] = callable_apply_binary(wp.min, 7.0, 4.0)
    out[4] = callable_apply_builtin_default()
    out[5] = callable_apply_typing(g=wp.cos, x=0.0)


def test_callable_builtin_func_parameter(test, device):
    """Verify Callable parameters accept regular built-in Warp functions."""
    out = wp.empty(6, dtype=float, device=device)

    wp.launch(callable_builtin_func_parameter_kernel, dim=1, outputs=[out], device=device)

    assert_np_equal(
        out.numpy(),
        np.array([np.sin(0.5), 3.0, 5.0, 4.0, np.sin(0.5), 1.0], dtype=np.float32),
    )


# This global is intentionally mutated by
# ``test_callable_argument_target_affects_module_hash``. The kernel reads it as
# a Callable argument so the module hash must change when the target changes.
CALLABLE_TARGET = callable_double_it


@wp.kernel
def callable_global_target_kernel(out: wp.array[float]):
    out[0] = callable_apply_typing(CALLABLE_TARGET, 3.0)


@wp.kernel
def callable_default_target_kernel(out: wp.array[float]):
    out[0] = callable_apply_default()


@wp.kernel(enable_backward=False, module="unique")
def callable_grad_kernel(out: wp.array[float]):
    out[0] = wp.grad(callable_apply_for_grad)(callable_square_it, 3.0)


@wp.kernel(enable_backward=False, module="unique")
def callable_builtin_grad_kernel(out: wp.array[float]):
    out[0] = wp.grad(callable_apply_builtin_for_grad)(wp.sin, 0.5)


@wp.func
def callable_read_array(arr: wp.array(dtype=float), i: int):
    return arr[i]


@wp.func
def callable_apply_array(g: TypingCallable, arr: wp.array(dtype=float), i: int):
    return g(arr, i)


@wp.kernel(module="unique")
def callable_array_read_kernel(arr: wp.array(dtype=float), out: wp.array(dtype=float)):
    out[0] = callable_apply_array(callable_read_array, arr, 0)


# These explicit modules are part of the behavior under test. Callable targets
# from provider modules must be registered as dependencies of consumer modules so
# provider unloads invalidate stale consumer kernels.
CALLABLE_DEPENDENCY_EXPLICIT_PROVIDER_MODULE = wp.Module("callable_dependency_explicit_provider")
CALLABLE_DEPENDENCY_EXPLICIT_CONSUMER_MODULE = wp.Module("callable_dependency_explicit_consumer")
CALLABLE_DEPENDENCY_DEFAULT_PROVIDER_MODULE = wp.Module("callable_dependency_default_provider")
CALLABLE_DEPENDENCY_DEFAULT_CONSUMER_MODULE = wp.Module("callable_dependency_default_consumer")
CALLABLE_DEPENDENCY_LOCAL_PROVIDER_MODULE = wp.Module("callable_dependency_local_provider")
CALLABLE_DEPENDENCY_LOCAL_CONSUMER_MODULE = wp.Module("callable_dependency_local_consumer")
CALLABLE_DEPENDENCY_EXTERNAL_CONSUMER_MODULE = wp.Module("callable_dependency_external_consumer")


@wp.func(module=CALLABLE_DEPENDENCY_EXPLICIT_PROVIDER_MODULE)
def callable_dependency_explicit_target(x: float):
    return x + 1.0


@wp.func(module=CALLABLE_DEPENDENCY_EXPLICIT_CONSUMER_MODULE)
def callable_dependency_apply_explicit(g: TypingCallable, x: float):
    return g(x)


@wp.kernel(module=CALLABLE_DEPENDENCY_EXPLICIT_CONSUMER_MODULE)
def callable_dependency_explicit_kernel(out: wp.array[float]):
    out[0] = callable_dependency_apply_explicit(callable_dependency_explicit_target, 2.0)


@wp.func(module=CALLABLE_DEPENDENCY_DEFAULT_PROVIDER_MODULE)
def callable_dependency_default_target(x: float):
    return x + 1.0


@wp.func(module=CALLABLE_DEPENDENCY_DEFAULT_CONSUMER_MODULE)
def callable_dependency_apply_default(g: TypingCallable = callable_dependency_default_target, x: float = 2.0):
    return g(x)


@wp.kernel(module=CALLABLE_DEPENDENCY_DEFAULT_CONSUMER_MODULE)
def callable_dependency_default_kernel(out: wp.array[float]):
    out[0] = callable_dependency_apply_default()


@wp.func(module=CALLABLE_DEPENDENCY_LOCAL_PROVIDER_MODULE)
def callable_dependency_local_target(x: float):
    return x + 1.0


@wp.func(module=CALLABLE_DEPENDENCY_LOCAL_CONSUMER_MODULE)
def callable_dependency_apply_local(g: TypingCallable, x: float):
    return g(x)


@wp.kernel(module=CALLABLE_DEPENDENCY_LOCAL_CONSUMER_MODULE)
def callable_dependency_local_kernel(out: wp.array[float]):
    f = callable_dependency_local_target
    out[0] = callable_dependency_apply_local(f, 2.0)


@wp.kernel(module=CALLABLE_DEPENDENCY_EXTERNAL_CONSUMER_MODULE)
def callable_dependency_external_module_kernel(cond: wp.array(dtype=bool), out: wp.array(dtype=float)):
    i = wp.tid()
    if cond[i]:
        out[i] = callable_apply_typing(callable_double_module.callable_external_module_double_it, 3.0)
    else:
        out[i] = callable_apply_typing(callable_triple_module.callable_external_module_triple_it, 3.0)


# These rejection fixtures live at module scope because custom grad and replay
# hooks are registered against a concrete ``@wp.func`` object.
@wp.func
def callable_custom_grad_unsupported(g: TypingCallable, x: float):
    return x


@wp.func_grad(callable_custom_grad_unsupported)
def adj_callable_custom_grad_unsupported(g: TypingCallable, x: float, adj_ret: float):
    wp.adjoint[x] += adj_ret


@wp.func
def callable_custom_replay_unsupported(g: TypingCallable, x: float):
    return x


@wp.func_replay(callable_custom_replay_unsupported)
def replay_callable_custom_replay_unsupported(g: TypingCallable, x: float):
    return x


class TestFuncCallable(unittest.TestCase):
    def test_callable_argument_target_affects_module_hash(self):
        """Verify explicit Callable targets participate in module hashes."""
        global CALLABLE_TARGET

        original_target = CALLABLE_TARGET
        try:
            CALLABLE_TARGET = callable_double_it
            double_hash = callable_global_target_kernel.module.hash_module()

            CALLABLE_TARGET = callable_triple_it
            triple_hash = callable_global_target_kernel.module.hash_module()
        finally:
            CALLABLE_TARGET = original_target

        self.assertNotEqual(double_hash, triple_hash)

    def test_callable_default_target_affects_module_hash(self):
        """Verify default Callable targets participate in module hashes."""
        original_defaults = callable_apply_default.defaults.copy()
        try:
            callable_apply_default.defaults["g"] = callable_double_it
            double_hash = callable_default_target_kernel.module.hash_module()

            callable_apply_default.defaults["g"] = callable_triple_it
            triple_hash = callable_default_target_kernel.module.hash_module()
        finally:
            callable_apply_default.defaults = original_defaults

        self.assertNotEqual(double_hash, triple_hash)

    def test_callable_grad_call(self):
        """Verify wp.grad() specializes functions with Callable targets."""

        out = wp.empty(1, dtype=float, device="cpu")

        wp.launch(callable_grad_kernel, dim=1, outputs=[out], device="cpu")

        assert_np_equal(out.numpy(), np.array([6.0], dtype=np.float32))

    def test_callable_builtin_grad_call(self):
        """Verify wp.grad() specializes functions with built-in Callable targets."""

        out = wp.empty(1, dtype=float, device="cpu")

        wp.launch(callable_builtin_grad_kernel, dim=1, outputs=[out], device="cpu")

        assert_np_equal(out.numpy(), np.array([np.cos(0.5)], dtype=np.float32))

    def test_callable_target_array_read_tracks_access(self):
        """Verify Callable target array reads propagate to tape access tracking."""

        original = wp.config.verify_autograd_array_access
        wp.config.verify_autograd_array_access = True
        try:
            arr = wp.array([2.0], dtype=float, device="cpu")
            out = wp.empty(1, dtype=float, device="cpu")

            with wp.Tape():
                wp.launch(callable_array_read_kernel, dim=1, inputs=[arr], outputs=[out], device="cpu")

            self.assertTrue(arr._is_read)
        finally:
            wp.config.verify_autograd_array_access = original

    def test_callable_wrong_return_annotation_reports_error(self):
        """Verify Callable calls report annotated return type errors."""

        @wp.func
        def callable_wrong_return_annotation(g: TypingCallable, x: float) -> int:
            return g(x)

        @wp.kernel(module="unique")
        def callable_wrong_return_annotation_kernel(out: wp.array[float]):
            out[0] = float(callable_wrong_return_annotation(callable_double_it, 2.0))

        out = wp.empty(1, dtype=float, device="cpu")

        with self.assertRaisesRegex(
            wp.WarpCodegenError,
            r"The function `callable_wrong_return_annotation` has its return type "
            r"annotated as `int` but the code returns a value of type `float32`.",
        ):
            wp.launch(callable_wrong_return_annotation_kernel, dim=1, outputs=[out], device="cpu")

    def test_callable_argument_target_updates_module_dependents(self):
        """Verify Callable targets register provider modules as dependencies.

        Explicit arguments, default arguments, and kernel-local aliases exercise
        the paths where callable targets can otherwise be missed during module
        reference discovery.
        """

        def unload_recursive(module, visited):
            module.unload()
            visited.add(module)
            for dependent in module.dependents:
                if dependent not in visited:
                    unload_recursive(dependent, visited)

        cases = (
            (
                "explicit",
                CALLABLE_DEPENDENCY_EXPLICIT_PROVIDER_MODULE,
                CALLABLE_DEPENDENCY_EXPLICIT_CONSUMER_MODULE,
                callable_dependency_explicit_kernel,
            ),
            (
                "default",
                CALLABLE_DEPENDENCY_DEFAULT_PROVIDER_MODULE,
                CALLABLE_DEPENDENCY_DEFAULT_CONSUMER_MODULE,
                callable_dependency_default_kernel,
            ),
            (
                "local",
                CALLABLE_DEPENDENCY_LOCAL_PROVIDER_MODULE,
                CALLABLE_DEPENDENCY_LOCAL_CONSUMER_MODULE,
                callable_dependency_local_kernel,
            ),
        )

        for name, provider_module, consumer_module, kernel in cases:
            with self.subTest(name=name):
                out = wp.empty(1, dtype=float, device="cpu")
                wp.launch(kernel, dim=1, outputs=[out], device="cpu")

                assert_np_equal(out.numpy(), np.array([3.0], dtype=np.float32))
                self.assertIn(provider_module, consumer_module.references)
                self.assertIn(consumer_module, provider_module.dependents)
                self.assertTrue(consumer_module.hashers)

                unload_recursive(provider_module, visited=set())

                self.assertFalse(consumer_module.hashers)

    def test_callable_external_module_targets_update_dependents(self):
        """Verify module-qualified Callable targets register provider modules."""

        def unload_recursive(module, visited):
            module.unload()
            visited.add(module)
            for dependent in module.dependents:
                if dependent not in visited:
                    unload_recursive(dependent, visited)

        cond = wp.array([True, False], dtype=bool, device="cpu")
        out = wp.empty(2, dtype=float, device="cpu")

        wp.launch(callable_dependency_external_module_kernel, dim=2, inputs=[cond], outputs=[out], device="cpu")

        assert_np_equal(out.numpy(), np.array([6.0, 9.0], dtype=np.float32))

        consumer_module = CALLABLE_DEPENDENCY_EXTERNAL_CONSUMER_MODULE
        provider_modules = (
            callable_double_module.callable_external_module_double_it.module,
            callable_triple_module.callable_external_module_triple_it.module,
        )

        for provider_module in provider_modules:
            self.assertIn(provider_module, consumer_module.references)
            self.assertIn(consumer_module, provider_module.dependents)

        self.assertTrue(consumer_module.hashers)

        unload_recursive(provider_modules[0], visited=set())

        self.assertFalse(consumer_module.hashers)

    def test_callable_custom_grad_rejected(self):
        """Verify Callable-specialized functions reject custom grad and replay hooks."""

        @wp.kernel(module="unique")
        def custom_grad_rejection_kernel(out: wp.array[float]):
            out[0] = callable_custom_grad_unsupported(callable_double_it, 2.0)

        @wp.kernel(module="unique")
        def custom_replay_rejection_kernel(out: wp.array[float]):
            out[0] = callable_custom_replay_unsupported(callable_double_it, 2.0)

        for kernel in (custom_grad_rejection_kernel, custom_replay_rejection_kernel):
            with self.subTest(kernel=kernel.key):
                out = wp.empty(1, dtype=float, device="cpu")

                with self.assertRaisesRegex(
                    wp.WarpCodegenError,
                    "Callable parameters.*custom gradients or replay functions",
                ):
                    wp.launch(kernel, dim=1, outputs=[out], device="cpu")

    def test_callable_non_regular_builtin_target_rejected(self):
        """Verify Callable parameters reject built-ins that need special dispatch."""

        @wp.kernel(module="unique")
        def callable_non_regular_builtin_target_kernel(out: wp.array[float]):
            out[0] = callable_apply_typing(wp.printf, 0.5)

        out = wp.empty(1, dtype=float, device="cpu")

        with self.assertRaisesRegex(
            wp.WarpCodegenError,
            "unsupported built-in function 'printf'",
        ):
            wp.launch(callable_non_regular_builtin_target_kernel, dim=1, outputs=[out], device="cpu")


devices = get_test_devices()

add_function_test(
    TestFuncCallable,
    func=test_callable_func_parameter,
    name="test_callable_func_parameter",
    devices=devices,
)
add_function_test(
    TestFuncCallable,
    func=test_callable_func_parameter_local_binding,
    name="test_callable_func_parameter_local_binding",
    devices=devices,
)
add_function_test(
    TestFuncCallable,
    func=test_callable_func_parameter_external_module,
    name="test_callable_func_parameter_external_module",
    devices=devices,
)
add_function_test(
    TestFuncCallable,
    func=test_callable_builtin_func_parameter,
    name="test_callable_builtin_func_parameter",
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
