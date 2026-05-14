# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Callable`` parameters in user-defined Warp functions.

These tests live outside ``test_func.py`` because ``Callable`` parameter support
needs a dedicated set of helper functions, kernels, module dependency checks,
specialization checks, and rejection-path coverage. Keep future ``Callable``
parameter tests in this module so ``test_func.py`` remains focused on general
``@wp.func`` behavior.
"""

import unittest
from collections.abc import Callable as CollectionsCallable
from typing import Any
from typing import Callable as TypingCallable  # noqa: UP035

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


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
def callable_apply_nested(x: float):
    return callable_apply_typing(callable_double_it, x)


@wp.func
def callable_forward_to_apply(g: TypingCallable, x: float):
    return callable_apply_typing(g, x)


@wp.kernel
def callable_func_parameter_kernel(out: wp.array(dtype=float)):
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
    out = wp.empty(10, dtype=float, device=device)

    wp.launch(callable_func_parameter_kernel, dim=1, outputs=[out], device=device)

    assert_np_equal(
        out.numpy(),
        np.array([6.0, 12.0, 10.0, 18.0, 14.0, 6.0, 24.0, 18.0, 30.0, 22.0], dtype=np.float32),
    )


CALLABLE_TARGET = callable_double_it


@wp.kernel
def callable_global_target_kernel(out: wp.array(dtype=float)):
    out[0] = callable_apply_typing(CALLABLE_TARGET, 3.0)


@wp.kernel
def callable_default_target_kernel(out: wp.array(dtype=float)):
    out[0] = callable_apply_default()


CALLABLE_DEPENDENCY_EXPLICIT_PROVIDER_MODULE = wp.Module("callable_dependency_explicit_provider")
CALLABLE_DEPENDENCY_EXPLICIT_CONSUMER_MODULE = wp.Module("callable_dependency_explicit_consumer")
CALLABLE_DEPENDENCY_DEFAULT_PROVIDER_MODULE = wp.Module("callable_dependency_default_provider")
CALLABLE_DEPENDENCY_DEFAULT_CONSUMER_MODULE = wp.Module("callable_dependency_default_consumer")


def callable_dependency_explicit_target(x: float):
    return x + 1.0


callable_dependency_explicit_target = wp.func(
    callable_dependency_explicit_target,
    module=CALLABLE_DEPENDENCY_EXPLICIT_PROVIDER_MODULE,
)


def callable_dependency_apply_explicit(g: TypingCallable, x: float):
    return g(x)


callable_dependency_apply_explicit = wp.func(
    callable_dependency_apply_explicit,
    module=CALLABLE_DEPENDENCY_EXPLICIT_CONSUMER_MODULE,
)


@wp.kernel(module=CALLABLE_DEPENDENCY_EXPLICIT_CONSUMER_MODULE)
def callable_dependency_explicit_kernel(out: wp.array(dtype=float)):
    out[0] = callable_dependency_apply_explicit(callable_dependency_explicit_target, 2.0)


def callable_dependency_default_target(x: float):
    return x + 1.0


callable_dependency_default_target = wp.func(
    callable_dependency_default_target,
    module=CALLABLE_DEPENDENCY_DEFAULT_PROVIDER_MODULE,
)


def callable_dependency_apply_default(g: TypingCallable = callable_dependency_default_target, x: float = 2.0):
    return g(x)


callable_dependency_apply_default = wp.func(
    callable_dependency_apply_default,
    module=CALLABLE_DEPENDENCY_DEFAULT_CONSUMER_MODULE,
)


@wp.kernel(module=CALLABLE_DEPENDENCY_DEFAULT_CONSUMER_MODULE)
def callable_dependency_default_kernel(out: wp.array(dtype=float)):
    out[0] = callable_dependency_apply_default()


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
    def test_callable_annotation_type_code(self):
        from warp._src.types import get_type_code  # noqa: PLC0415

        callable_annotations = (
            TypingCallable,
            CollectionsCallable,
            TypingCallable[[float], float],
            CollectionsCallable[[float], float],
        )

        for annotation in callable_annotations:
            with self.subTest(annotation=annotation):
                self.assertEqual(get_type_code(annotation), "c")

    def test_callable_argument_target_affects_module_hash(self):
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
        original_defaults = callable_apply_default.defaults.copy()
        try:
            callable_apply_default.defaults["g"] = callable_double_it
            double_hash = callable_default_target_kernel.module.hash_module()

            callable_apply_default.defaults["g"] = callable_triple_it
            triple_hash = callable_default_target_kernel.module.hash_module()
        finally:
            callable_apply_default.defaults = original_defaults

        self.assertNotEqual(double_hash, triple_hash)

    def test_callable_specialization_cache_not_shared_with_clone(self):
        from warp._src.codegen import specialize_callable_func  # noqa: PLC0415

        apply_overload = callable_forward_to_apply.user_overloads["c_f4"]
        specialized_apply = specialize_callable_func(apply_overload, {"g": callable_double_it})

        self.assertIn("_callable_specializations", apply_overload.__dict__)
        self.assertNotIn("_callable_specializations", specialized_apply.__dict__)

    def test_callable_specialized_adjoint_references_forwarded_target(self):
        from warp._src.codegen import specialize_callable_func  # noqa: PLC0415

        apply_overload = callable_forward_to_apply.user_overloads["c_f4"]
        specialized_apply = specialize_callable_func(apply_overload, {"g": callable_double_it})
        _, _, functions = specialized_apply.adj.get_references()

        self.assertIn(callable_apply_typing, functions)
        self.assertIn(callable_double_it, functions)

    def test_callable_specialization_preserves_return_annotation(self):
        func_module = wp.Module(f"callable_wrong_return_annotation_func_{id(self)}")
        kernel_module = wp.Module(f"callable_wrong_return_annotation_kernel_{id(self)}")

        @wp.func(module=func_module)
        def callable_wrong_return_annotation(g: TypingCallable, x: float) -> int:
            return g(x)

        @wp.kernel(module=kernel_module)
        def callable_wrong_return_annotation_kernel(out: wp.array(dtype=float)):
            out[0] = float(callable_wrong_return_annotation(callable_double_it, 2.0))

        out = wp.empty(1, dtype=float, device="cpu")

        with self.assertRaisesRegex(
            wp.WarpCodegenError,
            r"The function `callable_wrong_return_annotation` has its return type "
            r"annotated as `int` but the code returns a value of type `float32`.",
        ):
            wp.launch(callable_wrong_return_annotation_kernel, dim=1, outputs=[out], device="cpu")

    def test_callable_argument_target_updates_module_dependents(self):
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

    def test_callable_custom_grad_rejected(self):
        @wp.kernel(module="unique")
        def custom_grad_rejection_kernel(out: wp.array(dtype=float)):
            out[0] = callable_custom_grad_unsupported(callable_double_it, 2.0)

        @wp.kernel(module="unique")
        def custom_replay_rejection_kernel(out: wp.array(dtype=float)):
            out[0] = callable_custom_replay_unsupported(callable_double_it, 2.0)

        for kernel in (custom_grad_rejection_kernel, custom_replay_rejection_kernel):
            with self.subTest(kernel=kernel.key):
                out = wp.empty(1, dtype=float, device="cpu")

                with self.assertRaisesRegex(
                    wp.WarpCodegenError,
                    "Callable parameters.*custom gradients or replay functions",
                ):
                    wp.launch(kernel, dim=1, outputs=[out], device="cpu")

    def test_callable_builtin_target_rejected(self):
        @wp.kernel(module=wp.Module(f"callable_builtin_target_kernel_{id(self)}"))
        def callable_builtin_target_kernel(out: wp.array(dtype=float)):
            out[0] = callable_apply_typing(wp.sin, 0.5)

        out = wp.empty(1, dtype=float, device="cpu")

        with self.assertRaisesRegex(
            wp.WarpCodegenError,
            "Callable parameters currently require user-defined Warp functions",
        ):
            wp.launch(callable_builtin_target_kernel, dim=1, outputs=[out], device="cpu")


devices = get_test_devices()

add_function_test(
    TestFuncCallable,
    func=test_callable_func_parameter,
    name="test_callable_func_parameter",
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
