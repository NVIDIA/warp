# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``wp.Function`` parameters in user-defined Warp functions.

These tests live outside ``test_func.py`` because ``wp.Function`` parameter support
needs a dedicated set of helper functions, kernels, module dependency checks,
hashing checks, and rejection-path coverage. Keep future ``wp.Function``
parameter tests in this module so ``test_func.py`` remains focused on general
``@wp.func`` behavior.
"""

import unittest
from typing import Any
from unittest import mock

import numpy as np

import warp as wp
import warp.tests.aux_test_function_double as function_double_module
import warp.tests.aux_test_function_triple as function_triple_module
from warp.tests.unittest_utils import *


# User-facing helpers used by the runtime tests below. Keep these examples close
# to patterns a Warp user would write in kernels or user functions.
@wp.func
def function_double_it(x: float):
    return x * 2.0


@wp.func
def function_triple_it(x: float):
    return x * 3.0


@wp.func
def function_apply_unary(g: wp.Function, x: float):
    return g(x)


@wp.func
def function_apply_generic(g: wp.Function, x: Any):
    return g(x)


@wp.func
def function_apply_default(g: wp.Function = function_double_it, x: float = 3.0):
    return g(x)


@wp.func
def function_apply_builtin_default(g: wp.Function = wp.sin, x: float = 0.5):
    return g(x)


@wp.func
def function_apply_nested(x: float):
    return function_apply_unary(function_double_it, x)


@wp.func
def function_apply_binary(g: wp.Function, x: float, y: float):
    return g(x, y)


@wp.func
def function_apply_bound_parameter(g: wp.Function, x: float):
    f = g
    return f(x)


@wp.func
def function_square_it(x: float):
    return x * x


@wp.func
def function_apply_for_grad(g: wp.Function, x: float):
    return g(x)


@wp.func
def function_apply_builtin_for_grad(g: wp.Function, x: float):
    return g(x)


@wp.kernel
def function_parameter_kernel(out: wp.array[float]):
    out[0] = function_apply_unary(function_double_it, 3.0)
    out[1] = function_apply_unary(function_triple_it, 4.0)
    out[2] = function_apply_nested(7.0)
    out[3] = function_apply_default()
    out[4] = function_apply_unary(g=function_triple_it, x=8.0)
    out[5] = function_apply_generic(function_double_it, 11.0)


def test_function_parameter(test, device):
    """Verify Function parameters accept common user-facing call patterns."""
    out = wp.empty(6, dtype=float, device=device)

    wp.launch(function_parameter_kernel, dim=1, outputs=[out], device=device)

    assert_np_equal(
        out.numpy(),
        np.array([6.0, 12.0, 14.0, 6.0, 24.0, 22.0], dtype=np.float32),
    )


# ``wp.static()`` needs a global container so the kernel can resolve a function
# target statically while exercising the local-binding composition path.
FUNCTION_STATIC_TARGETS = {"double": function_double_it}


@wp.kernel
def function_parameter_local_binding_kernel(out: wp.array[float]):
    f = function_double_it
    out[0] = function_apply_unary(f, 3.0)

    out[1] = function_apply_bound_parameter(function_triple_it, 4.0)

    static_f = wp.static(FUNCTION_STATIC_TARGETS["double"])
    out[2] = function_apply_unary(static_f, 5.0)


def test_function_parameter_local_binding(test, device):
    """Verify Function parameters compose with function-valued local aliases."""
    out = wp.empty(3, dtype=float, device=device)

    wp.launch(function_parameter_local_binding_kernel, dim=1, outputs=[out], device=device)

    assert_np_equal(out.numpy(), np.array([6.0, 12.0, 10.0], dtype=np.float32))


@wp.kernel
def function_parameter_external_module_kernel(cond: wp.array[bool], out: wp.array[float]):
    i = wp.tid()
    if cond[i]:
        out[i] = function_apply_unary(function_double_module.function_external_module_double_it, 3.0)
    else:
        out[i] = function_apply_unary(function_triple_module.function_external_module_triple_it, 3.0)


def test_function_parameter_external_module(test, device):
    """Verify Function parameters accept targets imported from Python modules."""
    cond = wp.array([True, False], dtype=bool, device=device)
    out = wp.empty(2, dtype=float, device=device)

    wp.launch(function_parameter_external_module_kernel, dim=2, inputs=[cond], outputs=[out], device=device)

    assert_np_equal(out.numpy(), np.array([6.0, 9.0], dtype=np.float32))


@wp.kernel
def function_builtin_parameter_kernel(out: wp.array[float]):
    out[0] = function_apply_unary(wp.sin, 0.5)
    out[1] = function_apply_unary(wp.sqrt, 9.0)
    out[2] = function_apply_binary(wp.add, 2.0, 3.0)
    out[3] = function_apply_binary(wp.min, 7.0, 4.0)
    out[4] = function_apply_builtin_default()
    out[5] = function_apply_unary(g=wp.cos, x=0.0)


def test_function_builtin_parameter(test, device):
    """Verify Function parameters accept simple built-in Warp functions."""
    out = wp.empty(6, dtype=float, device=device)

    wp.launch(function_builtin_parameter_kernel, dim=1, outputs=[out], device=device)

    assert_np_equal(
        out.numpy(),
        np.array([np.sin(0.5), 3.0, 5.0, 4.0, np.sin(0.5), 1.0], dtype=np.float32),
    )


@wp.func
def function_write_after_callable(g: wp.Function, dst: wp.ref[wp.float32], x: wp.float32):
    dst = g(x)


@wp.kernel(enable_backward=False)
def function_ref_after_function_parameter_kernel(out: wp.array[wp.float32]):
    function_write_after_callable(function_double_it, out[0], wp.float32(3.0))
    function_write_after_callable(function_triple_it, out[1], wp.float32(4.0))


def test_function_parameter_before_ref_parameter(test, device):
    """Verify Function specialization preserves following wp.ref[T] arguments."""
    out = wp.empty(2, dtype=wp.float32, device=device)

    wp.launch(function_ref_after_function_parameter_kernel, dim=1, outputs=[out], device=device)

    assert_np_equal(out.numpy(), np.array([6.0, 12.0], dtype=np.float32))


# This global is intentionally mutated by
# ``test_function_argument_target_affects_module_hash``. The kernel reads it as
# a Function argument so the module hash must change when the target changes.
FUNCTION_TARGET = function_double_it


@wp.kernel
def function_global_target_kernel(out: wp.array[float]):
    out[0] = function_apply_unary(FUNCTION_TARGET, 3.0)


FUNCTION_BUILTIN_TARGET = wp.sin


@wp.kernel
def function_builtin_global_target_kernel(out: wp.array[float]):
    out[0] = function_apply_unary(FUNCTION_BUILTIN_TARGET, 0.5)


@wp.kernel
def function_default_target_kernel(out: wp.array[float]):
    out[0] = function_apply_default()


@wp.kernel
def function_builtin_default_target_kernel(out: wp.array[float]):
    out[0] = function_apply_builtin_default()


FUNCTION_OVERLOAD_TARGET = function_double_it


@wp.func
def function_apply_overloaded(x: int):
    return float(x)


@wp.func
def function_apply_overloaded(g: wp.Function, x: float):
    return g(x)


@wp.kernel
def function_overloaded_global_target_kernel(out: wp.array[float]):
    out[0] = function_apply_overloaded(FUNCTION_OVERLOAD_TARGET, 3.0)


@wp.func
def function_apply_overloaded_default(x: int):
    return float(x)


@wp.func
def function_apply_overloaded_default(x: float, g: wp.Function = function_double_it):
    return g(x)


@wp.kernel
def function_overloaded_default_target_kernel(out: wp.array[float]):
    out[0] = function_apply_overloaded_default(3.0)


@wp.kernel(enable_backward=False, module="unique")
def function_grad_kernel(out: wp.array[float]):
    out[0] = wp.grad(function_apply_for_grad)(function_square_it, 3.0)


@wp.kernel(enable_backward=False, module="unique")
def function_builtin_grad_kernel(out: wp.array[float]):
    out[0] = wp.grad(function_apply_builtin_for_grad)(wp.sin, 0.5)


@wp.func
def function_read_array(arr: wp.array[float], i: int):
    return arr[i]


@wp.func
def function_apply_array(g: wp.Function, arr: wp.array[float], i: int):
    return g(arr, i)


@wp.kernel(module="unique")
def function_array_read_kernel(arr: wp.array[float], out: wp.array[float]):
    out[0] = function_apply_array(function_read_array, arr, 0)


# These explicit modules are part of the behavior under test. Function targets
# from provider modules must be registered as dependencies of consumer modules so
# provider unloads invalidate stale consumer kernels.
FUNCTION_DEPENDENCY_EXPLICIT_PROVIDER_MODULE = wp.Module("function_dependency_explicit_provider")
FUNCTION_DEPENDENCY_EXPLICIT_CONSUMER_MODULE = wp.Module("function_dependency_explicit_consumer")
FUNCTION_DEPENDENCY_DEFAULT_PROVIDER_MODULE = wp.Module("function_dependency_default_provider")
FUNCTION_DEPENDENCY_DEFAULT_CONSUMER_MODULE = wp.Module("function_dependency_default_consumer")
FUNCTION_DEPENDENCY_LOCAL_PROVIDER_MODULE = wp.Module("function_dependency_local_provider")
FUNCTION_DEPENDENCY_LOCAL_CONSUMER_MODULE = wp.Module("function_dependency_local_consumer")
FUNCTION_DEPENDENCY_OVERLOAD_PROVIDER_MODULE = wp.Module("function_dependency_overload_provider")
FUNCTION_DEPENDENCY_OVERLOAD_CONSUMER_MODULE = wp.Module("function_dependency_overload_consumer")
FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_PROVIDER_MODULE = wp.Module("function_dependency_dynamic_overload_provider")
FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_CONSUMER_MODULE = wp.Module("function_dependency_dynamic_overload_consumer")
FUNCTION_DEPENDENCY_GRAD_PROVIDER_MODULE = wp.Module("function_dependency_grad_provider")
FUNCTION_DEPENDENCY_GRAD_CONSUMER_MODULE = wp.Module("function_dependency_grad_consumer")
FUNCTION_DEPENDENCY_EXTERNAL_CONSUMER_MODULE = wp.Module("function_dependency_external_consumer")


@wp.func(module=FUNCTION_DEPENDENCY_EXPLICIT_PROVIDER_MODULE)
def function_dependency_explicit_target(x: float):
    return x + 1.0


@wp.func(module=FUNCTION_DEPENDENCY_EXPLICIT_CONSUMER_MODULE)
def function_dependency_apply_explicit(g: wp.Function, x: float):
    return g(x)


@wp.kernel(module=FUNCTION_DEPENDENCY_EXPLICIT_CONSUMER_MODULE)
def function_dependency_explicit_kernel(out: wp.array[float]):
    out[0] = function_dependency_apply_explicit(function_dependency_explicit_target, 2.0)


@wp.func(module=FUNCTION_DEPENDENCY_DEFAULT_PROVIDER_MODULE)
def function_dependency_default_target(x: float):
    return x + 1.0


@wp.func(module=FUNCTION_DEPENDENCY_DEFAULT_CONSUMER_MODULE)
def function_dependency_apply_default(g: wp.Function = function_dependency_default_target, x: float = 2.0):
    return g(x)


@wp.kernel(module=FUNCTION_DEPENDENCY_DEFAULT_CONSUMER_MODULE)
def function_dependency_default_kernel(out: wp.array[float]):
    out[0] = function_dependency_apply_default()


@wp.func(module=FUNCTION_DEPENDENCY_LOCAL_PROVIDER_MODULE)
def function_dependency_local_target(x: float):
    return x + 1.0


@wp.func(module=FUNCTION_DEPENDENCY_LOCAL_CONSUMER_MODULE)
def function_dependency_apply_local(g: wp.Function, x: float):
    return g(x)


@wp.kernel(module=FUNCTION_DEPENDENCY_LOCAL_CONSUMER_MODULE)
def function_dependency_local_kernel(out: wp.array[float]):
    f = function_dependency_local_target
    out[0] = function_dependency_apply_local(f, 2.0)


@wp.func(module=FUNCTION_DEPENDENCY_OVERLOAD_PROVIDER_MODULE)
def function_dependency_overload_target(x: float):
    return x + 1.0


@wp.func(module=FUNCTION_DEPENDENCY_OVERLOAD_CONSUMER_MODULE)
def function_dependency_apply_overload(x: int):
    return float(x)


@wp.func(module=FUNCTION_DEPENDENCY_OVERLOAD_CONSUMER_MODULE)
def function_dependency_apply_overload(g: wp.Function, x: float):
    return g(x)


@wp.kernel(module=FUNCTION_DEPENDENCY_OVERLOAD_CONSUMER_MODULE)
def function_dependency_overload_kernel(out: wp.array[float]):
    out[0] = function_dependency_apply_overload(function_dependency_overload_target, 2.0)


@wp.func(module=FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_PROVIDER_MODULE)
def function_dependency_dynamic_overload_double(x: float):
    return x * 2.0


@wp.func(module=FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_PROVIDER_MODULE)
def function_dependency_dynamic_overload_triple(x: float):
    return x * 3.0


FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_TARGET = function_dependency_dynamic_overload_double


@wp.func(module=FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_CONSUMER_MODULE)
def function_dependency_apply_dynamic_overload(x: int):
    return float(x)


@wp.func(module=FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_CONSUMER_MODULE)
def function_dependency_apply_dynamic_overload(g: wp.Function, x: float):
    return g(x)


@wp.kernel(module=FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_CONSUMER_MODULE)
def function_dependency_dynamic_overload_kernel(vals: wp.array[float], out: wp.array[float]):
    out[0] = function_dependency_apply_dynamic_overload(FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_TARGET, vals[0])


@wp.func(module=FUNCTION_DEPENDENCY_GRAD_PROVIDER_MODULE)
def function_dependency_grad_square(x: float):
    return x * x


@wp.func(module=FUNCTION_DEPENDENCY_GRAD_PROVIDER_MODULE)
def function_dependency_grad_cube(x: float):
    return x * x * x


FUNCTION_DEPENDENCY_GRAD_TARGET = function_dependency_grad_square


@wp.func(module=FUNCTION_DEPENDENCY_GRAD_CONSUMER_MODULE)
def function_dependency_grad_apply(g: wp.Function, x: float):
    return g(x)


@wp.kernel(enable_backward=False, module=FUNCTION_DEPENDENCY_GRAD_CONSUMER_MODULE)
def function_dependency_grad_nested_kernel(out: wp.array[float]):
    out[0] = wp.grad(function_dependency_grad_apply)(FUNCTION_DEPENDENCY_GRAD_TARGET, 3.0)


@wp.kernel(module=FUNCTION_DEPENDENCY_EXTERNAL_CONSUMER_MODULE)
def function_dependency_external_module_kernel(cond: wp.array[bool], out: wp.array[float]):
    i = wp.tid()
    if cond[i]:
        out[i] = function_apply_unary(function_double_module.function_external_module_double_it, 3.0)
    else:
        out[i] = function_apply_unary(function_triple_module.function_external_module_triple_it, 3.0)


# These rejection fixtures live at module scope because custom grad and replay
# hooks are registered against a concrete ``@wp.func`` object.
@wp.func
def function_custom_grad_unsupported(g: wp.Function, x: float):
    return x


@wp.func_grad(function_custom_grad_unsupported)
def adj_function_custom_grad_unsupported(g: wp.Function, x: float, adj_ret: float):
    wp.adjoint[x] += adj_ret


@wp.func
def function_custom_replay_unsupported(g: wp.Function, x: float):
    return x


@wp.func_replay(function_custom_replay_unsupported)
def replay_function_custom_replay_unsupported(g: wp.Function, x: float):
    return x


class TestFuncParameterTargets(unittest.TestCase):
    def test_function_annotation_accepts_warp_function(self):
        """Verify ``wp.Function`` is the user-facing function annotation."""

        @wp.func(module="unique")
        def function_apply(g: wp.Function, x: float):
            return g(x)

        @wp.kernel(module="unique")
        def function_apply_kernel(out: wp.array[float]):
            out[0] = function_apply(function_double_it, 3.0)
            out[1] = function_apply(wp.sin, 0.5)

        out = wp.empty(2, dtype=float, device="cpu")

        wp.launch(function_apply_kernel, dim=1, outputs=[out], device="cpu")

        assert_np_equal(out.numpy(), np.array([6.0, np.sin(0.5)], dtype=np.float32))

    def test_function_parameter_python_scope(self):
        """Verify Function parameters accept Warp function targets at Python scope."""
        actual = np.array(
            [
                function_apply_unary(function_square_it, 3.0),
                function_apply_unary(g=function_triple_it, x=8.0),
                function_apply_default(),
                function_apply_unary(wp.sin, 0.5),
                function_apply_nested(7.0),
                function_apply_generic(function_double_it, 11.0),
            ]
        )
        expected = np.array([9.0, 24.0, 6.0, np.sin(0.5), 14.0, 22.0])

        np.testing.assert_allclose(actual, expected)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^Error calling function 'function_apply_unary', no overload found",
        ):
            function_apply_unary(lambda x: x, 3.0)

    def test_function_parameter_python_scope_argument_count_mismatch(self):
        """Verify Python-scope argument count errors include useful context."""
        with mock.patch.object(function_apply_unary, "get_overload", return_value=function_apply_nested):
            with self.assertRaisesRegex(
                RuntimeError,
                r"^Invalid number of arguments for function 'function_apply_unary', expected 1, got 2$",
            ):
                function_apply_unary(function_square_it, 3.0)

    def test_function_argument_target_affects_module_hash(self):
        """Verify explicit Function targets participate in module hashes."""
        global FUNCTION_TARGET

        original_target = FUNCTION_TARGET
        try:
            FUNCTION_TARGET = function_double_it
            double_hash = function_global_target_kernel.module.hash_module()

            FUNCTION_TARGET = function_triple_it
            triple_hash = function_global_target_kernel.module.hash_module()
        finally:
            FUNCTION_TARGET = original_target

        self.assertNotEqual(double_hash, triple_hash)

    def test_function_default_target_affects_module_hash(self):
        """Verify default Function targets participate in module hashes."""
        original_defaults = function_apply_default.defaults.copy()
        try:
            function_apply_default.defaults["g"] = function_double_it
            double_hash = function_default_target_kernel.module.hash_module()

            function_apply_default.defaults["g"] = function_triple_it
            triple_hash = function_default_target_kernel.module.hash_module()
        finally:
            function_apply_default.defaults = original_defaults

        self.assertNotEqual(double_hash, triple_hash)

    def test_function_builtin_argument_target_affects_module_hash(self):
        """Verify changing a global built-in Function target changes the module hash."""
        global FUNCTION_BUILTIN_TARGET

        original_target = FUNCTION_BUILTIN_TARGET
        try:
            FUNCTION_BUILTIN_TARGET = wp.sin
            sin_hash = function_builtin_global_target_kernel.module.hash_module()

            FUNCTION_BUILTIN_TARGET = wp.cos
            cos_hash = function_builtin_global_target_kernel.module.hash_module()
        finally:
            FUNCTION_BUILTIN_TARGET = original_target

        self.assertNotEqual(sin_hash, cos_hash)

    def test_function_builtin_default_target_affects_module_hash(self):
        """Verify changing a default built-in Function target changes the module hash."""
        original_defaults = function_apply_builtin_default.defaults.copy()
        try:
            function_apply_builtin_default.defaults["g"] = wp.sin
            sin_hash = function_builtin_default_target_kernel.module.hash_module()

            function_apply_builtin_default.defaults["g"] = wp.cos
            cos_hash = function_builtin_default_target_kernel.module.hash_module()
        finally:
            function_apply_builtin_default.defaults = original_defaults

        self.assertNotEqual(sin_hash, cos_hash)

    def test_function_overload_target_affects_module_hash(self):
        """Verify changing a non-primary function overload target changes the module hash."""
        global FUNCTION_OVERLOAD_TARGET

        original_target = FUNCTION_OVERLOAD_TARGET
        try:
            FUNCTION_OVERLOAD_TARGET = function_double_it
            double_hash = function_overloaded_global_target_kernel.module.hash_module()

            FUNCTION_OVERLOAD_TARGET = function_triple_it
            triple_hash = function_overloaded_global_target_kernel.module.hash_module()
        finally:
            FUNCTION_OVERLOAD_TARGET = original_target

        self.assertNotEqual(double_hash, triple_hash)

    def test_function_overload_default_target_affects_module_hash(self):
        """Verify changing a non-primary function overload default changes the module hash."""
        function_overload = function_apply_overloaded_default.get_overload([float], {})
        original_defaults = function_overload.defaults.copy()
        try:
            function_overload.defaults["g"] = function_double_it
            double_hash = function_overloaded_default_target_kernel.module.hash_module()

            function_overload.defaults["g"] = function_triple_it
            triple_hash = function_overloaded_default_target_kernel.module.hash_module()
        finally:
            function_overload.defaults = original_defaults

        self.assertNotEqual(double_hash, triple_hash)

    def test_function_dynamic_overload_target_affects_module_hash(self):
        """Verify dynamic overload selectors keep Function targets in the module hash."""
        global FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_TARGET

        original_target = FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_TARGET
        try:
            FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_TARGET = function_dependency_dynamic_overload_double
            double_hash = FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_CONSUMER_MODULE.hash_module()

            FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_TARGET = function_dependency_dynamic_overload_triple
            triple_hash = FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_CONSUMER_MODULE.hash_module()
        finally:
            FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_TARGET = original_target

        self.assertNotEqual(double_hash, triple_hash)

    def test_function_grad_target_affects_module_hash(self):
        """Verify Function targets passed through wp.grad() affect the module hash."""
        global FUNCTION_DEPENDENCY_GRAD_TARGET

        original_target = FUNCTION_DEPENDENCY_GRAD_TARGET
        try:
            FUNCTION_DEPENDENCY_GRAD_TARGET = function_dependency_grad_square
            square_hash = FUNCTION_DEPENDENCY_GRAD_CONSUMER_MODULE.hash_module()

            FUNCTION_DEPENDENCY_GRAD_TARGET = function_dependency_grad_cube
            cube_hash = FUNCTION_DEPENDENCY_GRAD_CONSUMER_MODULE.hash_module()
        finally:
            FUNCTION_DEPENDENCY_GRAD_TARGET = original_target

        self.assertNotEqual(square_hash, cube_hash)

    def test_function_grad_call(self):
        """Verify wp.grad() specializes functions with Function targets."""

        out = wp.empty(1, dtype=float, device="cpu")

        wp.launch(function_grad_kernel, dim=1, outputs=[out], device="cpu")

        assert_np_equal(out.numpy(), np.array([6.0], dtype=np.float32))

    def test_function_builtin_grad_call(self):
        """Verify wp.grad() specializes functions with built-in Function targets."""

        out = wp.empty(1, dtype=float, device="cpu")

        wp.launch(function_builtin_grad_kernel, dim=1, outputs=[out], device="cpu")

        assert_np_equal(out.numpy(), np.array([np.cos(0.5)], dtype=np.float32))

    def test_function_target_array_read_tracks_access(self):
        """Verify Function target array reads propagate to tape access tracking."""

        original = wp.config.verify_autograd_array_access
        wp.config.verify_autograd_array_access = True
        try:
            arr = wp.array([2.0], dtype=float, device="cpu")
            out = wp.empty(1, dtype=float, device="cpu")

            with wp.Tape():
                wp.launch(function_array_read_kernel, dim=1, inputs=[arr], outputs=[out], device="cpu")

            self.assertTrue(arr._is_read)
        finally:
            wp.config.verify_autograd_array_access = original

    def test_function_wrong_return_annotation_reports_error(self):
        """Verify Function calls report annotated return type errors."""

        @wp.func
        def function_wrong_return_annotation(g: wp.Function, x: float) -> int:
            return g(x)

        @wp.kernel(module="unique")
        def function_wrong_return_annotation_kernel(out: wp.array[float]):
            out[0] = float(function_wrong_return_annotation(function_double_it, 2.0))

        out = wp.empty(1, dtype=float, device="cpu")

        with self.assertRaisesRegex(
            wp.WarpCodegenError,
            r"The function `function_wrong_return_annotation` has its return type "
            r"annotated as `int` but the code returns a value of type `float32`.",
        ):
            wp.launch(function_wrong_return_annotation_kernel, dim=1, outputs=[out], device="cpu")

    def test_function_argument_target_updates_module_dependents(self):
        """Verify Function targets register provider modules as dependencies.

        Explicit arguments, default arguments, and kernel-local aliases exercise
        the paths where function targets can otherwise be missed during module
        reference discovery.
        """

        def unload_recursive(module, visited):
            module.unload()
            visited.add(module)
            for dependent in module.dependents:
                if dependent not in visited:
                    unload_recursive(dependent, visited)

        def no_inputs():
            return []

        cases = (
            (
                "explicit",
                FUNCTION_DEPENDENCY_EXPLICIT_PROVIDER_MODULE,
                FUNCTION_DEPENDENCY_EXPLICIT_CONSUMER_MODULE,
                function_dependency_explicit_kernel,
                no_inputs,
                3.0,
            ),
            (
                "default",
                FUNCTION_DEPENDENCY_DEFAULT_PROVIDER_MODULE,
                FUNCTION_DEPENDENCY_DEFAULT_CONSUMER_MODULE,
                function_dependency_default_kernel,
                no_inputs,
                3.0,
            ),
            (
                "local",
                FUNCTION_DEPENDENCY_LOCAL_PROVIDER_MODULE,
                FUNCTION_DEPENDENCY_LOCAL_CONSUMER_MODULE,
                function_dependency_local_kernel,
                no_inputs,
                3.0,
            ),
            (
                "overload",
                FUNCTION_DEPENDENCY_OVERLOAD_PROVIDER_MODULE,
                FUNCTION_DEPENDENCY_OVERLOAD_CONSUMER_MODULE,
                function_dependency_overload_kernel,
                no_inputs,
                3.0,
            ),
            (
                "dynamic overload",
                FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_PROVIDER_MODULE,
                FUNCTION_DEPENDENCY_DYNAMIC_OVERLOAD_CONSUMER_MODULE,
                function_dependency_dynamic_overload_kernel,
                lambda: [wp.array([2.0], dtype=float, device="cpu")],
                4.0,
            ),
            (
                "grad",
                FUNCTION_DEPENDENCY_GRAD_PROVIDER_MODULE,
                FUNCTION_DEPENDENCY_GRAD_CONSUMER_MODULE,
                function_dependency_grad_nested_kernel,
                no_inputs,
                6.0,
            ),
        )

        for name, provider_module, consumer_module, kernel, make_inputs, expected in cases:
            with self.subTest(name=name):
                out = wp.empty(1, dtype=float, device="cpu")
                wp.launch(kernel, dim=1, inputs=make_inputs(), outputs=[out], device="cpu")

                assert_np_equal(out.numpy(), np.array([expected], dtype=np.float32))
                self.assertIn(provider_module, consumer_module.references)
                self.assertIn(consumer_module, provider_module.dependents)
                self.assertTrue(consumer_module.hashers)

                unload_recursive(provider_module, visited=set())

                self.assertFalse(consumer_module.hashers)

    def test_function_external_module_targets_update_dependents(self):
        """Verify module-qualified Function targets register provider modules."""

        def unload_recursive(module, visited):
            module.unload()
            visited.add(module)
            for dependent in module.dependents:
                if dependent not in visited:
                    unload_recursive(dependent, visited)

        cond = wp.array([True, False], dtype=bool, device="cpu")
        out = wp.empty(2, dtype=float, device="cpu")

        wp.launch(function_dependency_external_module_kernel, dim=2, inputs=[cond], outputs=[out], device="cpu")

        assert_np_equal(out.numpy(), np.array([6.0, 9.0], dtype=np.float32))

        consumer_module = FUNCTION_DEPENDENCY_EXTERNAL_CONSUMER_MODULE
        provider_modules = (
            function_double_module.function_external_module_double_it.module,
            function_triple_module.function_external_module_triple_it.module,
        )

        for provider_module in provider_modules:
            self.assertIn(provider_module, consumer_module.references)
            self.assertIn(consumer_module, provider_module.dependents)

        self.assertTrue(consumer_module.hashers)

        unload_recursive(provider_modules[0], visited=set())

        self.assertFalse(consumer_module.hashers)

    def test_function_custom_grad_rejected(self):
        """Verify Function-specialized functions reject custom grad and replay hooks."""

        @wp.kernel(module="unique")
        def custom_grad_rejection_kernel(out: wp.array[float]):
            out[0] = function_custom_grad_unsupported(function_double_it, 2.0)

        @wp.kernel(module="unique")
        def custom_replay_rejection_kernel(out: wp.array[float]):
            out[0] = function_custom_replay_unsupported(function_double_it, 2.0)

        for kernel in (custom_grad_rejection_kernel, custom_replay_rejection_kernel):
            with self.subTest(kernel=kernel.key):
                out = wp.empty(1, dtype=float, device="cpu")

                with self.assertRaisesRegex(
                    wp.WarpCodegenError,
                    "Function parameters.*custom gradients or replay functions",
                ):
                    wp.launch(kernel, dim=1, outputs=[out], device="cpu")

    def test_function_non_regular_builtin_target_rejected(self):
        """Verify Function parameters reject built-ins that need special dispatch."""

        @wp.kernel(module="unique")
        def function_non_regular_builtin_target_kernel(out: wp.array[float]):
            out[0] = function_apply_unary(wp.printf, 0.5)

        out = wp.empty(1, dtype=float, device="cpu")

        with self.assertRaisesRegex(
            wp.WarpCodegenError,
            "unsupported built-in function 'printf'",
        ):
            wp.launch(function_non_regular_builtin_target_kernel, dim=1, outputs=[out], device="cpu")


devices = get_test_devices()

add_function_test(
    TestFuncParameterTargets,
    func=test_function_parameter,
    name="test_function_parameter",
    devices=devices,
)
add_function_test(
    TestFuncParameterTargets,
    func=test_function_parameter_local_binding,
    name="test_function_parameter_local_binding",
    devices=devices,
)
add_function_test(
    TestFuncParameterTargets,
    func=test_function_parameter_external_module,
    name="test_function_parameter_external_module",
    devices=devices,
)
add_function_test(
    TestFuncParameterTargets,
    func=test_function_builtin_parameter,
    name="test_function_builtin_parameter",
    devices=devices,
)
add_function_test(
    TestFuncParameterTargets,
    func=test_function_parameter_before_ref_parameter,
    name="test_function_parameter_before_ref_parameter",
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
