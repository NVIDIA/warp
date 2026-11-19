# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

import warp as wp
from warp._src.context import ModuleBuilder
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices, run_test_in_subprocess


def make_struct_query_kernel(dtype, inline):
    """Build a composed, data-dependent query with a by-value field handle."""
    vector = wp.types.vector(length=3, dtype=dtype)
    vector4 = wp.types.vector(length=4, dtype=dtype)
    matrix = wp.types.matrix(shape=(3, 3), dtype=dtype)
    matrix4 = wp.types.matrix(shape=(4, 4), dtype=dtype)

    @wp.struct
    class FieldState:
        keys: wp.array[wp.int64]
        coefficients: wp.array[dtype]
        cumulative: wp.array[dtype]
        linear: matrix
        translation: vector
        depth: int
        reference: bool
        transforms: wp.array[matrix4]
        has_transform: bool

    @wp.func
    def find_node(keys: wp.array[wp.int64], key: wp.int64):
        lower = int(0)
        upper = keys.shape[0]
        while lower < upper:
            middle = (lower + upper) // 2
            if keys[middle] < key:
                lower = middle + 1
            else:
                upper = middle
        if lower < keys.shape[0]:
            if keys[lower] == key:
                return lower
        return -1

    @wp.func
    def interpolate(keys: wp.array[wp.int64], coefficients: wp.array[dtype], point: vector, depth: int):
        count = 1 << depth
        scaled = point * dtype(count)
        cell = wp.vec3i(
            wp.clamp(int(wp.ceil(scaled[0])) - 1, 0, count - 1),
            wp.clamp(int(wp.ceil(scaled[1])) - 1, 0, count - 1),
            wp.clamp(int(wp.ceil(scaled[2])) - 1, 0, count - 1),
        )
        base = wp.vec3i(int(wp.floor(scaled[0])), int(wp.floor(scaled[1])), int(wp.floor(scaled[2])))
        fraction = scaled - vector(base)
        result = vector4(dtype(0))
        for corner in range(8):
            node = cell + wp.vec3i(corner // 4, (corner // 2) % 2, corner % 2)
            if wp.min(node - base) >= 0:
                key = (
                    (wp.int64(depth) << wp.int64(48))
                    | (wp.int64(node[0]) << wp.int64(32))
                    | (wp.int64(node[1]) << wp.int64(16))
                    | wp.int64(node[2])
                )
                index = find_node(keys, key)
                if index >= 0:
                    weight_x = wp.where(node[0] > base[0], fraction[0], dtype(1) - fraction[0])
                    weight_y = wp.where(node[1] > base[1], fraction[1], dtype(1) - fraction[1])
                    weight_z = wp.where(node[2] > base[2], fraction[2], dtype(1) - fraction[2])
                    sign_x = wp.where(node[0] > base[0], dtype(1), dtype(-1))
                    sign_y = wp.where(node[1] > base[1], dtype(1), dtype(-1))
                    sign_z = wp.where(node[2] > base[2], dtype(1), dtype(-1))
                    coefficient = coefficients[index]
                    result += vector4(
                        coefficient * weight_x * weight_y * weight_z,
                        coefficient * dtype(count) * sign_x * weight_y * weight_z,
                        coefficient * dtype(count) * weight_x * sign_y * weight_z,
                        coefficient * dtype(count) * weight_x * weight_y * sign_z,
                    )
        return result

    @wp.func
    def evaluate(
        keys: wp.array[wp.int64],
        coefficients: wp.array[dtype],
        cumulative: wp.array[dtype],
        point: vector,
        linear: matrix,
        translation: vector,
        depth: int,
    ):
        point = linear * point + translation
        valid = wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])
        valid = valid and wp.min(point) >= dtype(0) and wp.max(point) <= dtype(1)
        value = dtype(wp.nan)
        gradient = vector(dtype(wp.nan))
        if valid:
            result = interpolate(keys, coefficients, point, depth)
            value = result[0]
            gradient = wp.transpose(linear) * vector(result[1], result[2], result[3])
        return value, gradient, int(valid)

    @wp.func(inline=True)
    def field_query(field: FieldState, point: vector):
        return evaluate(
            field.keys, field.coefficients, field.cumulative, point, field.linear, field.translation, field.depth
        )

    @wp.func(inline=inline)
    def composed_query(field: FieldState, point: vector):
        value, gradient, _valid = field_query(field, point)
        return value + dtype(0.1) * wp.length_sq(gradient)

    @wp.kernel(module="unique", module_options={"fuse_fp": False})
    def composed_query_kernel(field: FieldState, points: wp.array[vector], output: wp.array[dtype]):
        index = wp.tid()
        output[index] = composed_query(field, points[index]) + composed_query(
            field, points[index] + vector(dtype(0.01), dtype(0.02), dtype(0.03))
        )

    return FieldState, composed_query_kernel, composed_query


def test_struct_query_composition(test, device):
    """Compose struct-valued queries and verify coefficient and point derivatives."""
    keys = [
        (depth << 48) | (x << 32) | (y << 16) | z
        for depth in range(3)
        for x in range((1 << depth) + 1)
        for y in range((1 << depth) + 1)
        for z in range((1 << depth) + 1)
    ]
    for dtype in (wp.float32, wp.float64):
        expected = None
        for inline in (None, False):
            field_type, kernel, _ = make_struct_query_kernel(dtype, inline)
            vector = wp.types.vector(length=3, dtype=dtype)
            field = field_type()
            field.keys = wp.array(keys, dtype=wp.int64, device=device)
            field.coefficients = wp.array(
                np.random.default_rng(23).normal(size=len(keys)), dtype=dtype, device=device, requires_grad=True
            )
            field.cumulative = wp.empty(0, dtype=dtype, device=device)
            field.transforms = wp.empty(0, dtype=wp.types.matrix(shape=(4, 4), dtype=dtype), device=device)
            field.linear = wp.types.matrix(shape=(3, 3), dtype=dtype)(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            field.depth = 2
            points = wp.array([[0.13, 0.21, 0.17], [0.31, 0.32, 0.41]], dtype=vector, device=device, requires_grad=True)
            output = wp.empty(len(points), dtype=dtype, device=device, requires_grad=True)
            with wp.Tape() as tape:
                wp.launch(kernel, len(points), inputs=[field, points, output], device=device)
            actual = output.numpy().copy()
            if expected is None:
                expected = actual
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            tape.backward(grads={output: wp.ones_like(output)})
            for array in (field.coefficients, points):
                original = array.numpy().copy()
                gradient = array.grad.numpy().copy()
                direction = np.random.default_rng(79).normal(size=original.shape).astype(original.dtype)
                epsilon = 1e-4 if dtype == wp.float32 else 1e-6
                losses = []
                for sign in (1, -1):
                    array.assign(original + sign * epsilon * direction)
                    wp.launch(kernel, len(points), inputs=[field, points, output], device=device)
                    losses.append(output.numpy().astype(float).sum())
                array.assign(original)
                np.testing.assert_allclose(
                    np.sum(gradient.astype(float) * direction),
                    (losses[0] - losses[1]) / (2 * epsilon),
                    rtol=1e-2 if dtype == wp.float32 else 1e-7,
                    atol=1e-2 if dtype == wp.float32 else 1e-8,
                )


def module_source(kernel, device: str) -> str:
    """Generate the full module source, including @wp.func bodies, for a kernel's module."""
    options = kernel.module.resolve_options(wp.config) | kernel.options
    return ModuleBuilder(kernel.module, options).codegen(device)


@wp.func(inline=False)
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
    def test_default_struct_inline_policy_and_cache(self):
        """Apply and hash the CUDA struct policy while honoring explicit hints."""
        for inline in (None, False, True):
            _, kernel, helper = make_struct_query_kernel(wp.float32, inline)
            for device in ("cpu", "cuda"):
                source = module_source(kernel, device)
                definitions = [
                    line
                    for line in source.splitlines()
                    if re.search(rf"\b(?:adj_)?{re.escape(helper.native_func)}\w*\($", line)
                ]
                self.assertTrue(definitions)
                expected = (
                    "WP_FORCEINLINE"
                    if inline is True or (inline is None and device == "cuda")
                    else "WP_NOINLINE"
                    if inline is False
                    else None
                )
                for definition in definitions:
                    self.assertEqual(
                        [macro for macro in ("WP_FORCEINLINE", "WP_NOINLINE") if macro in definition],
                        [expected] if expected else [],
                    )
            current_hash = kernel.module.hash_module()
            with mock.patch("warp._src.codegen.get_func_inline_hint", side_effect=lambda hint, types, device: hint):
                original_hash = kernel.module.hash_module()
            kernel.module.hash_module()
            if inline is None:
                self.assertNotEqual(current_hash, original_hash)
            else:
                self.assertEqual(current_hash, original_hash)

    def test_generic_struct_inline_policy_and_cache(self):
        """Invalidate cached specializations of generic functions taking structs."""

        @wp.struct
        class Parameters:
            scale: float

        @wp.func
        def generic_helper(parameters: Any, value: float):
            return parameters.scale * value

        @wp.kernel(module="unique")
        def generic_kernel(parameters: Parameters, output: wp.array[float]):
            output[wp.tid()] = generic_helper(parameters, 2.0)

        current_hash = generic_kernel.module.hash_module()
        with mock.patch("warp._src.codegen.get_func_inline_hint", side_effect=lambda hint, types, device: hint):
            original_hash = generic_kernel.module.hash_module()
        generic_kernel.module.hash_module()
        self.assertNotEqual(current_hash, original_hash)
        source = module_source(generic_kernel, "cuda")
        definitions = [
            line
            for line in source.splitlines()
            if re.search(rf"\b(?:adj_)?{re.escape(generic_helper.native_func)}\w*\($", line)
        ]
        self.assertTrue(definitions)
        for definition in definitions:
            self.assertIn("WP_FORCEINLINE", definition)

    def test_custom_grad_struct_inline_policy_and_cache(self):
        """Hash the policy when only the custom gradient takes a struct."""

        @wp.struct
        class OutputState:
            value: float

        @wp.func
        def make_output(value: float):
            result = OutputState()
            result.value = value
            return result

        @wp.func_grad(make_output)
        def make_output_grad(value: float, adj_result: OutputState):
            wp.adjoint[value] += adj_result.value

        @wp.kernel(module="unique")
        def output_kernel(values: wp.array[float], output: wp.array[float]):
            output[wp.tid()] = make_output(values[wp.tid()]).value

        current_hash = output_kernel.module.hash_module()
        with mock.patch("warp._src.codegen.get_func_inline_hint", side_effect=lambda hint, types, device: hint):
            original_hash = output_kernel.module.hash_module()
        output_kernel.module.hash_module()
        self.assertNotEqual(current_hash, original_hash)
        source = module_source(output_kernel, "cuda")
        definitions = [
            line
            for line in source.splitlines()
            if re.search(rf"\badj_{re.escape(make_output.native_func)}\w*\($", line)
        ]
        self.assertTrue(definitions)
        for definition in definitions:
            self.assertIn("WP_FORCEINLINE", definition)

    def test_invalid_inline_hint_rejected(self):
        """Reject non-bool values for the inline hint."""
        for value in (1, 1.5, "yes"):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "inline must be a bool"):

                @wp.func(inline=value, module="unique")
                def invalid(x: float) -> float:
                    return x

    def test_codegen_emits_inline_attrs(self):
        """Emit the requested inline attribute for CPU and CUDA, and nothing when none is requested."""
        for inline, macro in ((None, None), (False, "WP_NOINLINE"), (True, "WP_FORCEINLINE")):

            @wp.func(inline=inline, module="unique")
            def helper(x: float) -> float:
                return x + 1.0

            @wp.kernel(enable_backward=False, module=helper.module)
            def uses_helper(a: wp.array[float]):
                a[wp.tid()] = helper(1.0)

            for device in ("cpu", "cuda"):
                with self.subTest(device=device, inline=inline):
                    # Match the helper definition: CPU kernel wrappers can carry their own
                    # inline attributes, and the module header defines both macros.
                    source = module_source(uses_helper, device)
                    helper_name = re.compile(rf"\b(?:adj_)?{re.escape(helper.native_func)}\w*\($")
                    definitions = [line for line in source.splitlines() if helper_name.search(line)]
                    self.assertTrue(definitions, "expected generated definitions for the helper")
                    for definition in definitions:
                        emitted = [m for m in ("WP_NOINLINE", "WP_FORCEINLINE") if m in definition]
                        self.assertEqual(emitted, [macro] if macro else [], definition)

    def test_inline_attr_applied_to_adjoint(self):
        """Apply the inline attribute to the generated adjoint as well as the forward function."""

        @wp.func(inline=False)
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

        @wp.func(inline=False)
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

    @staticmethod
    def _compile_ptx(mode):
        """Compile a kernel calling one hinted-out-of-line and one hinted-inline helper, return its PTX."""

        @wp.func(inline=False)
        def kept_out_of_line(x: float, y: float) -> float:
            a = x * y + wp.sin(x)
            return wp.sqrt(wp.abs(a)) + wp.cos(y)

        @wp.func(inline=True)
        def forced_inline(x: float, y: float) -> float:
            a = x * y + wp.cos(x)
            return wp.sqrt(wp.abs(a)) + wp.sin(y)

        @wp.kernel(enable_backward=False, module="unique", module_options={"mode": mode})
        def uses_both(out: wp.array[float]):
            tid = wp.tid()
            out[tid] = kept_out_of_line(float(tid), 2.0) + forced_inline(float(tid), 3.0)

        # Compiling into a temporary directory keeps this self-contained: module headers are not
        # part of the module hash, so a shared kernel cache could serve PTX built with a
        # different macro definition.
        with tempfile.TemporaryDirectory() as module_dir:
            artifacts = wp.compile_aot_module(
                uses_both.module, device=None, arch=80, module_dir=module_dir, use_ptx=True
            )
            ptx_files = [p for p in artifacts if str(p).endswith(".ptx")]
            if not ptx_files:
                raise AssertionError(f"no PTX artifact produced: {artifacts}")
            with open(ptx_files[0]) as f:
                return f.read()

    @staticmethod
    def _out_of_line(ptx, name):
        """Whether ``name`` survives in the PTX as a real function rather than being inlined away."""
        defined = any(line.strip().startswith(".func") and name in line for line in ptx.splitlines())
        # A PTX call spans several lines: `call.uni (retval), <mangled name>, (params);`
        called = re.search(rf"call\.uni[^;]*?{name}", ptx, re.S) is not None
        return defined, called

    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA support")
    def test_noinline_honored_in_ptx(self):
        """Check ``inline=False`` keeps a helper out of line in an optimized build.

        Release mode is what makes this discriminating: the compiler inlines a small helper by
        choice, so an ignored attribute would leave no ``.func`` behind. In debug mode nothing is
        inlined anyway and the assertion would hold with or without the attribute.
        """
        ptx = self._compile_ptx("release")
        defined, called = self._out_of_line(ptx, "kept_out_of_line")
        self.assertTrue(defined, "inline=False helper was inlined away")
        self.assertTrue(called, "inline=False helper has no call site")

    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA support")
    def test_forceinline_honored_in_ptx(self):
        """Check ``inline=True`` inlines a helper in a debug build.

        Debug mode is what makes this discriminating: it disables heuristic inlining, so the
        compiler would leave the helper out of line unless the attribute is honored. In release
        mode it inlines small helpers anyway and the assertion would hold either way.
        """
        if run_test_in_subprocess(self):
            return

        ptx = self._compile_ptx("debug")
        defined, called = self._out_of_line(ptx, "forced_inline")
        self.assertFalse(defined, "inline=True helper was left out of line")
        self.assertFalse(called, "inline=True helper is still called out of line")

    def test_inline_macros_defined_in_builtin_header(self):
        """Define the inline macros for every backend in the shared native header."""
        header = (Path(wp.__file__).parent / "native" / "builtin.h").read_text()

        self.assertIn("#define WP_NOINLINE __declspec(noinline)", header)
        # newline-anchored: __forceinline is a prefix of __forceinline__, so a bare substring passes
        self.assertIn("#define WP_FORCEINLINE __forceinline\n", header)
        self.assertIn("#define WP_NOINLINE __attribute__((noinline))", header)
        self.assertIn("#define WP_FORCEINLINE inline __attribute__((always_inline))", header)


devices = get_test_devices()
add_function_test(TestFuncInline, "test_noinline_func_runs", test_noinline_func_runs, devices=devices)
add_function_test(TestFuncInline, "test_struct_query_composition", test_struct_query_composition, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
