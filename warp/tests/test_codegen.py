# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import functools
import importlib
import inspect
import linecache
import math
import os
import sys
import tempfile
import textwrap
import types
import unittest
from typing import Any
from unittest import mock

import warp as wp
from warp._src import codegen
from warp.tests import aux_test_extract_source_patterns as patterns
from warp.tests.aux_test_extract_source_patterns import contains_truncating_string
from warp.tests.unittest_utils import *

_KERNEL_RETURN_ERROR_PATTERN = (
    r"Warp kernels cannot return values\. Write results to output arguments, "
    r"and omit the return annotation or use `-> None`\."
)


@wp.kernel
def test_expect():
    a = 1.0
    a += 2.0

    wp.expect_eq(123, 123)
    wp.expect_neq(123, 234)

    wp.expect_eq(wp.vec2(1.0, 2.0), wp.vec2(1.0, 2.0))
    wp.expect_neq(wp.vec2(1.0, 2.0), wp.vec2(2.0, 3.0))

    wp.expect_eq(wp.mat22(1.0, 2.0, 3.0, 4.0), wp.mat22(1.0, 2.0, 3.0, 4.0))
    wp.expect_neq(wp.mat22(1.0, 2.0, 3.0, 4.0), wp.mat22(2.0, 3.0, 4.0, 5.0))


def parenthesized_multiline_lambda():
    # qd is intentionally unused to match a two-argument callback signature.
    # fmt: off
    return lambda q, qd: (
        q[0] == 0.0
        and q[1] == 0.0
    )
    # fmt: on


@wp.kernel
def test_rename():
    a = 0
    b = 1

    a = b
    a = 2

    wp.expect_eq(a, 2)
    wp.expect_eq(b, 1)


@wp.kernel
def test_inplace():
    a = 1.0
    a += 2.0

    wp.expect_eq(a, 3.0)


@wp.kernel
def test_constant(c: float):
    a = 0.0
    a = c + 1.0

    wp.expect_eq(a, 2.0)


@wp.kernel
def test_dynamic_for_rename(n: int):
    f0 = int(0.0)
    f1 = int(1.0)

    for _i in range(0, n):
        f = f0 + f1

        f0 = f1
        f1 = f

    wp.expect_eq(f1, 89)


@wp.kernel
def test_dynamic_for_inplace(n: int):
    a = float(0.0)

    for _i in range(0, n):
        a += 1.0

    wp.expect_eq(a, float(n))


@wp.kernel
def test_reassign():
    f0 = 1.0
    f1 = f0

    f1 = f1 + 2.0

    wp.expect_eq(f1, 3.0)
    wp.expect_eq(f0, 1.0)


@wp.kernel
def test_dynamic_reassign(n: int):
    f0 = wp.vec3()
    f1 = f0

    for _i in range(0, n):
        f1 = f1 - wp.vec3(2.0, 0.0, 0.0)

    wp.expect_eq(f1, wp.vec3(-4.0, 0.0, 0.0))
    wp.expect_eq(f0, wp.vec3())


@wp.kernel
def test_range_static_sum(result: wp.array[int]):
    a = int(0)
    for _i in range(10):
        a = a + 1

    b = int(0)
    for _i in range(0, 10):
        b = b + 1

    c = int(0)
    for _i in range(0, 20, 2):
        c = c + 1

    result[0] = a
    result[1] = b
    result[2] = c


@wp.kernel
def test_range_dynamic_sum(start: int, end: int, step: int, result: wp.array[int]):
    a = int(0)
    for _i in range(end):
        a = a + 1

    b = int(0)
    for _i in range(start, end):
        b = b + 1

    c = int(0)
    for _i in range(start, end * step, step):
        c = c + 1

    d = int(0)
    for _i in range(end * step, start, -step):
        d = d + 1

    result[0] = a
    result[1] = b
    result[2] = c
    result[3] = d


@wp.kernel
def test_range_dynamic(start: int, end: int, step: int, result: wp.array[int]):
    output = int(0)
    for i in range(start, end, step):
        result[output] = i
        output += 1


@wp.kernel
def test_range_dynamic_nested(n: int):
    sum1 = float(0.0)
    sum2 = float(0.0)
    sum3 = float(0.0)

    for _i in range(n):
        sum1 = sum1 + 1.0
        sum3 = sum3 + 1.0

        for _j in range(n):
            sum2 = sum2 + 1.0
            sum3 = sum3 + 1.0

        sum3 = sum3 + 1.0

    wp.expect_eq(sum1, float(n))
    wp.expect_eq(sum2, float(n * n))
    wp.expect_eq(sum3, float(n * n + 2 * n))


@wp.kernel
def test_while(n: int):
    i = int(0)

    while i < n:
        i = i + 1

    wp.expect_eq(i, n)


@wp.kernel
def test_pass(n: int):
    i = int(0)

    while i < n:
        if False:
            pass
        else:
            i = i + 1

    wp.expect_eq(i, n)


@wp.kernel
def test_break(n: int):
    a = int(0)

    for _i in range(0, n):
        if a == 5:
            break

        a += 1

    wp.expect_eq(a, 5)


@wp.kernel
def test_break_early(n: int):
    a = int(0)

    for i in range(0, n):
        if i > 5:
            a = 1
            break

    wp.expect_eq(a, 1)


@wp.kernel
def test_break_unroll():
    a = int(0)

    for i in range(0, 10):
        if i > 5:
            a = i
            break

    wp.expect_eq(a, 6)


@wp.kernel
def test_break_while():
    a = int(0)

    while a < 10:
        if a > 5:
            break
        a += 1

    wp.expect_eq(a, 6)


# while True with conditional break and observable side effects
@wp.kernel
def test_break_while_true():
    a = int(0)
    total = int(0)

    while True:
        total += a
        a += 1
        if a >= 5:
            break

    wp.expect_eq(a, 5)
    wp.expect_eq(total, 10)  # 0+1+2+3+4


# nested while True loops with conditional breaks
@wp.kernel
def test_break_while_true_nested():
    total = int(0)
    i = int(0)

    while True:
        j = int(0)
        while True:
            total += 1
            j += 1
            if j >= 3:
                break
        i += 1
        if i >= 4:
            break

    wp.expect_eq(total, 12)  # 4 * 3


@wp.kernel
def test_break_multiple(n: int):
    a = int(0)

    for i in range(0, n):
        if i == 6:
            a = 1
            break

        if i == 5:
            a = 2
            break

        if i == 7:
            a = 3
            break

    wp.expect_eq(a, 2)


@wp.kernel
def test_continue(n: int):
    a = int(0)

    for i in range(0, n):
        if i == 5:
            continue

        a += 1

    wp.expect_eq(a, n - 1)


@wp.kernel
def test_continue_unroll():
    a = int(0)

    for i in range(0, 10):
        if i == 5:
            continue

        a += 1

    wp.expect_eq(a, 9)


lower = wp.constant(-3)
upper = wp.constant(3)
step = wp.constant(2)


# test unrolling of loops with constant size params
# we can't easily test if unrolling has occurred
# so just verify correctness at this stage
@wp.kernel
def test_range_constant():
    s = 0
    for i in range(upper):
        s += i

    # sum [0, 3)
    wp.expect_eq(s, 3)

    s = 0
    for i in range(lower, upper):
        s += i

    # sum [-3, 3)
    wp.expect_eq(s, -3)

    s = 0
    for i in range(lower, upper, step):
        s += i

    # sum [-3, 3)
    wp.expect_eq(s, -3)


N = wp.constant(3)


# test a dynamic loop nested between loops expected to be unrolled.
@wp.kernel
def test_range_constant_dynamic_nested(m: int):
    s = int(0)
    for _i in range(N):
        for _k in range(m):
            for _j in range(N):
                s += 1

    wp.expect_eq(s, N * m * N)


@wp.kernel
def test_range_expression():
    idx = 1
    batch_size = 100

    a = wp.float(0.0)
    c = wp.float(1.0)

    # constant expression with a function
    for _i in range(4 * idx, wp.min(4 * idx + 4, batch_size)):
        a += c

    for _i in range(4 * idx, min(4 * idx + 4, batch_size)):
        a += c

    tid = wp.tid()

    # dynamic expression with a function
    for _i in range(4 * idx, wp.min(4 * idx, tid + 1000)):
        a += c

    for _i in range(4 * idx, min(4 * idx, tid + 1000)):
        a += c

    wp.expect_eq(a, 8.0)


def test_unresolved_func(test, device):
    # kernel with unresolved function must be in a separate module, otherwise the current module would fail to load
    # Import the bad fixture only for this test so it can be removed from
    # Warp's user module registry before later force-load checks.
    unresolved_func_module = importlib.import_module("warp.tests.aux_test_unresolved_func")
    unresolved_func_kernel = unresolved_func_module.unresolved_func_kernel

    # ensure that an appropriate exception is raised when the bad module gets loaded
    with test.assertRaisesRegex(AttributeError, "Could not find function wp.missing_func"):
        wp.launch(unresolved_func_kernel, dim=1, inputs=[], device=device)

    # remove all references to the bad module so that subsequent calls to wp.force_load()
    # won't try to load it unless we explicitly re-import it again
    del wp._src.context.user_modules["warp.tests.aux_test_unresolved_func"]
    del sys.modules["warp.tests.aux_test_unresolved_func"]


def test_unresolved_symbol(test, device):
    # kernel with unresolved symbol must be in a separate module, otherwise the current module would fail to load
    # Import the bad fixture only for this test so it can be removed from
    # Warp's user module registry before later force-load checks.
    unresolved_symbol_module = importlib.import_module("warp.tests.aux_test_unresolved_symbol")
    unresolved_symbol_kernel = unresolved_symbol_module.unresolved_symbol_kernel

    # ensure that an appropriate exception is raised when the bad module gets loaded
    with test.assertRaisesRegex(KeyError, "Referencing undefined symbol: missing_symbol"):
        wp.launch(unresolved_symbol_kernel, dim=1, inputs=[], device=device)

    # remove all references to the bad module so that subsequent calls to wp.force_load()
    # won't try to load it unless we explicitly re-import it again
    del wp._src.context.user_modules["warp.tests.aux_test_unresolved_symbol"]
    del sys.modules["warp.tests.aux_test_unresolved_symbol"]


def test_invalid_namespace_path(test, device):
    """Test that invalid namespace paths in kernel function calls are rejected."""

    # Test 1: Invalid intermediate namespace (wp.foo.bar.tid)
    def kernel_invalid_intermediate():
        tid = wp.foo.bar.tid()
        print(tid)

    kernel = wp.Kernel(func=kernel_invalid_intermediate)
    with test.assertRaisesRegex(AttributeError, r"`foo` is not an attribute of"):
        wp.launch(kernel, dim=1, device=device)

    # Test 2: Valid submodule but function doesn't exist there (wp.types.tid)
    def kernel_submodule_missing_func():
        tid = wp.types.tid()
        print(tid)

    kernel = wp.Kernel(func=kernel_submodule_missing_func)
    with test.assertRaisesRegex(AttributeError, r"Could not find function wp.types.tid"):
        wp.launch(kernel, dim=1, device=device)

    # Test 3: Missing function on warp module (wp.nonexistent_func)
    def kernel_missing_func():
        x = wp.nonexistent_func()
        print(x)

    kernel = wp.Kernel(func=kernel_missing_func)
    with test.assertRaisesRegex(AttributeError, r"Could not find function wp.nonexistent_func"):
        wp.launch(kernel, dim=1, device=device)


def test_error_global_var(test, device):
    arr = wp.array((1.0, 2.0, 3.0), dtype=float, device=device)

    def kernel_1_fn(out: wp.array[float]):
        out[0] = arr[0]

    def kernel_2_fn(out: wp.array[float]):
        out[0] = arr

    def kernel_3_fn(out: wp.array[float]):
        out[0] = wp.lower_bound(arr, 2.0)

    out = wp.empty_like(arr)

    kernel = wp.Kernel(func=kernel_1_fn)
    with test.assertRaisesRegex(TypeError, r"Invalid external reference type: <class 'warp._src.types.array'>"):
        wp.launch(kernel, dim=out.shape, inputs=(), outputs=(out,), device=device)

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(TypeError, r"Invalid external reference type: <class 'warp._src.types.array'>"):
        wp.launch(kernel, dim=out.shape, inputs=(), outputs=(out,), device=device)

    kernel = wp.Kernel(func=kernel_3_fn)
    with test.assertRaisesRegex(TypeError, r"Invalid external reference type: <class 'warp._src.types.array'>"):
        wp.launch(kernel, dim=out.shape, inputs=(), outputs=(out,), device=device)


def test_error_collection_construct(test, device):
    def kernel_1_fn():
        x = [1.0, 2.0, 3.0]

    def kernel_2_fn():
        x = {"a": 1.0, "b": 2.0, "c": 3.0}

    kernel = wp.Kernel(func=kernel_1_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"List constructs are not supported in kernels. Use vectors like `wp.vec3\(\)` for small fixed-size collections, or `wp.zeros\(shape=N, dtype=\.\.\.\)` for stack-allocated arrays.",
    ):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(RuntimeError, r"Construct `ast.Dict` not supported in kernels."):
        wp.launch(kernel, dim=1, device=device)


def test_error_unmatched_arguments(test, device):
    def kernel_1_fn():
        a = 1 * 1.0

    def kernel_2_fn():
        x = wp.dot(wp.vec2(1.0, 2.0), wp.vec2h(1.0, 2.0))

    kernel = wp.Kernel(func=kernel_1_fn)
    with test.assertRaisesRegex(RuntimeError, r"Input types must be the same, got \['int32', 'float32'\]"):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Input types must be exactly the same, got \['vec2f', 'vec2h'\]",
    ):
        wp.launch(kernel, dim=1, device=device)


def test_error_kernel_return_value(test, device):
    # kernels can return without a value
    @wp.kernel(module="unique")
    def f0(x: float):
        return

    wp.launch(f0, dim=1, inputs=[3.0], device=device)

    # kernels can explicitly annotate that they return nothing
    @wp.kernel(module="unique")
    def f0_none(x: float) -> None:
        return

    wp.launch(f0_none, dim=1, inputs=[3.0], device=device)

    # Python's NoneType spelling is equivalent to a None return annotation
    @wp.kernel(module="unique")
    def f0_none_type(x: float) -> types.NoneType:
        return

    wp.launch(f0_none_type, dim=1, inputs=[3.0], device=device)

    # return None is still a value-returning statement and is not valid in kernels
    @wp.kernel(module="unique")
    def f0_return_none(x: float):
        return None

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(f0_return_none, dim=1, inputs=[3.0], device=device)

    @wp.kernel(module="unique")
    def f0_none_return_none(x: float) -> None:
        return None

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(f0_none_return_none, dim=1, inputs=[3.0], device=device)

    # kernels can't return a value
    @wp.kernel(module="unique")
    def f1(x: float) -> float:
        return x

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(f1, dim=1, inputs=[3.0], device=device)

    # types that have no C-equivalent can't be returned from kernels either
    @wp.kernel(module="unique")
    def f2(x: float) -> wp.vec4f:
        return wp.vec4f(x)

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(f2, dim=1, inputs=[3.0], device=device)

    # also when the return type is not defined, no value can be returned
    @wp.kernel(module="unique")
    def f3(x: float):
        return x

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(f3, dim=1, inputs=[3.0], device=device)

    # specifying a non-None return annotation is invalid even with a bare return
    @wp.kernel(module="unique")
    def f4(x: float) -> float:
        return

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(f4, dim=1, inputs=[3.0], device=device)

    # kernel diagnostics should win over function-style return type mismatch diagnostics
    @wp.kernel(module="unique")
    def f5(x: float) -> int:
        return x

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(f5, dim=1, inputs=[3.0], device=device)

    # generic kernel argument inference should ignore invalid return annotations
    @wp.kernel(module="unique")
    def f6(x: Any) -> float:
        return

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(f6, dim=1, inputs=[3.0], device=device)


def test_error_kernel_return_alias_unique_module_reuse(test, device):
    """Verify aliased kernel return annotations prevent unique-module reuse."""

    Ret = None

    @wp.kernel(module="unique")
    def aliased_return_kernel(x: float) -> Ret:
        return

    wp.launch(aliased_return_kernel, dim=1, inputs=[3.0], device=device)

    Ret = float

    @wp.kernel(module="unique")
    def aliased_return_kernel(x: float) -> Ret:
        return

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(aliased_return_kernel, dim=1, inputs=[3.0], device=device)


def test_error_generic_kernel_return_alias_unique_module_reuse(test, device):
    """Verify generic aliased kernel returns prevent unique-module reuse."""

    Ret = None

    @wp.kernel(module="unique")
    def aliased_generic_return_kernel(x: Any) -> Ret:
        return

    wp.launch(aliased_generic_return_kernel, dim=1, inputs=[3.0], device=device)

    Ret = float

    @wp.kernel(module="unique")
    def aliased_generic_return_kernel(x: Any) -> Ret:
        return

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, _KERNEL_RETURN_ERROR_PATTERN):
        wp.launch(aliased_generic_return_kernel, dim=1, inputs=[3.0], device=device)


def test_error_mutating_constant_in_dynamic_loop(test, device):
    @wp.kernel(module="unique")
    def dynamic_loop_kernel(n: int, input: wp.array[float]):
        my_constant = 0.0
        for i in range(n):
            my_constant += input[i]

    inputs = wp.array([1.0, 2.0, 3.0], dtype=float, device=device)
    with test.assertRaisesRegex(
        wp.WarpCodegenError,
        r"Error mutating a constant my_constant inside a dynamic loop, use the following syntax\: pi = float\(3\.141\) to declare a dynamic variable",
    ):
        wp.launch(dynamic_loop_kernel, dim=1, inputs=[3, inputs], device=device)

    # the following nested loop must not raise an error
    const_a = 7
    const_b = 5

    @wp.kernel(module="unique")
    def mixed_dyn_static_loop_kernel(dyn_a: int, dyn_b: int, dyn_c: int, output: wp.array2d[float]):
        tid = wp.tid()
        for i in range(const_a + 1):
            for j in range(dyn_a + 1):
                for k in range(dyn_b + 1):
                    for l in range(const_b + 1):
                        for m in range(dyn_c + 1):
                            coeff = i + j + k + l + m
                            output[tid, coeff] = 1.0

    dyn_a, dyn_b, dyn_c = 3, 4, 5
    num_threads = 10
    output = wp.empty([num_threads, const_a + const_b + dyn_a + dyn_b + dyn_c + 1], dtype=float, device=device)
    wp.launch(
        mixed_dyn_static_loop_kernel,
        num_threads,
        inputs=[
            dyn_a,
            dyn_b,
            dyn_c,
        ],
        outputs=[output],
        device=device,
    )
    assert_np_equal(output.numpy(), np.ones([num_threads, const_a + const_b + dyn_a + dyn_b + dyn_c + 1]))

    @wp.kernel(module="unique")
    def static_then_dynamic_loop_kernel(mats: wp.array[wp.mat33d]):
        tid = wp.tid()
        mat = wp.mat33d()
        for i in range(3):
            for j in range(3):
                mat[i, j] = wp.float64(0.0)

        dim = 2
        for i in range(dim + 1):
            for j in range(dim + 1):
                mat[i, j] = wp.float64(1.0)

        mats[tid] = mat

    mats = wp.empty(1, dtype=wp.mat33d, device=device)
    wp.launch(static_then_dynamic_loop_kernel, dim=1, inputs=[mats], device=device)
    assert_np_equal(mats.numpy(), np.ones((1, 3, 3)))

    @wp.kernel(module="unique")
    def dynamic_then_static_loop_kernel(mats: wp.array[wp.mat33d]):
        tid = wp.tid()
        mat = wp.mat33d()

        dim = 2
        for i in range(dim + 1):
            for j in range(dim + 1):
                mat[i, j] = wp.float64(1.0)

        for i in range(3):
            for j in range(3):
                mat[i, j] = wp.float64(0.0)

        mats[tid] = mat

    mats = wp.empty(1, dtype=wp.mat33d, device=device)
    wp.launch(dynamic_then_static_loop_kernel, dim=1, inputs=[mats], device=device)
    assert_np_equal(mats.numpy(), np.zeros((1, 3, 3)))


def test_error_return_annotation_mismatch(test, device):
    @wp.func
    def foo_1(x: wp.int32) -> wp.int16:
        return wp.int8(x)

    def kernel_1_fn():
        x = foo_1(123)

    @wp.func
    def foo_2(x: int) -> int:
        return (x + x, x * x)

    def kernel_2_fn():
        x = foo_2(123)

    @wp.func
    def foo_3(x: int) -> tuple[int, int]:
        return (x, 1.23)

    def kernel_3_fn():
        _x, _y = foo_3(123)

    @wp.func
    def foo_4(x: int) -> tuple[int, int, int]:
        return (x + x, x * x)

    def kernel_4_fn():
        _x, _y, _z = foo_4(123)

    kernel = wp.Kernel(func=kernel_1_fn)
    with test.assertRaisesRegex(
        wp.WarpCodegenError,
        r"The function `foo_1` has its return type annotated as `int16` but the code returns a value of type `int8`.",
    ):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(
        wp.WarpCodegenError,
        r"The function `foo_2` has its return type annotated as `int` but the code returns 2 values.",
    ):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_3_fn)
    with test.assertRaisesRegex(
        wp.WarpCodegenError,
        r"The function `foo_3` has its return type annotated as `tuple\[int, int\]` but the code returns a tuple with types `\(int32, float32\)`.",
    ):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_4_fn)
    with test.assertRaisesRegex(
        wp.WarpCodegenError,
        r"The function `foo_4` has its return type annotated as a tuple of 3 elements but the code returns 2 values.",
    ):
        wp.launch(kernel, dim=1, device=device)


@wp.kernel
def test_call_syntax():
    expected_pow = 16.0
    wp.expect_eq(wp.pow(2.0, 4.0), expected_pow)
    wp.expect_eq(wp.pow(x=2.0, y=4.0), expected_pow)
    wp.expect_eq(wp.pow(2.0, y=4.0), expected_pow)
    wp.expect_eq(wp.pow(y=4.0, x=2.0), expected_pow)

    expected_matrix = wp.mat44(2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    pos = wp.vec3(1.0, 2.0, 3.0)
    rot = wp.quat(0.0, 0.0, 0.0, 1.0)
    scale = wp.vec3(2.0, 3.0, 4.0)
    wp.expect_eq(wp.transform_compose(pos, rot, scale), expected_matrix)
    wp.expect_eq(wp.transform_compose(position=pos, rotation=rot, scale=scale), expected_matrix)
    wp.expect_eq(wp.transform_compose(rotation=rot, position=pos, scale=scale), expected_matrix)


# test shadowing builtin functions
@wp.func
def sum(a: wp.vec3) -> float:
    return a[0] + a[1] + a[2]


@wp.kernel
def test_shadow_builtin():
    wp.expect_eq(sum(wp.vec3(1.0)), 3.0)


@wp.struct
class Iterator:
    valid: wp.bool


@wp.kernel(enable_backward=False)
def test_while_condition_eval():
    it = Iterator()
    it.valid = True
    while it.valid:
        it.valid = False


@wp.kernel
def conditional_return_or_sum(result: wp.array[wp.int32]):
    tid = wp.tid()

    if tid < 256:
        return

    wp.atomic_add(result, 0, 1)


def test_codegen_return_in_kernel(test, device):
    result = wp.zeros(1, dtype=wp.int32, device=device)

    grid_size = 1024

    # On CUDA devices, this becomes a grid-stride loop
    wp.launch(conditional_return_or_sum, dim=grid_size, inputs=[result], block_dim=256, max_blocks=1, device=device)

    test.assertEqual(result.numpy()[0], grid_size - 256)


@wp.kernel
def conditional_ifexp(x: float, result: wp.array[wp.int32]):
    wp.atomic_add(result, 0, 1) if x > 0.0 else wp.atomic_add(result, 1, 1)


def test_ifexp_only_executes_one_branch(test, device):
    result = wp.zeros(2, dtype=wp.int32, device=device)

    wp.launch(conditional_ifexp, dim=1, inputs=[1.0, result], device=device)

    values = result.numpy()
    # Only first branch is taken
    test.assertEqual(values[0], 1)
    test.assertEqual(values[1], 0)


@wp.kernel
def test_multiple_return_values_quat_to_axis_angle_kernel(
    q: wp.quath,
    expected_axis: wp.vec3h,
    expected_angle: wp.float16,
):
    axis, angle = wp.quat_to_axis_angle(q)

    wp.expect_near(axis[0], expected_axis[0], tolerance=wp.float16(1e-3))
    wp.expect_near(axis[1], expected_axis[1], tolerance=wp.float16(1e-3))
    wp.expect_near(axis[2], expected_axis[2], tolerance=wp.float16(1e-3))

    wp.expect_near(angle, expected_angle, tolerance=wp.float16(1e-3))


@wp.kernel
def test_multiple_return_values_svd3_kernel(
    A: wp.mat33f,
    expected_U: wp.mat33f,
    expected_sigma: wp.vec3f,
    expected_V: wp.mat33f,
):
    U, sigma, V = wp.svd3(A)

    wp.expect_near(U[0][0], expected_U[0][0], tolerance=1e-5)
    wp.expect_near(U[0][1], expected_U[0][1], tolerance=1e-5)
    wp.expect_near(U[0][2], expected_U[0][2], tolerance=1e-5)
    wp.expect_near(U[1][0], expected_U[1][0], tolerance=1e-5)
    wp.expect_near(U[1][1], expected_U[1][1], tolerance=1e-5)
    wp.expect_near(U[1][2], expected_U[1][2], tolerance=1e-5)
    wp.expect_near(U[2][0], expected_U[2][0], tolerance=1e-5)
    wp.expect_near(U[2][1], expected_U[2][1], tolerance=1e-5)
    wp.expect_near(U[2][2], expected_U[2][2], tolerance=1e-5)

    wp.expect_near(sigma[0], expected_sigma[0], tolerance=1e-5)
    wp.expect_near(sigma[1], expected_sigma[1], tolerance=1e-5)
    wp.expect_near(sigma[2], expected_sigma[2], tolerance=1e-5)

    wp.expect_near(V[0][0], expected_V[0][0], tolerance=1e-5)
    wp.expect_near(V[0][1], expected_V[0][1], tolerance=1e-5)
    wp.expect_near(V[0][2], expected_V[0][2], tolerance=1e-5)
    wp.expect_near(V[1][0], expected_V[1][0], tolerance=1e-5)
    wp.expect_near(V[1][1], expected_V[1][1], tolerance=1e-5)
    wp.expect_near(V[1][2], expected_V[1][2], tolerance=1e-5)
    wp.expect_near(V[2][0], expected_V[2][0], tolerance=1e-5)
    wp.expect_near(V[2][1], expected_V[2][1], tolerance=1e-5)
    wp.expect_near(V[2][2], expected_V[2][2], tolerance=1e-5)


def test_multiple_return_values(test, device):
    q = wp.quath(1.0, 2.0, 3.0, 4.0)
    expected_axis = wp.vec3h(0.26726124, 0.53452247, 0.80178368)
    expected_angle = 1.50408018

    axis, angle = wp.quat_to_axis_angle(q)

    test.assertAlmostEqual(float(axis[0]), float(expected_axis[0]), places=3)
    test.assertAlmostEqual(float(axis[1]), float(expected_axis[1]), places=3)
    test.assertAlmostEqual(float(axis[2]), float(expected_axis[2]), places=3)

    test.assertAlmostEqual(float(angle), expected_angle, places=3)

    wp.launch(
        test_multiple_return_values_quat_to_axis_angle_kernel,
        dim=1,
        inputs=(q, expected_axis, expected_angle),
        device=device,
    )

    # fmt: off
    A = wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    expected_U = wp.mat33(
        0.21483721, 0.88723058, -0.40824816,
        0.52058744, 0.24964368, 0.81649637,
        0.82633746, -0.38794267, -0.40824834,
    )
    expected_sigma = wp.vec3(16.84809875, 1.06836915, 0.00000019)
    expected_V = wp.mat33(
        0.47967088, -0.77669406, 0.40824246,
        0.57236743, -0.07568054, -0.81649727,
        0.66506463, 0.62531471, 0.40825251,
    )
    # fmt: on

    U, sigma, V = wp.svd3(A)

    test.assertAlmostEqual(U[0][0], expected_U[0][0], places=5)
    test.assertAlmostEqual(U[0][1], expected_U[0][1], places=5)
    test.assertAlmostEqual(U[0][2], expected_U[0][2], places=5)
    test.assertAlmostEqual(U[1][0], expected_U[1][0], places=5)
    test.assertAlmostEqual(U[1][1], expected_U[1][1], places=5)
    test.assertAlmostEqual(U[1][2], expected_U[1][2], places=5)
    test.assertAlmostEqual(U[2][0], expected_U[2][0], places=5)
    test.assertAlmostEqual(U[2][1], expected_U[2][1], places=5)
    test.assertAlmostEqual(U[2][2], expected_U[2][2], places=5)

    test.assertAlmostEqual(sigma[0], expected_sigma[0], places=5)
    test.assertAlmostEqual(sigma[1], expected_sigma[1], places=5)
    test.assertAlmostEqual(sigma[2], expected_sigma[2], places=5)

    test.assertAlmostEqual(V[0][0], expected_V[0][0], places=5)
    test.assertAlmostEqual(V[0][1], expected_V[0][1], places=5)
    test.assertAlmostEqual(V[0][2], expected_V[0][2], places=4)  # precision issue on ARM64 (GH-905)
    test.assertAlmostEqual(V[1][0], expected_V[1][0], places=5)
    test.assertAlmostEqual(V[1][1], expected_V[1][1], places=4)  # precision issue on ARM64 (GH-905)
    test.assertAlmostEqual(V[1][2], expected_V[1][2], places=5)
    test.assertAlmostEqual(V[2][0], expected_V[2][0], places=5)
    test.assertAlmostEqual(V[2][1], expected_V[2][1], places=5)
    test.assertAlmostEqual(V[2][2], expected_V[2][2], places=5)

    wp.launch(
        test_multiple_return_values_svd3_kernel,
        dim=1,
        inputs=(A, expected_U, expected_sigma, expected_V),
    )


@wp.struct
class Pun:
    f: wp.float16
    i: wp.int16


@wp.kernel
def test_cast():
    x = wp.int32(0x3FA00000)
    x_casted = wp.cast(x, wp.float32)
    wp.expect_eq(x_casted, 1.25)

    p = Pun()
    p.f = wp.float16(2.0)
    p.i = wp.int16(123)
    p_casted = wp.cast(p, wp.int32)
    wp.expect_eq(p_casted, 0x007B4000)


@wp.kernel
def test_reference_params_kernel(fs: wp.array[float], vs: wp.array[wp.vec3], qs: wp.array[wp.quat]):
    tid = wp.tid()

    v = wp.vec3(fs[tid], fs[tid], fs[tid])
    wp.expect_eq(v, wp.vec3(1.0, 1.0, 1.0))

    q = wp.quat(fs[tid], fs[tid], fs[tid], fs[tid])
    wp.expect_eq(q, wp.quat(1.0, 1.0, 1.0, 1.0))

    m1 = wp.mat22(fs[tid], fs[tid], fs[tid], fs[tid])
    wp.expect_eq(m1, wp.mat22(1.0, 1.0, 1.0, 1.0))

    m2 = wp.matrix_from_rows(vs[tid], vs[tid], vs[tid])
    wp.expect_eq(m2, wp.mat33(2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0))

    t = wp.transformation(p=vs[tid], q=qs[tid])
    wp.expect_eq(t, wp.transformation(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))

    sv = wp.spatial_vector(vs[tid], vs[tid])
    wp.expect_eq(sv, wp.spatial_vector(2.0, 3.0, 4.0, 2.0, 3.0, 4.0))


def test_reference_params(test, device):
    fs = wp.array((1.0,), dtype=float, device=device)
    vs = wp.array((wp.vec3(2.0, 3.0, 4.0),), dtype=wp.vec3, device=device)
    qs = wp.array((wp.quat(5.0, 6.0, 7.0, 8.0),), dtype=wp.quat, device=device)

    wp.launch(
        test_reference_params_kernel,
        dim=1,
        inputs=(fs, vs, qs),
        device=device,
    )
    wp.synchronize_device(device)


@wp.func
def side_effect_add(counter: wp.array[int], a: float, b: float) -> float:
    """Add two values and increment counter to track call count."""
    wp.atomic_add(counter, 0, 1)
    return a + b


@wp.kernel
def test_augassign_no_double_eval_kernel(
    counter: wp.array[int],
    result: wp.array[float],
):
    total = float(0.0)
    # The RHS should be evaluated exactly once per augmented assignment.
    # Before the fix, the codegen would evaluate the RHS twice for
    # augmented assignments on simple name targets (x += expr).
    total += side_effect_add(counter, 1.0, 2.0)
    total += side_effect_add(counter, 3.0, 4.0)
    total *= side_effect_add(counter, 1.0, 1.0)
    total -= side_effect_add(counter, 0.0, 0.0)
    result[0] = total


def test_augassign_no_double_eval(test, device):
    counter = wp.zeros(1, dtype=int, device=device)
    result = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        test_augassign_no_double_eval_kernel,
        dim=1,
        inputs=[counter, result],
        device=device,
    )

    # 4 augmented assignments (+=, +=, *=, -=), each calling side_effect_add once = 4 calls
    test.assertEqual(counter.numpy()[0], 4, "RHS of augmented assignment was evaluated more than once")
    # (0 + 3) + 7 = 10, 10 * 2 = 20, 20 - 0 = 20
    test.assertAlmostEqual(result.numpy()[0], 20.0)


@wp.struct
class AugAssignTestStruct:
    value: float


@wp.func
def side_effect_inc_int16(counter: wp.array[int], val: wp.int16) -> wp.int16:
    wp.atomic_add(counter, 0, 1)
    return val


@wp.func
def side_effect_vec3s(counter: wp.array[int], val: wp.vec3s) -> wp.vec3s:
    wp.atomic_add(counter, 0, 1)
    return val


@wp.func
def side_effect_return_float(counter: wp.array[int], val: float) -> float:
    wp.atomic_add(counter, 0, 1)
    return val


@wp.func
def side_effect_index(counter: wp.array[int], idx: int) -> int:
    wp.atomic_add(counter, 0, 1)
    return idx


# Attribute target (s.value += expr)
@wp.kernel
def test_augassign_no_double_eval_attribute_kernel(
    counter: wp.array[int],
    result: wp.array[float],
):
    s = AugAssignTestStruct()
    s.value = 1.0
    s.value += side_effect_add(counter, 2.0, 3.0)
    result[0] = s.value


def test_augassign_no_double_eval_attribute(test, device):
    counter = wp.zeros(1, dtype=int, device=device)
    result = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        test_augassign_no_double_eval_attribute_kernel,
        dim=1,
        inputs=[counter, result],
        device=device,
    )

    test.assertEqual(counter.numpy()[0], 1, "RHS of augmented assignment on attribute was evaluated more than once")
    test.assertAlmostEqual(result.numpy()[0], 6.0)


# Non-atomic subscript (arr_int16[i] += expr)
@wp.kernel
def test_augassign_no_double_eval_nonatomic_subscript_kernel(
    counter: wp.array[int],
    data: wp.array[wp.int16],
):
    i = wp.tid()
    data[i] += side_effect_inc_int16(counter, wp.int16(10))


def test_augassign_no_double_eval_nonatomic_subscript(test, device):
    counter = wp.zeros(1, dtype=int, device=device)
    data = wp.array([wp.int16(5)], dtype=wp.int16, device=device)

    wp.launch(
        test_augassign_no_double_eval_nonatomic_subscript_kernel,
        dim=1,
        inputs=[counter, data],
        device=device,
    )

    test.assertEqual(
        counter.numpy()[0], 1, "RHS of augmented assignment on non-atomic subscript was evaluated more than once"
    )
    test.assertEqual(data.numpy()[0], 15)


# Composite non-atomic subscript (arr_vec3s[i] += expr)
@wp.kernel
def test_augassign_no_double_eval_composite_nonatomic_kernel(
    counter: wp.array[int],
    data: wp.array[wp.vec3s],
):
    i = wp.tid()
    data[i] += side_effect_vec3s(counter, wp.vec3s(wp.int16(1), wp.int16(2), wp.int16(3)))


def test_augassign_no_double_eval_composite_nonatomic(test, device):
    counter = wp.zeros(1, dtype=int, device=device)
    data = wp.array([wp.vec3s(wp.int16(10), wp.int16(20), wp.int16(30))], dtype=wp.vec3s, device=device)

    wp.launch(
        test_augassign_no_double_eval_composite_nonatomic_kernel,
        dim=1,
        inputs=[counter, data],
        device=device,
    )

    test.assertEqual(
        counter.numpy()[0],
        1,
        "RHS of augmented assignment on composite non-atomic subscript was evaluated more than once",
    )
    result = data.numpy()[0]
    test.assertEqual(result[0], 11)
    test.assertEqual(result[1], 22)
    test.assertEqual(result[2], 33)


# Mul on array subscript (arr[i] *= expr)
@wp.kernel
def test_augassign_no_double_eval_mul_subscript_kernel(
    counter: wp.array[int],
    data: wp.array[float],
):
    i = wp.tid()
    data[i] *= side_effect_return_float(counter, 3.0)


def test_augassign_no_double_eval_mul_subscript(test, device):
    counter = wp.zeros(1, dtype=int, device=device)
    data = wp.array([5.0], dtype=float, device=device)

    wp.launch(
        test_augassign_no_double_eval_mul_subscript_kernel,
        dim=1,
        inputs=[counter, data],
        device=device,
    )

    test.assertEqual(
        counter.numpy()[0], 1, "RHS of augmented assignment with *= on subscript was evaluated more than once"
    )
    test.assertAlmostEqual(data.numpy()[0], 15.0)


# Pow on array subscript (arr[i] **= expr)
@wp.kernel
def test_augassign_no_double_eval_pow_subscript_kernel(
    counter: wp.array[int],
    data: wp.array[float],
):
    i = wp.tid()
    data[i] **= side_effect_return_float(counter, 2.0)


def test_augassign_no_double_eval_pow_subscript(test, device):
    counter = wp.zeros(1, dtype=int, device=device)
    data = wp.array([5.0], dtype=float, device=device)

    wp.launch(
        test_augassign_no_double_eval_pow_subscript_kernel,
        dim=1,
        inputs=[counter, data],
        device=device,
    )

    test.assertEqual(
        counter.numpy()[0], 1, "RHS of augmented assignment with **= on subscript was evaluated more than once"
    )
    test.assertAlmostEqual(data.numpy()[0], 25.0)


# Mul on vec component (v[0] *= expr)
@wp.kernel
def test_augassign_no_double_eval_mul_vec_component_kernel(
    counter: wp.array[int],
    data: wp.array[wp.vec3],
    result: wp.array[float],
):
    i = wp.tid()
    v = data[i]
    v[0] *= side_effect_return_float(counter, 2.0)
    data[i] = v
    result[0] = v[0]


def test_augassign_no_double_eval_mul_vec_component(test, device):
    counter = wp.zeros(1, dtype=int, device=device)
    data = wp.array([wp.vec3(3.0, 4.0, 5.0)], dtype=wp.vec3, device=device)
    result = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        test_augassign_no_double_eval_mul_vec_component_kernel,
        dim=1,
        inputs=[counter, data, result],
        device=device,
    )

    test.assertEqual(
        counter.numpy()[0], 1, "RHS of augmented assignment with *= on vec component was evaluated more than once"
    )
    test.assertAlmostEqual(result.numpy()[0], 6.0)


# Adjoint (wp.adjoint[x] += expr)
@wp.func
def augassign_custom_scale(counter: wp.array[int], x: float, s: float) -> float:
    return x * s


@wp.func_grad(augassign_custom_scale)
def adj_augassign_custom_scale(counter: wp.array[int], x: float, s: float, adj_ret: float):
    wp.adjoint[x] += side_effect_return_float(counter, adj_ret * s)
    wp.adjoint[s] += adj_ret * x


@wp.kernel
def test_augassign_no_double_eval_adjoint_kernel(
    counter: wp.array[int],
    input_val: wp.array[float],
    scale_val: wp.array[float],
    output_val: wp.array[float],
):
    tid = wp.tid()
    output_val[tid] = augassign_custom_scale(counter, input_val[tid], scale_val[tid])


def test_augassign_no_double_eval_adjoint(test, device):
    counter = wp.zeros(1, dtype=int, device=device)
    input_val = wp.array([3.0], dtype=float, device=device, requires_grad=True)
    scale_val = wp.array([2.0], dtype=float, device=device, requires_grad=True)
    output_val = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(
            test_augassign_no_double_eval_adjoint_kernel,
            dim=1,
            inputs=[counter, input_val, scale_val, output_val],
            device=device,
        )

    tape.backward(grads={output_val: wp.array([1.0], dtype=float, device=device)})

    # wp.adjoint[x] += side_effect_return_float(counter, adj_ret * s)
    # should call side_effect_return_float exactly once during backward
    test.assertEqual(counter.numpy()[0], 1, "RHS of wp.adjoint augmented assignment was evaluated more than once")
    # d(x*s)/dx = s = 2.0
    test.assertAlmostEqual(input_val.grad.numpy()[0], 2.0)


# Aggregate double-eval: data[side_effect_index(counter, i)].value += expr
@wp.kernel
def test_augassign_no_double_eval_subscript_attribute_kernel(
    counter: wp.array[int],
    data: wp.array[AugAssignTestStruct],
):
    i = wp.tid()
    data[side_effect_index(counter, i)].value += 5.0


def test_augassign_no_double_eval_subscript_attribute(test, device):
    counter = wp.zeros(1, dtype=int, device=device)
    data = wp.zeros(1, dtype=AugAssignTestStruct, device=device)
    # Initialize the struct's value field to 10.0
    data_np = data.numpy()
    data_np[0][0] = 10.0
    data = wp.array(data_np, dtype=AugAssignTestStruct, device=device)

    wp.launch(
        test_augassign_no_double_eval_subscript_attribute_kernel,
        dim=1,
        inputs=[counter, data],
        device=device,
    )

    test.assertEqual(
        counter.numpy()[0],
        1,
        "Aggregate index in augmented assignment on struct attribute was evaluated more than once",
    )
    test.assertAlmostEqual(data.numpy()[0][0], 15.0)


# Both aggregate and RHS double-eval:
# data[side_effect_index(idx_ctr, i)].value += side_effect_return_float(rhs_ctr, v)
@wp.kernel
def test_augassign_no_double_eval_both_kernel(
    idx_counter: wp.array[int],
    rhs_counter: wp.array[int],
    data: wp.array[AugAssignTestStruct],
):
    i = wp.tid()
    data[side_effect_index(idx_counter, i)].value += side_effect_return_float(rhs_counter, 5.0)


def test_augassign_no_double_eval_both(test, device):
    idx_counter = wp.zeros(1, dtype=int, device=device)
    rhs_counter = wp.zeros(1, dtype=int, device=device)
    data = wp.zeros(1, dtype=AugAssignTestStruct, device=device)
    data_np = data.numpy()
    data_np[0][0] = 10.0
    data = wp.array(data_np, dtype=AugAssignTestStruct, device=device)

    wp.launch(
        test_augassign_no_double_eval_both_kernel,
        dim=1,
        inputs=[idx_counter, rhs_counter, data],
        device=device,
    )

    test.assertEqual(idx_counter.numpy()[0], 1, "Aggregate index was evaluated more than once")
    test.assertEqual(rhs_counter.numpy()[0], 1, "RHS was evaluated more than once")
    test.assertAlmostEqual(data.numpy()[0][0], 15.0)


@wp.func
def func_to_local_double(a: float):
    return a * 2.0


@wp.func
def func_to_local_square(a: float):
    return a * a


@wp.func
def func_to_local_halve(a: float):
    return a * 0.5


func_to_local_handlers = {"double": func_to_local_double, "square": func_to_local_square}


# Binding a @wp.func to a local and calling through it should behave like calling it directly.
@wp.kernel
def assign_function_to_local_kernel(out: wp.array[float]):
    f = func_to_local_double
    out[0] = f(3.0)


# A function reached through wp.static(...) can be bound to a local and called.
@wp.kernel
def assign_static_function_to_local_kernel(out: wp.array[float]):
    add = wp.static(func_to_local_handlers["double"])
    sq = wp.static(func_to_local_handlers["square"])
    out[0] = add(3.0)
    out[1] = sq(3.0)


# wp.grad() of a function that was first bound to a local should match wp.grad() of the function.
@wp.kernel(enable_backward=False)
def grad_of_function_bound_to_local_kernel(out: wp.array[float]):
    f = func_to_local_square
    g = wp.grad(f)
    out[0] = g(3.0)  # d/da a^2 = 2a = 6 at a = 3


def test_assign_function_to_local(test, device):
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(assign_function_to_local_kernel, dim=1, outputs=[out], device=device)
    test.assertEqual(out.numpy()[0], 6.0)


def test_assign_static_function_to_local(test, device):
    out = wp.zeros(2, dtype=float, device=device)
    wp.launch(assign_static_function_to_local_kernel, dim=1, outputs=[out], device=device)
    test.assertEqual(out.numpy()[0], 6.0)
    test.assertEqual(out.numpy()[1], 9.0)


def test_grad_of_function_bound_to_local(test, device):
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(grad_of_function_bound_to_local_kernel, dim=1, outputs=[out], device=device)
    test.assertEqual(out.numpy()[0], 6.0)


# Rebinding a function-valued local to a different function has no codegen-able meaning
# (Warp has no function pointers) and should raise a clear error rather than miscompile.
def test_rebind_function_local_errors(test, device):
    @wp.kernel
    def rebind_function_local_kernel(out: wp.array[float]):
        f = func_to_local_double
        f = func_to_local_halve
        out[0] = f(3.0)

    with test.assertRaisesRegex(wp.WarpCodegenError, "rebinding function-valued local"):
        wp.launch(rebind_function_local_kernel, dim=1, outputs=[wp.zeros(1, dtype=float, device=device)], device=device)


# Rebinding a function-valued local to a non-function value (e.g. `f = some_func; f = 1.0`) has no
# codegen-able meaning either and should raise a clear error rather than fail on the generic type check.
def test_rebind_function_local_to_value_errors(test, device):
    @wp.kernel
    def rebind_function_local_to_value_kernel(out: wp.array[float]):
        f = func_to_local_double
        f = 1.0
        out[0] = f

    with test.assertRaisesRegex(wp.WarpCodegenError, "rebinding function-valued local.*non-function"):
        wp.launch(
            rebind_function_local_to_value_kernel,
            dim=1,
            outputs=[wp.zeros(1, dtype=float, device=device)],
            device=device,
        )


@wp.func
def gradwrapper_rebind_square(x: float):
    return x * x


def test_rebind_gradwrapper_local_to_value_errors(test, device):
    @wp.kernel(enable_backward=False, module="unique")
    def rebind_gradwrapper_local_to_value_kernel(out: wp.array[float]):
        g = wp.grad(gradwrapper_rebind_square)
        g = 1.0
        out[0] = g

    with test.assertRaisesRegex(wp.WarpCodegenError, "Cannot reassign local 'g'.*wp.grad"):
        wp.launch(
            rebind_gradwrapper_local_to_value_kernel,
            dim=1,
            outputs=[wp.zeros(1, dtype=float, device=device)],
            device=device,
        )


def test_rebind_gradwrapper_local_through_tuple_errors(test, device):
    @wp.kernel(enable_backward=False, module="unique")
    def rebind_gradwrapper_local_through_tuple_kernel(out: wp.array[float]):
        g = wp.grad(gradwrapper_rebind_square)
        g, x = (1.0, 2.0)
        out[0] = x

    with test.assertRaisesRegex(wp.WarpCodegenError, "Cannot reassign local 'g'.*wp.grad"):
        wp.launch(
            rebind_gradwrapper_local_through_tuple_kernel,
            dim=1,
            outputs=[wp.zeros(1, dtype=float, device=device)],
            device=device,
        )


def test_rebind_gradwrapper_local_through_augassign_errors(test, device):
    @wp.kernel(enable_backward=False, module="unique")
    def rebind_gradwrapper_local_through_augassign_kernel(out: wp.array[float]):
        g = wp.grad(gradwrapper_rebind_square)
        g += 1.0
        out[0] = 1.0

    with test.assertRaisesRegex(wp.WarpCodegenError, "Cannot reassign local 'g'.*wp.grad"):
        wp.launch(
            rebind_gradwrapper_local_through_augassign_kernel,
            dim=1,
            outputs=[wp.zeros(1, dtype=float, device=device)],
            device=device,
        )


F64_CONST = wp.constant(wp.float64(2.0 * math.pi))
I64_CONST = wp.constant(wp.int64(7))


@wp.kernel(module="test_unary_minus_on_64bit_constant_f64")
def _unary_minus_f64_kernel(values: wp.array[wp.float64], out: wp.array[wp.float64]):
    b = values[0]
    c = values[1]
    out[0] = -F64_CONST * b / (c * c)


@wp.kernel(module="test_unary_minus_on_64bit_constant_i64")
def _unary_minus_i64_kernel(values: wp.array[wp.int64], out: wp.array[wp.int64]):
    out[0] = -I64_CONST * values[0]


def test_unary_minus_on_64bit_constant(test, device):
    b, c = 3.0, 5.0
    expected_f64 = -(2.0 * math.pi) * b / (c * c)
    values_f64 = wp.array([b, c], dtype=wp.float64, device=device)
    out_f64 = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(_unary_minus_f64_kernel, dim=1, inputs=[values_f64, out_f64], device=device)

    values_i64 = wp.array([2], dtype=wp.int64, device=device)
    out_i64 = wp.zeros(1, dtype=wp.int64, device=device)
    wp.launch(_unary_minus_i64_kernel, dim=1, inputs=[values_i64, out_i64], device=device)

    np.testing.assert_allclose(out_f64.numpy(), [expected_f64], rtol=1e-15)
    np.testing.assert_array_equal(out_i64.numpy(), [-7 * 2])


class TestCodeGen(unittest.TestCase):
    def test_grid_stride_precedence(self):
        from warp._src.codegen import resolve_grid_stride  # noqa: PLC0415

        # resolve_grid_stride is the single build-time resolver: an explicit per-kernel choice wins,
        # otherwise the kernel inherits the resolved module/global default.
        self.assertFalse(resolve_grid_stride({}, False))
        self.assertTrue(resolve_grid_stride({}, True))
        self.assertTrue(resolve_grid_stride({"grid_stride": True}, False))
        self.assertFalse(resolve_grid_stride({"grid_stride": False}, True))

        # The effective value is resolved once at build and frozen on Kernel.grid_stride; launches read
        # it directly and never re-resolve from mutable config.
        saved_config = wp.config.default_grid_stride
        try:
            wp.config.default_grid_stride = True

            @wp.kernel(module="unique")
            def gs_unset(a: wp.array[float]):
                a[wp.tid()] = 1.0

            @wp.kernel(grid_stride=True, module="unique")
            def gs_true(a: wp.array[float]):
                a[wp.tid()] = 1.0

            @wp.kernel(grid_stride=False, module="unique")
            def gs_false(a: wp.array[float]):
                a[wp.tid()] = 1.0

            for k in (gs_unset, gs_true, gs_false):
                k.module.get_module_hash()  # build-time resolution records Kernel.grid_stride

            self.assertTrue(gs_unset.grid_stride)  # inherited the default (True) at build
            self.assertTrue(gs_true.grid_stride)  # explicit True overrides
            self.assertFalse(gs_false.grid_stride)  # explicit False overrides

            # Flipping the global default after the build does not mutate the frozen value.
            wp.config.default_grid_stride = False
            self.assertTrue(gs_unset.grid_stride)
        finally:
            wp.config.default_grid_stride = saved_config

    def _make_adjoint_with_filename(self, filename):
        source = "def kernel_fn(x: int):\n    y = x + 1\n    return y\n"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        namespace = {}
        try:
            exec(compile(source, filename, "exec"), namespace)
            adj = wp._src.codegen.Adjoint(namespace["kernel_fn"])
            adj.builder_options = {"lineinfo": True, "line_directives": True, "mode": "release"}
            return adj
        finally:
            linecache.cache.pop(filename, None)

    def test_get_arg_type_preserves_any(self):
        """Verify ``get_arg_type`` returns ``Any`` for generic parameters.

        ``Any`` marks an unspecialized generic parameter and must be returned
        as-is, both when passed directly and when carried on a ``Var``. Before
        Python 3.11 ``Any`` is a ``typing._SpecialForm`` instance rather than a
        ``type``, so a regressed implementation falls through to ``type(arg)``
        and yields ``typing._SpecialForm``, which has no type code.
        """
        self.assertIs(codegen.get_arg_type(Any), Any)
        self.assertIs(codegen.get_arg_type(codegen.Var("coords", Any)), Any)

    def test_line_directive_escapes_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'warp_poc"\n\r\t\x00\x1f\x7fint injected_from_filename;\n//.py')
            adj = self._make_adjoint_with_filename(os.path.join(tmpdir, "warp_poc.py"))
            adj.filename = filename

            directive = adj.get_line_directive("int x;", 0)

            normalized_dir = tmpdir.replace("\\", "/")
            self.assertEqual(
                directive,
                f'#line 1 "{normalized_dir}/warp_poc\\"\\n\\r\\t\\000\\037\\177int injected_from_filename;\\n//.py"',
            )
            self.assertEqual(directive.count("\n"), 0)

    def test_line_directive_preserves_normalized_path(self):
        filename = "C:\\warp\\kernels\\example.py"
        adj = self._make_adjoint_with_filename(filename)

        directive = adj.get_line_directive("int x;", 0)

        self.assertEqual(directive, '#line 1 "C:/warp/kernels/example.py"')

    def test_extract_function_source_slow_path_when_fast_returns_none(self):
        """When ``_try_extract_function_source`` returns ``None`` (e.g. for an
        ``exec``-defined function with no linecache entry), ``extract_function_source``
        falls through to ``inspect.getsourcelines`` exactly once and parses its result.
        """
        slow_source = "def generated():\n    return 42\n"

        with mock.patch.object(codegen.Adjoint, "_try_extract_function_source", return_value=None):
            with mock.patch.object(codegen.inspect, "getsourcelines", return_value=([slow_source], 123)) as get_lines:
                source, lineno, tree = codegen.Adjoint.extract_function_source(lambda: None)

        self.assertEqual(source, slow_source)  # already at column 0; dedent is a no-op
        self.assertEqual(lineno, 123)
        self.assertEqual(tree.body[0].name, "generated")
        get_lines.assert_called_once()

    def test_extract_function_source_fast_path_patterns(self):
        """Every fixture in ``aux_test_extract_source_patterns`` is served by the
        fast path. We force a hard failure if ``inspect.getsourcelines`` is ever
        called, so each ``subTest`` proves the corresponding branch of the forward
        walk produced a parseable slice on its own.
        """
        fixtures = [
            patterns.plain,
            patterns.multiline_paren_return,
            patterns.with_multiline_docstring,
            patterns.with_indented_body_multiline_string,
            patterns.with_nested_def,
            patterns.trailing_body_comment,
            patterns.async_function,
            patterns.the_function_that_follows_the_comment,
            patterns.plain_with_trailing_blanks,
        ]

        for fn in fixtures:
            with self.subTest(fixture=fn.__name__):
                # Sanity: the fast extractor produces *some* slice for the fixture.
                fast = codegen.Adjoint._try_extract_function_source(fn.__code__)
                self.assertIsNotNone(fast, f"fast extractor returned None for {fn.__name__}")

                # Now run the full extraction with inspect.getsourcelines mocked to
                # raise — if it gets called, the fast path failed for this pattern.
                with mock.patch.object(
                    codegen.inspect,
                    "getsourcelines",
                    side_effect=AssertionError(f"inspect.getsourcelines must not run for {fn.__name__}"),
                ):
                    _source, lineno, tree = codegen.Adjoint.extract_function_source(fn)

                self.assertEqual(lineno, fn.__code__.co_firstlineno)
                self.assertGreaterEqual(len(tree.body), 1)
                self.assertIn(type(tree.body[0]), (ast.FunctionDef, ast.AsyncFunctionDef))
                # body[0].name == co_name follows from the proof in extract_function_source's
                # docstring; assert it as a regression check.
                self.assertEqual(tree.body[0].name, fn.__code__.co_name)

    def test_extract_function_source_unwraps_like_inspect(self):
        """``extract_function_source`` follows ``__wrapped__`` so the fast path is
        a true substitute for ``inspect.getsourcelines`` on ``functools.wraps``-style
        decorators.
        """

        def real_kernel():
            return 42

        @functools.wraps(real_kernel)
        def wrapper():
            return real_kernel()

        # Confirm the fixture actually exercises the wrap: __code__ differs but
        # __wrapped__ points back at real_kernel.
        self.assertIs(wrapper.__wrapped__, real_kernel)
        self.assertIsNot(wrapper.__code__, real_kernel.__code__)

        reference_lines, _ = inspect.getsourcelines(wrapper)
        reference_dedented = textwrap.dedent("".join(reference_lines))

        with mock.patch.object(codegen.inspect, "getsourcelines", side_effect=AssertionError("fast path should run")):
            source, lineno, tree = codegen.Adjoint.extract_function_source(wrapper)

        self.assertEqual(source, reference_dedented)
        self.assertEqual(lineno, real_kernel.__code__.co_firstlineno)
        self.assertEqual(tree.body[0].name, "real_kernel")

    def test_adjoint_recovers_from_truncated_fast_extract(self):
        """When the fast extractor truncates a multi-line string, the parse-time
        fallback inside :meth:`extract_function_source` recovers via
        ``inspect.getsourcelines``.
        """
        # Sanity: the fast path really does produce a truncated, unparsable slice
        # for this fixture (otherwise the test would silently pass without exercising
        # the fallback).
        fast = codegen.Adjoint._try_extract_function_source(contains_truncating_string.__code__)
        self.assertIsNotNone(fast)
        with self.assertRaises(SyntaxError):
            ast.parse(fast[0])

        # The full extract must transparently fall back; tree, source, and the
        # constructed Adjoint must all reflect the inspect-recovered source.
        source, _lineno, tree = codegen.Adjoint.extract_function_source(contains_truncating_string)
        self.assertEqual(tree.body[0].name, "contains_truncating_string")

        self.assertEqual(source, textwrap.dedent(inspect.getsource(contains_truncating_string)))

        adj = codegen.Adjoint(contains_truncating_string)
        self.assertEqual(adj.tree.body[0].name, "contains_truncating_string")
        self.assertEqual(adj.source, textwrap.dedent(inspect.getsource(contains_truncating_string)))

    def test_extract_function_source_refreshes_stale_linecache(self):
        """The fast path must not accept stale ``linecache`` content for a file that
        was rewritten and recompiled in the same process.
        """

        def load_function(path, source):
            with open(path, "w", encoding="utf-8") as f:
                f.write(source)
            namespace = {"__file__": path, "__name__": "stale_linecache_fixture"}
            exec(compile(source, path, "exec"), namespace)
            return namespace["stale_func"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stale_linecache_fixture.py")
            source_a = "def stale_func():\n    return 1\n"
            source_b = "def stale_func():\n    return 22\n"

            try:
                func_a = load_function(path, source_a)
                first_source, _, _ = codegen.Adjoint.extract_function_source(func_a)
                self.assertEqual(first_source, source_a)

                func_b = load_function(path, source_b)

                source, _lineno, tree = codegen.Adjoint.extract_function_source(func_b)
                self.assertEqual(source, source_b)
                self.assertEqual(tree.body[0].name, "stale_func")
            finally:
                linecache.cache.pop(path, None)

    def test_extract_function_source_rejects_non_function_fast_slice(self):
        """If malformed line metadata makes the fast slice start inside a function,
        the parse may still succeed. That slice must be rejected and recovered via
        ``inspect.getsourcelines``.
        """
        source = "def line_shifted():\n    x = 1\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "line_metadata_fixture.py")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(source)
                namespace = {"__file__": path, "__name__": "line_metadata_fixture"}
                exec(compile(source, path, "exec"), namespace)

                func = namespace["line_shifted"]
                shifted_code = func.__code__.replace(co_firstlineno=func.__code__.co_firstlineno + 1)
                shifted_func = types.FunctionType(shifted_code, func.__globals__, func.__name__)

                with mock.patch.object(
                    codegen.Adjoint,
                    "_inspect_extract_function_source",
                    return_value=(source, func.__code__.co_firstlineno),
                ) as slow_extract:
                    extracted_source, lineno, tree = codegen.Adjoint.extract_function_source(shifted_func)

                self.assertEqual(extracted_source, source)
                self.assertEqual(lineno, func.__code__.co_firstlineno)
                self.assertIn(type(tree.body[0]), (ast.FunctionDef, ast.AsyncFunctionDef))
                self.assertEqual(tree.body[0].name, "line_shifted")
                slow_extract.assert_called_once_with(shifted_func)
            finally:
                linecache.cache.pop(path, None)

    def test_extract_lambda_source_parenthesized_multiline_body(self):
        body = codegen.Adjoint.extract_lambda_source(parenthesized_multiline_lambda(), only_body=True)

        self.assertIsNotNone(body)
        self.assertIn("\n", body)
        self.assertTrue(body.strip().startswith("("))
        self.assertTrue(body.strip().endswith(")"))
        self.assertIn("q[0] == 0.0", body)
        ast.parse(f"def generated(q, qd):\n    return {body}\n")

    def test_replace_static_expressions_replaces_call_in_ast(self):
        """The walker actually mutates ``adj.tree``: every resolvable ``wp.static``
        Call gets replaced with an ``ast.Constant`` (or ``ast.Name`` for a
        Function result). This pins the deferred-replacement application step,
        which is the only behavioural difference vs upstream's in-flight
        replacement.
        """
        _value_a = 7
        _value_b = 13

        def _kernel_with_two_statics(out: wp.array[int]):
            i = wp.tid()
            out[i] = wp.static(_value_a)
            out[i] += wp.static(_value_b)

        adj = codegen.Adjoint(_kernel_with_two_statics)

        # No wp.static Calls should remain in the tree after replacement.
        remaining_static_calls = [
            node
            for node in ast.walk(adj.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "static"
        ]
        self.assertEqual(remaining_static_calls, [])

        # Both constants should appear as ast.Constant nodes in the tree.
        constants = {node.value for node in ast.walk(adj.tree) if isinstance(node, ast.Constant)}
        self.assertIn(_value_a, constants)
        self.assertIn(_value_b, constants)

    def test_replace_static_expressions_defers_loop_var_reference(self):
        """A ``wp.static`` call inside a ``for`` body that references the loop
        variable must be deferred — ``has_unresolved_static_expressions`` set,
        Call left in the AST for codegen-time resolution. This pins the
        loop-variable tracking in ``visit_For`` / ``visit_Call``.
        """

        def _kernel_with_loop_var_static(out: wp.array[int]):
            for i in range(10):
                out[i] = wp.static(i + 1)

        adj = codegen.Adjoint(_kernel_with_loop_var_static)

        self.assertTrue(adj.has_unresolved_static_expressions)
        # wp.static(i + 1) should still be a Call in the AST (not eagerly replaced).
        remaining_static_calls = [
            node
            for node in ast.walk(adj.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "static"
        ]
        self.assertEqual(len(remaining_static_calls), 1)


devices = get_test_devices()

add_kernel_test(TestCodeGen, name="test_expect", kernel=test_expect, dim=1, devices=devices)
add_kernel_test(TestCodeGen, name="test_inplace", kernel=test_inplace, dim=1, devices=devices)
add_kernel_test(TestCodeGen, name="test_rename", kernel=test_rename, dim=1, devices=devices)
add_kernel_test(TestCodeGen, name="test_constant", kernel=test_constant, inputs=[1.0], dim=1, devices=devices)
add_kernel_test(
    TestCodeGen, name="test_dynamic_for_rename", kernel=test_dynamic_for_rename, inputs=[10], dim=1, devices=devices
)
add_kernel_test(
    TestCodeGen, name="test_dynamic_for_inplace", kernel=test_dynamic_for_inplace, inputs=[10], dim=1, devices=devices
)
add_kernel_test(TestCodeGen, name="test_reassign", kernel=test_reassign, dim=1, devices=devices)
add_kernel_test(
    TestCodeGen, name="test_dynamic_reassign", kernel=test_dynamic_reassign, inputs=[2], dim=1, devices=devices
)

add_kernel_test(
    TestCodeGen,
    name="test_range_dynamic_forward",
    kernel=test_range_dynamic,
    dim=1,
    inputs=[0, 4, 1],
    expect=[0, 1, 2, 3],
    devices=devices,
)
add_kernel_test(
    TestCodeGen,
    name="test_range_dynamic_reverse",
    kernel=test_range_dynamic,
    dim=1,
    inputs=[4, 0, -1],
    expect=[4, 3, 2, 1],
    devices=devices,
)
add_kernel_test(
    TestCodeGen,
    name="test_range_dynamic_forward_step",
    kernel=test_range_dynamic,
    dim=1,
    inputs=[0, 8, 2],
    expect=[0, 2, 4, 6],
    devices=devices,
)
add_kernel_test(
    TestCodeGen,
    name="test_range_dynamic_reverse_step",
    kernel=test_range_dynamic,
    dim=1,
    inputs=[8, 0, -2],
    expect=[8, 6, 4, 2],
    devices=devices,
)

add_kernel_test(
    TestCodeGen, name="test_range_static_sum", kernel=test_range_static_sum, dim=1, expect=[10, 10, 10], devices=devices
)
add_kernel_test(
    TestCodeGen,
    name="test_range_dynamic_sum",
    kernel=test_range_dynamic_sum,
    dim=1,
    inputs=[0, 10, 2],
    expect=[10, 10, 10, 10],
    devices=devices,
)
add_kernel_test(
    TestCodeGen,
    name="test_range_dynamic_sum_zero",
    kernel=test_range_dynamic_sum,
    dim=1,
    inputs=[0, 0, 1],
    expect=[0, 0, 0, 0],
    devices=devices,
)
add_kernel_test(TestCodeGen, name="test_range_constant", kernel=test_range_constant, dim=1, devices=devices)
add_kernel_test(
    TestCodeGen,
    name="test_range_constant_dynamic_nested",
    kernel=test_range_constant_dynamic_nested,
    dim=1,
    inputs=[10],
    devices=devices,
)
add_kernel_test(
    TestCodeGen, name="test_range_dynamic_nested", kernel=test_range_dynamic_nested, dim=1, inputs=[4], devices=devices
)
add_kernel_test(TestCodeGen, name="test_range_expression", kernel=test_range_expression, dim=1, devices=devices)

add_kernel_test(TestCodeGen, name="test_while_zero", kernel=test_while, dim=1, inputs=[0], devices=devices)
add_kernel_test(TestCodeGen, name="test_while_positive", kernel=test_while, dim=1, inputs=[16], devices=devices)
add_kernel_test(TestCodeGen, name="test_pass", kernel=test_pass, dim=1, inputs=[16], devices=devices)

add_kernel_test(TestCodeGen, name="test_break", kernel=test_break, dim=1, inputs=[10], devices=devices)
add_kernel_test(TestCodeGen, name="test_break_early", kernel=test_break_early, dim=1, inputs=[10], devices=devices)
add_kernel_test(TestCodeGen, name="test_break_unroll", kernel=test_break_unroll, dim=1, devices=devices)
add_kernel_test(TestCodeGen, name="test_break_while", kernel=test_break_while, dim=1, devices=devices)
add_kernel_test(TestCodeGen, name="test_break_while_true", kernel=test_break_while_true, dim=1, devices=devices)
add_kernel_test(
    TestCodeGen, name="test_break_while_true_nested", kernel=test_break_while_true_nested, dim=1, devices=devices
)
add_kernel_test(
    TestCodeGen, name="test_break_multiple", kernel=test_break_multiple, dim=1, inputs=[10], devices=devices
)
add_kernel_test(TestCodeGen, name="test_continue", kernel=test_continue, dim=1, inputs=[10], devices=devices)
add_kernel_test(TestCodeGen, name="test_continue_unroll", kernel=test_continue_unroll, dim=1, devices=devices)

add_function_test(TestCodeGen, func=test_unresolved_func, name="test_unresolved_func", devices=devices)
add_function_test(TestCodeGen, func=test_unresolved_symbol, name="test_unresolved_symbol", devices=devices)
add_function_test(TestCodeGen, func=test_invalid_namespace_path, name="test_invalid_namespace_path", devices=devices)
add_function_test(TestCodeGen, func=test_error_global_var, name="test_error_global_var", devices=devices)
add_function_test(
    TestCodeGen, func=test_error_collection_construct, name="test_error_collection_construct", devices=devices
)
add_function_test(
    TestCodeGen, func=test_error_unmatched_arguments, name="test_error_unmatched_arguments", devices=devices
)
add_function_test(
    TestCodeGen,
    func=test_error_kernel_return_value,
    name="test_error_kernel_return_value",
    devices=devices,
)
add_function_test(
    TestCodeGen,
    func=test_error_kernel_return_alias_unique_module_reuse,
    name="test_error_kernel_return_alias_unique_module_reuse",
    devices=devices,
)
add_function_test(
    TestCodeGen,
    func=test_error_generic_kernel_return_alias_unique_module_reuse,
    name="test_error_generic_kernel_return_alias_unique_module_reuse",
    devices=devices,
)
add_function_test(
    TestCodeGen,
    func=test_error_mutating_constant_in_dynamic_loop,
    name="test_error_mutating_constant_in_dynamic_loop",
    devices=devices,
)
add_function_test(
    TestCodeGen,
    func=test_error_return_annotation_mismatch,
    name="test_error_return_annotation_mismatch",
    devices=devices,
)
add_kernel_test(TestCodeGen, name="test_call_syntax", kernel=test_call_syntax, dim=1, devices=devices)
add_kernel_test(TestCodeGen, name="test_shadow_builtin", kernel=test_shadow_builtin, dim=1, devices=devices)
add_kernel_test(TestCodeGen, name="test_while_condition_eval", kernel=test_while_condition_eval, dim=1, devices=devices)
add_function_test(TestCodeGen, "test_codegen_return_in_kernel", test_codegen_return_in_kernel, devices=devices)
add_function_test(
    TestCodeGen, "test_ifexp_only_executes_one_branch", test_ifexp_only_executes_one_branch, devices=devices
)
add_function_test(
    TestCodeGen,
    func=test_multiple_return_values,
    name="test_multiple_return_values",
    devices=devices,
)
add_kernel_test(TestCodeGen, name="test_cast", kernel=test_cast, dim=1, devices=devices)
add_function_test(TestCodeGen, "test_reference_params", test_reference_params, devices=devices)
add_function_test(TestCodeGen, "test_augassign_no_double_eval", test_augassign_no_double_eval, devices=devices)
add_function_test(
    TestCodeGen, "test_augassign_no_double_eval_attribute", test_augassign_no_double_eval_attribute, devices=devices
)
add_function_test(
    TestCodeGen,
    "test_augassign_no_double_eval_nonatomic_subscript",
    test_augassign_no_double_eval_nonatomic_subscript,
    devices=devices,
)
add_function_test(
    TestCodeGen,
    "test_augassign_no_double_eval_composite_nonatomic",
    test_augassign_no_double_eval_composite_nonatomic,
    devices=devices,
)
add_function_test(
    TestCodeGen,
    "test_augassign_no_double_eval_mul_subscript",
    test_augassign_no_double_eval_mul_subscript,
    devices=devices,
)
add_function_test(
    TestCodeGen,
    "test_augassign_no_double_eval_pow_subscript",
    test_augassign_no_double_eval_pow_subscript,
    devices=devices,
)
add_function_test(
    TestCodeGen,
    "test_augassign_no_double_eval_mul_vec_component",
    test_augassign_no_double_eval_mul_vec_component,
    devices=devices,
)
add_function_test(
    TestCodeGen, "test_augassign_no_double_eval_adjoint", test_augassign_no_double_eval_adjoint, devices=devices
)
add_function_test(
    TestCodeGen,
    "test_augassign_no_double_eval_subscript_attribute",
    test_augassign_no_double_eval_subscript_attribute,
    devices=devices,
)
add_function_test(
    TestCodeGen,
    "test_augassign_no_double_eval_both",
    test_augassign_no_double_eval_both,
    devices=devices,
)
add_function_test(TestCodeGen, "test_assign_function_to_local", test_assign_function_to_local, devices=devices)
add_function_test(
    TestCodeGen, "test_assign_static_function_to_local", test_assign_static_function_to_local, devices=devices
)
add_function_test(
    TestCodeGen, "test_grad_of_function_bound_to_local", test_grad_of_function_bound_to_local, devices=devices
)
add_function_test(TestCodeGen, "test_rebind_function_local_errors", test_rebind_function_local_errors, devices=devices)
add_function_test(
    TestCodeGen,
    "test_rebind_function_local_to_value_errors",
    test_rebind_function_local_to_value_errors,
    devices=devices,
)
add_function_test(
    TestCodeGen,
    "test_rebind_gradwrapper_local_to_value_errors",
    test_rebind_gradwrapper_local_to_value_errors,
    devices=devices,
)
add_function_test(
    TestCodeGen,
    "test_rebind_gradwrapper_local_through_tuple_errors",
    test_rebind_gradwrapper_local_through_tuple_errors,
    devices=devices,
)
add_function_test(
    TestCodeGen,
    "test_rebind_gradwrapper_local_through_augassign_errors",
    test_rebind_gradwrapper_local_through_augassign_errors,
    devices=devices,
)
add_function_test(
    TestCodeGen,
    "test_unary_minus_on_64bit_constant",
    test_unary_minus_on_64bit_constant,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
