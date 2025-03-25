# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest
from typing import Tuple

import warp as wp
from warp.tests.unittest_utils import *


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
def test_range_static_sum(result: wp.array(dtype=int)):
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
def test_range_dynamic_sum(start: int, end: int, step: int, result: wp.array(dtype=int)):
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
def test_range_dynamic(start: int, end: int, step: int, result: wp.array(dtype=int)):
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
    from warp.tests.aux_test_unresolved_func import unresolved_func_kernel

    # ensure that an appropriate exception is raised when the bad module gets loaded
    with test.assertRaisesRegex(RuntimeError, "Could not find function wp.missing_func"):
        wp.launch(unresolved_func_kernel, dim=1, inputs=[], device=device)

    # remove all references to the bad module so that subsequent calls to wp.force_load()
    # won't try to load it unless we explicitly re-import it again
    del wp.context.user_modules["warp.tests.aux_test_unresolved_func"]
    del sys.modules["warp.tests.aux_test_unresolved_func"]


def test_unresolved_symbol(test, device):
    # kernel with unresolved symbol must be in a separate module, otherwise the current module would fail to load
    from warp.tests.aux_test_unresolved_symbol import unresolved_symbol_kernel

    # ensure that an appropriate exception is raised when the bad module gets loaded
    with test.assertRaisesRegex(KeyError, "Referencing undefined symbol: missing_symbol"):
        wp.launch(unresolved_symbol_kernel, dim=1, inputs=[], device=device)

    # remove all references to the bad module so that subsequent calls to wp.force_load()
    # won't try to load it unless we explicitly re-import it again
    del wp.context.user_modules["warp.tests.aux_test_unresolved_symbol"]
    del sys.modules["warp.tests.aux_test_unresolved_symbol"]


def test_error_global_var(test, device):
    arr = wp.array((1.0, 2.0, 3.0), dtype=float, device=device)

    def kernel_1_fn(out: wp.array(dtype=float)):
        out[0] = arr[0]

    def kernel_2_fn(out: wp.array(dtype=float)):
        out[0] = arr

    def kernel_3_fn(out: wp.array(dtype=float)):
        out[0] = wp.lower_bound(arr, 2.0)

    out = wp.empty_like(arr)

    kernel = wp.Kernel(func=kernel_1_fn)
    with test.assertRaisesRegex(TypeError, r"Invalid external reference type: <class 'warp.types.array'>"):
        wp.launch(kernel, dim=out.shape, inputs=(), outputs=(out,), device=device)

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(TypeError, r"Invalid external reference type: <class 'warp.types.array'>"):
        wp.launch(kernel, dim=out.shape, inputs=(), outputs=(out,), device=device)

    kernel = wp.Kernel(func=kernel_3_fn)
    with test.assertRaisesRegex(TypeError, r"Invalid external reference type: <class 'warp.types.array'>"):
        wp.launch(kernel, dim=out.shape, inputs=(), outputs=(out,), device=device)


def test_error_collection_construct(test, device):
    def kernel_1_fn():
        x = [1.0, 2.0, 3.0]

    def kernel_2_fn():
        x = (1.0, 2.0, 3.0)

    def kernel_3_fn():
        x = {"a": 1.0, "b": 2.0, "c": 3.0}

    def kernel_4_fn():
        wp.length((1.0, 2.0, 3.0))

    kernel = wp.Kernel(func=kernel_1_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"List constructs are not supported in kernels. Use vectors like `wp.vec3\(\)` for small collections instead.",
    ):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Tuple constructs are not supported in kernels. Use vectors like `wp.vec3\(\)` for small collections instead.",
    ):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_3_fn)
    with test.assertRaisesRegex(RuntimeError, r"Construct `ast.Dict` not supported in kernels."):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_4_fn)
    with test.assertRaisesRegex(
        RuntimeError, r"Tuple constructs are not supported in kernels. Use vectors like `wp.vec3\(\)` instead."
    ):
        wp.launch(kernel, dim=1, device=device)


def test_error_unmatched_arguments(test, device):
    def kernel_1_fn():
        a = 1 * 1.0

    def kernel_2_fn():
        x = wp.dot(wp.vec2(1.0, 2.0), wp.vec2h(wp.float16(1.0), wp.float16(2.0)))

    kernel = wp.Kernel(func=kernel_1_fn)
    with test.assertRaisesRegex(RuntimeError, r"Input types must be the same, got \['int32', 'float32'\]"):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Input types must be exactly the same, got \[\"vector\(length=2, dtype=<class 'warp.types.float32'>\)\", \"vector\(length=2, dtype=<class 'warp.types.float16'>\)\"\]",
    ):
        wp.launch(kernel, dim=1, device=device)


def test_error_mutating_constant_in_dynamic_loop(test, device):
    @wp.kernel
    def dynamic_loop_kernel(n: int, input: wp.array(dtype=float)):
        my_constant = 0.0
        for i in range(n):
            my_constant += input[i]

    inputs = wp.array([1.0, 2.0, 3.0], dtype=float, device=device)
    with test.assertRaisesRegex(
        wp.codegen.WarpCodegenError,
        r"Error mutating a constant my_constant inside a dynamic loop, use the following syntax\: pi = float\(3\.141\) to declare a dynamic variable",
    ):
        wp.launch(dynamic_loop_kernel, dim=1, inputs=[3, inputs], device=device)

    # the following nested loop must not raise an error
    const_a = 7
    const_b = 5

    @wp.kernel
    def mixed_dyn_static_loop_kernel(dyn_a: int, dyn_b: int, dyn_c: int, output: wp.array(dtype=float, ndim=2)):
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

    @wp.kernel
    def static_then_dynamic_loop_kernel(mats: wp.array(dtype=wp.mat33d)):
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

    @wp.kernel
    def dynamic_then_static_loop_kernel(mats: wp.array(dtype=wp.mat33d)):
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
    def foo_3(x: int) -> Tuple[int, int]:
        return (x, 1.23)

    def kernel_3_fn():
        x, y = foo_3(123)

    @wp.func
    def foo_4(x: int) -> Tuple[int, int, int]:
        return (x + x, x * x)

    def kernel_4_fn():
        x, y, z = foo_4(123)

    kernel = wp.Kernel(func=kernel_1_fn)
    with test.assertRaisesRegex(
        wp.codegen.WarpCodegenError,
        r"The function `foo_1` has its return type annotated as `int16` but the code returns a value of type `int8`.",
    ):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(
        wp.codegen.WarpCodegenError,
        r"The function `foo_2` has its return type annotated as `int` but the code returns 2 values.",
    ):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_3_fn)
    with test.assertRaisesRegex(
        wp.codegen.WarpCodegenError,
        r"The function `foo_3` has its return type annotated as `Tuple\[int, int\]` but the code returns a tuple with types `\(int32, float32\)`.",
    ):
        wp.launch(kernel, dim=1, device=device)

    kernel = wp.Kernel(func=kernel_4_fn)
    with test.assertRaisesRegex(
        wp.codegen.WarpCodegenError,
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
    wp.expect_eq(wp.matrix(pos, rot, scale, wp.float32), expected_matrix)
    wp.expect_eq(wp.matrix(pos=pos, rot=rot, scale=scale, dtype=wp.float32), expected_matrix)
    wp.expect_eq(wp.matrix(pos, rot, scale, dtype=wp.float32), expected_matrix)
    wp.expect_eq(wp.matrix(rot=rot, pos=pos, dtype=wp.float32, scale=scale), expected_matrix)


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
def conditional_return_or_sum(result: wp.array(dtype=wp.int32)):
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


class TestCodeGen(unittest.TestCase):
    pass


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
add_kernel_test(
    TestCodeGen, name="test_break_multiple", kernel=test_break_multiple, dim=1, inputs=[10], devices=devices
)
add_kernel_test(TestCodeGen, name="test_continue", kernel=test_continue, dim=1, inputs=[10], devices=devices)
add_kernel_test(TestCodeGen, name="test_continue_unroll", kernel=test_continue_unroll, dim=1, devices=devices)

add_function_test(TestCodeGen, func=test_unresolved_func, name="test_unresolved_func", devices=devices)
add_function_test(TestCodeGen, func=test_unresolved_symbol, name="test_unresolved_symbol", devices=devices)
add_function_test(TestCodeGen, func=test_error_global_var, name="test_error_global_var", devices=devices)
add_function_test(
    TestCodeGen, func=test_error_collection_construct, name="test_error_collection_construct", devices=devices
)
add_function_test(
    TestCodeGen, func=test_error_unmatched_arguments, name="test_error_unmatched_arguments", devices=devices
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

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
