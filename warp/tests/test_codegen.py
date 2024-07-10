# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import unittest

import warp as wp
from warp.tests.unittest_utils import *


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
    arr = wp.array(
        (1.0, 2.0, 3.0),
        dtype=float,
        device=device,
    )

    def kernel_1_fn(
        out: wp.array(dtype=float),
    ):
        out[0] = arr[0]

    def kernel_2_fn(
        out: wp.array(dtype=float),
    ):
        out[0] = arr

    def kernel_3_fn(
        out: wp.array(dtype=float),
    ):
        out[0] = wp.lower_bound(arr, 2.0)

    out = wp.empty_like(arr)

    kernel = wp.Kernel(func=kernel_1_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Cannot reference a global variable from a kernel unless `wp.constant\(\)` is being used",
    ):
        wp.launch(kernel, dim=out.shape, inputs=(), outputs=(out,))

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Cannot reference a global variable from a kernel unless `wp.constant\(\)` is being used",
    ):
        wp.launch(kernel, dim=out.shape, inputs=(), outputs=(out,))

    kernel = wp.Kernel(func=kernel_3_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Cannot reference a global variable from a kernel unless `wp.constant\(\)` is being used",
    ):
        wp.launch(kernel, dim=out.shape, inputs=(), outputs=(out,))


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
        wp.launch(kernel, dim=1)

    kernel = wp.Kernel(func=kernel_2_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Tuple constructs are not supported in kernels. Use vectors like `wp.vec3\(\)` for small collections instead.",
    ):
        wp.launch(kernel, dim=1)

    kernel = wp.Kernel(func=kernel_3_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Construct `ast.Dict` not supported in kernels.",
    ):
        wp.launch(kernel, dim=1)

    kernel = wp.Kernel(func=kernel_4_fn)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Tuple constructs are not supported in kernels. Use vectors like `wp.vec3\(\)` instead.",
    ):
        wp.launch(kernel, dim=1)


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


class TestCodeGen(unittest.TestCase):
    pass


devices = get_test_devices()

add_kernel_test(TestCodeGen, name="test_inplace", kernel=test_inplace, dim=1, devices=devices)
add_kernel_test(TestCodeGen, name="test_rename", kernel=test_rename, dim=1, devices=devices)
add_kernel_test(TestCodeGen, name="test_constant", kernel=test_constant, inputs=[1.0], dim=1, devices=devices)
add_kernel_test(
    TestCodeGen, name="test_dynamic_for_rename", kernel=test_dynamic_for_rename, inputs=[10], dim=1, devices=devices
)
add_kernel_test(
    TestCodeGen,
    name="test_dynamic_for_inplace",
    kernel=test_dynamic_for_inplace,
    inputs=[10],
    dim=1,
    devices=devices,
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
    TestCodeGen,
    name="test_range_static_sum",
    kernel=test_range_static_sum,
    dim=1,
    expect=[10, 10, 10],
    devices=devices,
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
    TestCodeGen,
    name="test_range_dynamic_nested",
    kernel=test_range_dynamic_nested,
    dim=1,
    inputs=[4],
    devices=devices,
)
add_kernel_test(
    TestCodeGen,
    name="test_range_expression",
    kernel=test_range_expression,
    dim=1,
    devices=devices,
)

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

add_kernel_test(TestCodeGen, name="test_call_syntax", kernel=test_call_syntax, dim=1, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
