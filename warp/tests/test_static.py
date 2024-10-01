# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
from typing import Dict, List

import numpy as np

import warp
import warp as wp
from warp.tests.unittest_utils import *

global_variable = 3


@wp.func
def static_global_variable_func():
    static_var = warp.static(global_variable + 2)
    return static_var


@wp.kernel
def static_global_variable_kernel(results: wp.array(dtype=int)):
    # evaluate a constant expression at codegen time
    static_var = static_global_variable_func()
    const_var = 3
    # call a function at codegen time
    static_func_result = wp.static(static_global_variable_func() + const_var)
    results[0] = static_var
    results[1] = static_func_result


@wp.struct
class StaticallyConstructableStruct:
    mat: wp.mat33
    vec: wp.vec3
    i: int


@wp.struct
class StaticallyConstructableNestedStruct:
    s: StaticallyConstructableStruct
    tf: wp.transform
    quat: wp.quat


@wp.func
def construct_struct(mat: wp.mat33, vec: wp.vec3, i: int):
    s = StaticallyConstructableStruct()
    s.mat = mat
    s.vec = vec
    s.i = i
    return s


@wp.func
def construct_nested_struct(mat: wp.mat33, vec: wp.vec3, i: int, tf: wp.transform, quat: wp.quat):
    n = StaticallyConstructableNestedStruct()
    n.s = construct_struct(mat, vec, i)
    n.tf = tf
    n.quat = quat
    return n


@wp.kernel
def construct_static_struct_kernel(results: wp.array(dtype=StaticallyConstructableStruct)):
    static_struct = wp.static(
        construct_struct(
            wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
            wp.vec3(1.0, 2.0, 3.0),
            1,
        )
    )
    results[0] = static_struct


@wp.kernel
def construct_static_nested_struct_kernel(results: wp.array(dtype=StaticallyConstructableNestedStruct)):
    static_struct = wp.static(
        construct_nested_struct(
            wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
            wp.vec3(1.0, 2.0, 3.0),
            1,
            wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 2.0)),
            wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), wp.pi / 2.0),
        )
    )
    results[0] = static_struct


def test_static_global_variable(test, device):
    results = wp.zeros(2, dtype=int, device=device)
    wp.launch(static_global_variable_kernel, 1, [results], device=device)
    assert_np_equal(results.numpy(), np.array([5, 8], dtype=int))


def test_construct_static_struct(test, device):
    results = wp.zeros(1, dtype=StaticallyConstructableStruct, device=device)
    wp.launch(construct_static_struct_kernel, 1, [results], device=device)
    results = results.numpy()
    assert_np_equal(results[0][0], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    assert_np_equal(results[0][1], np.array([1.0, 2.0, 3.0]))
    assert_np_equal(results[0][2], 1)


def test_construct_static_nested_struct(test, device):
    results = wp.zeros(1, dtype=StaticallyConstructableNestedStruct, device=device)
    wp.launch(construct_static_nested_struct_kernel, 1, [results], device=device)
    results = results.numpy()

    tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 2.0))
    quat = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), wp.pi / 2.0)

    assert_np_equal(results[0][0][0], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    assert_np_equal(results[0][0][1], np.array([1.0, 2.0, 3.0]))
    assert_np_equal(results[0][0][2], 1)
    assert_np_equal(results[0][1], np.array(tf))
    assert_np_equal(results[0][2], np.array(quat))


def test_invalid_static_expression(test, device):
    @wp.kernel
    def invalid_kernel():
        wp.static(1.0 / 0.0)

    with test.assertRaisesRegex(
        warp.codegen.WarpCodegenError, r"Error evaluating static expression\: float division by zero"
    ):
        wp.launch(invalid_kernel, 1, device=device)

    @wp.kernel
    def invalid_kernel(i: int):
        wp.static(i * 2)

    with test.assertRaisesRegex(
        wp.codegen.WarpCodegenError,
        r"Error evaluating static expression\: name 'i' is not defined\. Make sure all variables used in the static expression are constant\.",
    ):
        wp.launch(invalid_kernel, 1, device=device, inputs=[3])


def test_static_expression_return_types(test, device):
    @wp.kernel
    def invalid_kernel():
        wp.static(wp.zeros(3, device=device))

    with test.assertRaisesRegex(
        warp.codegen.WarpCodegenError,
        r"Static expression returns an unsupported value\: a Warp array cannot be created inside Warp kernels",
    ):
        wp.launch(invalid_kernel, 1, device=device)

    @wp.struct
    class Baz:
        data: wp.array(dtype=int)
        z: wp.vec3

    @wp.struct
    class Bar:
        baz: Baz
        y: float

    @wp.struct
    class Foo:
        bar: Bar
        x: int

    def create_struct():
        foo = Foo()
        foo.bar = Bar()
        foo.bar.baz = Baz()
        foo.bar.baz.data = wp.zeros(3, dtype=int, device=device)
        foo.bar.baz.z = wp.vec3(1, 2, 3)
        foo.bar.y = 1.23
        foo.x = 123
        return foo

    @wp.kernel
    def invalid_kernel():
        wp.static(create_struct())

    with test.assertRaisesRegex(
        warp.codegen.WarpCodegenError,
        r"Static expression returns an unsupported value: the returned Warp struct contains a data type that cannot be constructed inside Warp kernels\: a Warp array cannot be created inside Warp kernels at .*?Foo\.bar\.baz",
    ):
        wp.launch(invalid_kernel, 1, device=device)

    def function_with_no_return_value():
        pass

    @wp.kernel
    def invalid_kernel():
        wp.static(function_with_no_return_value())

    with test.assertRaisesRegex(
        warp.codegen.WarpCodegenError,
        r"Static expression returns an unsupported value\: None is returned",
    ):
        wp.launch(invalid_kernel, 1, device=device)

    class MyClass:
        pass

    @wp.kernel
    def invalid_kernel():
        wp.static(MyClass())

    with test.assertRaisesRegex(
        warp.codegen.WarpCodegenError,
        r"Static expression returns an unsupported value\: value of type .*?MyClass",
    ):
        wp.launch(invalid_kernel, 1, device=device)


def test_function_variable(test, device):
    # create a function and pass it in as a static variable to the kernel
    @wp.func
    def func1(a: int, b: int):
        return a + b

    @wp.func
    def func2(a: int, b: int):
        return a - b

    for func in [func1, func2]:
        # note that this example also works without using wp.static()

        @wp.kernel
        def function_variable_kernel(results: wp.array(dtype=int)):
            results[0] = wp.static(func)(3, 2)  # noqa: B023

        results = wp.zeros(1, dtype=int, device=device)
        # note that the kernel has to be recompiled everytime the value of func changes
        wp.launch(function_variable_kernel, 1, [results], device=device)
        assert_np_equal(results.numpy(), np.array([func(3, 2)], dtype=int))


def test_function_lookup(test, device):
    @wp.func
    def do_add(a: float, b: float):
        return a + b

    @wp.func
    def do_sub(a: float, b: float):
        return a - b

    @wp.func
    def do_mul(a: float, b: float):
        return a * b

    op_handlers = {
        "add": do_add,
        "sub": do_sub,
        "mul": do_mul,
    }

    inputs = wp.array([[1, 2], [3, 0]], dtype=wp.float32)

    outputs = wp.empty(2, dtype=wp.float32)

    for op in op_handlers.keys():

        @wp.kernel
        def operate(input: wp.array(dtype=inputs.dtype, ndim=2), output: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            a, b = input[tid, 0], input[tid, 1]
            # retrieve the right function to use for the captured dtype variable
            output[tid] = wp.static(op_handlers[op])(a, b)  # noqa: B023

        wp.launch(operate, dim=2, inputs=[inputs], outputs=[outputs])
        outputs_np = outputs.numpy()
        inputs_np = inputs.numpy()
        for i in range(len(outputs_np)):
            test.assertEqual(outputs_np[i], op_handlers[op](float(inputs_np[i][0]), float(inputs_np[i][1])))


def count_ssa_occurrences(kernel: wp.Kernel, ssas: List[str]) -> Dict[str, int]:
    # analyze the generated code
    counts = {ssa: 0 for ssa in ssas}
    for line in kernel.adj.blocks[0].body_forward:
        for ssa in ssas:
            if ssa in line:
                counts[ssa] += 1
    return counts


def test_static_for_loop(test, device):
    @wp.kernel
    def static_loop_variable(results: wp.array(dtype=int)):
        s = 0
        for i in range(wp.static(static_global_variable_func())):
            s += wp.static(i)
        results[0] = s

    wp.set_module_options(
        options={"max_unroll": static_global_variable_func()},
    )

    results = wp.zeros(1, dtype=int, device=device)
    wp.launch(static_loop_variable, 1, [results], device=device)
    results = results.numpy()

    s = 0
    for i in range(wp.static(static_global_variable_func())):
        s += wp.static(i)

    test.assertEqual(results[0], s, "Static for loop has to compute the correct solution")

    # analyze the generated code
    if hasattr(static_loop_variable.adj, "blocks"):
        counts = count_ssa_occurrences(static_loop_variable, ["add", "for"])

        test.assertEqual(counts["add"], static_global_variable_func(), "Static for loop must be unrolled")
        # there is just one occurrence of "for" in the comment referring to the original Python code
        test.assertEqual(counts["for"], 1, "Static for loop must be unrolled")


def test_static_if_else_elif(test, device):
    @wp.kernel
    def static_condition1(results: wp.array(dtype=int)):
        if wp.static(static_global_variable_func() in {2, 3, 5}):
            results[0] = 1
        elif wp.static(static_global_variable_func() in {0, 1}):
            results[0] = 2
        else:
            results[0] = 3

    results = wp.zeros(1, dtype=int, device=device)
    wp.launch(static_condition1, 1, [results], device=device)
    results = results.numpy()
    assert_np_equal(results[0], 1)
    # TODO this needs fixing to ensure we can run these tests multiple times
    if hasattr(static_condition1.adj, "blocks"):
        counts = count_ssa_occurrences(static_condition1, ["if", "else"])

        # if, else, elif can appear as comments but the generated code must not contain
        # such keywords since the conditions are resolved at the time of code generation
        assert_np_equal(counts["if"], 1)
        assert_np_equal(counts["else"], 0)

    captured_var = "hello"

    @wp.kernel
    def static_condition2(results: wp.array(dtype=int)):
        if wp.static(captured_var == "world"):
            results[0] = 1
        else:
            results[0] = 2

    results = wp.zeros(1, dtype=int, device=device)
    wp.launch(static_condition2, 1, [results], device=device)
    results = results.numpy()
    assert_np_equal(results[0], 2)
    if hasattr(static_condition2.adj, "blocks"):
        counts = count_ssa_occurrences(static_condition2, ["if", "else"])
        assert_np_equal(counts["if"], 1)
        assert_np_equal(counts["else"], 0)

    my_list = [1, 2, 3]

    @wp.kernel
    def static_condition3(results: wp.array(dtype=int)):
        if wp.static(len(my_list) == 0):
            results[0] = 0
        elif wp.static(len(my_list) == 1):
            results[0] = 1
        elif wp.static(len(my_list) == 2):
            results[0] = 2
        elif wp.static(len(my_list) == 3):
            results[0] = 3

    results = wp.zeros(1, dtype=int, device=device)
    wp.launch(static_condition3, 1, [results], device=device)
    results = results.numpy()
    assert_np_equal(results[0], 3)
    if hasattr(static_condition3.adj, "blocks"):
        counts = count_ssa_occurrences(static_condition3, ["if", "else"])
        assert_np_equal(counts["if"], 4)
        assert_np_equal(counts["else"], 0)


devices = get_test_devices()


class TestStatic(unittest.TestCase):
    def test_static_python_call(self):
        # ensure wp.static() works from a Python context
        self.assertEqual(static_global_variable_func(), 5)


add_function_test(TestStatic, "test_static_global_variable", test_static_global_variable, devices=devices)
add_function_test(TestStatic, "test_construct_static_struct", test_construct_static_struct, devices=devices)
add_function_test(
    TestStatic, "test_construct_static_nested_struct", test_construct_static_nested_struct, devices=devices
)
add_function_test(TestStatic, "test_function_variable", test_function_variable, devices=devices)
add_function_test(TestStatic, "test_function_lookup", test_function_lookup, devices=devices)
add_function_test(TestStatic, "test_invalid_static_expression", test_invalid_static_expression, devices=devices)
add_function_test(
    TestStatic, "test_static_expression_return_types", test_static_expression_return_types, devices=devices
)
add_function_test(TestStatic, "test_static_for_loop", test_static_for_loop, devices=devices)
add_function_test(TestStatic, "test_static_if_else_elif", test_static_if_else_elif, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
