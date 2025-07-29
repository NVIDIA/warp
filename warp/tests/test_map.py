# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np

import warp as wp
import warp.context
import warp.tests.aux_test_name_clash1 as name_clash_module_1
import warp.tests.aux_test_name_clash2 as name_clash_module_2
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_cuda_test_devices, get_test_devices


@wp.struct
class MyStruct:
    a: float
    b: float


@wp.func
def add(x: MyStruct, y: MyStruct):
    s = MyStruct()
    s.a = x.a + y.a
    s.b = x.b + y.b
    return s


@wp.func
def create_struct(a: float, b: float):
    s = MyStruct()
    s.a = a
    s.b = b
    return s


def test_mixed_inputs(test, device):
    conds = wp.array([True, False, True], dtype=bool, device=device)
    out = wp.map(wp.where, conds, 1.0, 0.0)
    assert isinstance(out, wp.array)
    expected = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    assert_np_equal(out.numpy(), expected)

    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.half_pi)
    tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), rot)
    points = wp.array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], dtype=wp.vec3, device=device)
    out = wp.map(wp.transform_point, tf, points)
    assert isinstance(out, wp.array)
    expected = wp.array([(2.0, 5.0, 1.0), (5.0, 8.0, -2.0)], dtype=wp.vec3, device=device)
    assert_np_equal(out.numpy(), expected.numpy(), tol=1e-6)

    floats = wp.array([-10.0, -5.0, 0.5, 2.0, 8.0], dtype=wp.float32, device=device)
    out = wp.map(wp.clamp, floats, -0.5, 0.5)
    assert isinstance(out, wp.array)
    expected = np.array([-0.5, -0.5, 0.5, 0.5, 0.5])
    assert_np_equal(out.numpy(), expected)


def test_lambda(test, device):
    a1 = wp.array(np.arange(10, dtype=np.float32), device=device)
    out = wp.map(lambda a: a + 2.0, a1)
    assert isinstance(out, wp.array)
    expected = np.array(np.arange(10) + 2.0, dtype=np.float32)
    assert_np_equal(out.numpy(), expected)

    out = wp.map(lambda a: create_struct(a, a), a1)
    assert isinstance(out, wp.array)
    out = out.list()
    for i in range(10):
        test.assertEqual(out[i].a, i)
        test.assertEqual(out[i].b, i)

    local_var = 3.0
    out2 = wp.map(lambda a: a + local_var, a1)
    assert isinstance(out2, wp.array)
    expected = np.array(np.arange(10) + local_var, dtype=np.float32)
    assert_np_equal(out2.numpy(), expected)

    local_var = 3.0

    @wp.func
    def my_func(a: float):
        return a * local_var

    out = wp.empty_like(a1)
    wp.map(lambda a: (a + local_var, my_func(a)), a1, out=[out, out2])
    expected = np.array(np.arange(10) + local_var, dtype=np.float32)
    assert_np_equal(out.numpy(), expected)
    expected = np.array(np.arange(10) * local_var, dtype=np.float32)
    assert_np_equal(out2.numpy(), expected)


def test_multiple_return_values(test, device):
    @wp.func
    def multiple_return(a: float):
        return a + 2.0, a + 3.0, wp.vec3(a, a, a)

    a1 = wp.array(np.arange(10, dtype=np.float32), device=device)
    out = wp.map(multiple_return, a1)
    assert isinstance(out, list)
    out = [o.list() for o in out]
    for i in range(10):
        test.assertEqual(out[0][i], i + 2.0)
        test.assertEqual(out[1][i], i + 3.0)
        test.assertEqual(out[2][i].x, i)
        test.assertEqual(out[2][i].y, i)
        test.assertEqual(out[2][i].z, i)

    out = wp.map(lambda a: multiple_return(a), a1)
    assert isinstance(out, list)
    out = [o.list() for o in out]
    for i in range(10):
        test.assertEqual(out[0][i], i + 2.0)
        test.assertEqual(out[1][i], i + 3.0)
        test.assertEqual(out[2][i].x, i)
        test.assertEqual(out[2][i].y, i)
        test.assertEqual(out[2][i].z, i)


def test_custom_struct_operator(test, device):
    x1, x2 = MyStruct(), MyStruct()
    x1.a = 1.0
    x1.b = 2.0
    x2.a = 3.0
    x2.b = 4.0
    y1, y2 = MyStruct(), MyStruct()
    y1.a = 10.0
    y1.b = 20.0
    y2.a = 30.0
    y2.b = 40.0
    xs = wp.array([x1, x2], dtype=MyStruct, device=device)
    ys = wp.array([y1, y2], dtype=MyStruct, device=device)
    zs = wp.map(add, xs, ys)
    assert isinstance(zs, wp.array)
    zs = zs.list()
    test.assertEqual(zs[0].a, 11.0)
    test.assertEqual(zs[0].b, 22.0)
    test.assertEqual(zs[1].a, 33.0)
    test.assertEqual(zs[1].b, 44.0)


def test_name_clash(test, device):
    vec5 = wp.types.vector(5, dtype=wp.float32)

    @wp.func
    def name_clash_structs_args_func(
        s1: name_clash_module_1.SameStruct,
        s2: name_clash_module_2.SameStruct,
        d1: name_clash_module_1.DifferentStruct,
        d2: name_clash_module_2.DifferentStruct,
    ):
        return vec5(s1.x, s2.x, d1.v, d2.v[0], d2.v[1])

    s1 = name_clash_module_1.SameStruct()
    s2 = name_clash_module_2.SameStruct()
    d1 = name_clash_module_1.DifferentStruct()
    d2 = name_clash_module_2.DifferentStruct()
    s1.x = 1.0
    s2.x = 2.0
    d1.v = 3.0
    d2.v = wp.vec2(4.0, 5.0)
    s1s = wp.array([s1], dtype=name_clash_module_1.SameStruct, device=device)
    s2s = wp.array([s2], dtype=name_clash_module_2.SameStruct, device=device)
    d1s = wp.array([d1], dtype=name_clash_module_1.DifferentStruct, device=device)
    d2s = wp.array([d2], dtype=name_clash_module_2.DifferentStruct, device=device)
    out = wp.map(name_clash_structs_args_func, s1s, s2s, d1s, d2s)
    assert isinstance(out, wp.array)
    assert_np_equal(out.numpy(), np.array([[1, 2, 3, 4, 5]], dtype=np.float32))


def test_gradient(test, device):
    @wp.func
    def my_func(a: float):
        return 2.0 * a - 10.0

    a = wp.array(np.arange(10, dtype=np.float32), requires_grad=True, device=device)
    assert a.grad is not None
    tape = wp.Tape()
    with tape:
        out = wp.map(my_func, a)
    assert isinstance(out, wp.array)
    assert out.grad is not None
    out.grad.fill_(1.0)
    tape.backward()
    expected = np.full(10, 2.0, dtype=np.float32)
    assert_np_equal(a.grad.numpy(), expected)
    a.grad *= 2.0
    assert_np_equal(a.grad.numpy(), expected * 2.0)


def test_array_ops(test, device):
    a = wp.array(np.arange(10, dtype=np.float32), device=device)
    b = wp.array(np.arange(1, 11, dtype=np.float32), device=device)
    a_np = a.numpy()
    b_np = b.numpy()
    assert_np_equal((+a).numpy(), a_np)
    assert_np_equal((-a).numpy(), -a_np)
    assert_np_equal((a + b).numpy(), a_np + b_np)
    assert_np_equal((a + 2.0).numpy(), a_np + 2.0)
    assert_np_equal((a - b).numpy(), a_np - b_np)
    assert_np_equal((a - 2.0).numpy(), a_np - 2.0)
    assert_np_equal((2.0 - a).numpy(), 2.0 - a_np)
    assert_np_equal((a * b).numpy(), a_np * b_np)
    assert_np_equal((2.0 * a).numpy(), 2.0 * a_np)
    assert_np_equal((a * 2.0).numpy(), a_np * 2.0)
    np.testing.assert_allclose((a**b).numpy(), a_np**b_np, rtol=1.5e-7)
    np.testing.assert_allclose((a**2.0).numpy(), a_np**2.0)
    assert_np_equal((a / b).numpy(), a_np / b_np)
    assert_np_equal((a / 2.0).numpy(), a_np / 2.0)
    assert_np_equal((a // b).numpy(), a_np // b_np)
    assert_np_equal((a // 2.0).numpy(), a_np // 2.0)
    assert_np_equal((2.0 / b).numpy(), 2.0 / b_np)
    ai = wp.array(np.arange(10, dtype=np.int32), device=device)
    bi = wp.array(np.arange(1, 11, dtype=np.int32), device=device)
    ai_np = ai.numpy()
    bi_np = bi.numpy()
    div = ai / bi
    # XXX note in Warp div on int32 is integer division
    test.assertEqual(div.dtype, wp.int32)
    assert_np_equal(div.numpy(), ai_np // bi_np)

    @wp.func
    def make_vec(a: float):
        return wp.vec3(a, a + 1.0, a + 2.0)

    vecs_a = wp.map(make_vec, a)
    vecs_b = wp.map(make_vec, b)
    assert isinstance(vecs_a, wp.array)
    assert isinstance(vecs_b, wp.array)
    vecs_a_np = vecs_a.numpy()
    vecs_b_np = vecs_b.numpy()
    assert_np_equal((-vecs_a).numpy(), -vecs_a_np)
    assert_np_equal((vecs_a + vecs_b).numpy(), vecs_a_np + vecs_b_np)
    assert_np_equal((vecs_a * 2.0).numpy(), vecs_a_np * 2.0)


def test_indexedarrays(test, device):
    arr = wp.array(data=np.arange(10, dtype=np.float32), device=device)
    indices = wp.array([1, 3, 5, 7, 9], dtype=int, device=device)
    iarr = wp.indexedarray1d(arr, [indices])
    out = wp.map(lambda x: x * 10.0, iarr)
    assert isinstance(out, wp.array)
    expected = np.array([10.0, 30.0, 50.0, 70.0, 90.0], dtype=np.float32)
    assert_np_equal(out.numpy(), expected)
    wp.map(lambda x: x * 10.0, iarr, out=iarr)
    assert isinstance(iarr, wp.indexedarray)
    assert_np_equal(iarr.numpy(), expected)

    newarr = 10.0 * iarr
    assert isinstance(newarr, wp.array)
    expected = np.array([100.0, 300.0, 500.0, 700.0, 900.0], dtype=np.float32)
    assert_np_equal(newarr.numpy(), expected)
    iarr += 1.0
    assert isinstance(iarr, wp.indexedarray)
    expected = np.array([11.0, 31.0, 51.0, 71.0, 91.0], dtype=np.float32)
    assert_np_equal(iarr.numpy(), expected)


def test_broadcasting(test, device):
    a = wp.array(np.zeros((1, 3, 1, 4), dtype=np.float32), device=device)
    b = wp.array(np.ones((5, 4), dtype=np.float32), device=device)
    out = wp.map(wp.add, a, b)
    assert isinstance(out, wp.array)
    test.assertEqual(out.shape, (1, 3, 5, 4))
    expected = np.ones((1, 3, 5, 4), dtype=np.float32)
    assert_np_equal(out.numpy(), expected)

    out = wp.map(wp.add, b, a)
    assert isinstance(out, wp.array)
    expected = np.ones((1, 3, 5, 4), dtype=np.float32)
    assert_np_equal(out.numpy(), expected)

    c = wp.array(np.ones((2, 3, 5, 4), dtype=np.float32), device=device)
    out = wp.map(lambda a, b, c: a + b + c, a, b, c)
    assert isinstance(out, wp.array)
    test.assertEqual(out.shape, (2, 3, 5, 4))
    expected = np.ones((2, 3, 5, 4), dtype=np.float32) * 2.0
    assert_np_equal(out.numpy(), expected)


def test_input_validity(test, device):
    @wp.func
    def empty_function(f: float):
        pass

    a1 = wp.empty(3, dtype=wp.float32)
    with test.assertRaisesRegex(
        TypeError,
        "The provided function must return a value$",
    ):
        wp.map(empty_function, a1)

    @wp.func
    def unary_function(f: float):
        return 2.0 * f

    with test.assertRaisesRegex(
        TypeError,
        r"Number of input arguments \(2\) does not match expected number of function arguments \(1\)$",
    ):
        wp.map(unary_function, a1, a1)

    @wp.func
    def int_function(i: int):
        return 5.0 * float(i)

    with test.assertRaisesRegex(
        TypeError,
        "Function test_input_validity__locals__int_function does not support the provided argument types float32",
    ):
        wp.map(int_function, a1)

    i1 = wp.zeros((3, 2, 1), dtype=wp.float32)
    i2 = wp.ones((3, 3, 2), dtype=wp.float32)
    with test.assertRaisesRegex(
        ValueError,
        r"Shapes \(3, 2, 1\) and \(3, 3, 2\) are not broadcastable$",
    ):
        wp.map(wp.add, i1, i2)

    xs = wp.zeros(3, dtype=wp.float32)
    ys = wp.zeros(5, dtype=wp.float32)
    with test.assertRaisesRegex(
        ValueError,
        r"Shapes \(3,\) and \(5,\) are not broadcastable$",
    ):
        wp.map(lambda a, b, c: a + b * c, 0.0, xs, ys)

    with test.assertRaisesRegex(
        ValueError,
        "map requires at least one warp.array input$",
    ):
        wp.map(lambda a, b, c: a * b * c, 2.0, 0.4, [5.0])


def test_output_validity(test, device):
    xs = wp.zeros(3, dtype=wp.float32)
    ys = wp.ones(3, dtype=wp.float32)
    out = wp.empty(2, dtype=wp.float32)
    with test.assertRaisesRegex(
        TypeError,
        r"Output array shape \(2,\) does not match expected shape \(3,\)$",
    ):
        wp.map(wp.sub, xs, ys, out=out)

    out = wp.empty((2, 3), dtype=wp.float32)
    with test.assertRaisesRegex(
        TypeError,
        r"Invalid output provided, expected 2 Warp arrays with shape \(3,\) and dtypes \(float32, float32\)$",
    ):
        wp.map(lambda x, y: (x, y), xs, ys, out=out)

    out = wp.empty(3, dtype=wp.int32)
    with test.assertRaisesRegex(
        TypeError,
        "Output array dtype int32 does not match expected dtype float32$",
    ):
        wp.map(lambda x, y: x - y, xs, ys, out=out)

    out = wp.empty((3, 1), dtype=wp.float32)
    with test.assertRaisesRegex(
        TypeError,
        r"Output array shape \(3, 1\) does not match expected shape \(3,\)$",
    ):
        wp.map(wp.mul, xs, ys, out=out)

    out1 = wp.empty(3, dtype=wp.float32)
    out2 = wp.empty(3, dtype=wp.int32)
    with test.assertRaisesRegex(
        TypeError,
        "Output array 1 dtype int32 does not match expected dtype float32$",
    ):
        wp.map(lambda x, y: (x, y), xs, ys, out=[out1, out2])

    out1 = wp.empty(3, dtype=wp.float32)
    out2 = wp.empty(3, dtype=wp.float32)
    out3 = wp.empty(1, dtype=wp.float32)
    with test.assertRaisesRegex(
        TypeError,
        r"Number of provided output arrays \(3\) does not match expected number of function outputs \(2\)$",
    ):
        wp.map(lambda x, y: (x, y), xs, ys, out=[out1, out2, out3])

    out1 = wp.empty(3, dtype=wp.float32)
    out2 = wp.empty((3, 1), dtype=wp.float32)
    with test.assertRaisesRegex(
        TypeError,
        r"Output array 1 shape \(3, 1\) does not match expected shape \(3,\)$",
    ):
        wp.map(lambda x, y: (x, y), xs, ys, out=[out1, out2])


def test_kernel_creation(test, device):
    a = wp.array(np.arange(10, dtype=np.float32), device=device)
    kernel = wp.map(lambda a: a + 2.0, a, return_kernel=True)
    test.assertIsInstance(kernel, wp.Kernel)

    b = wp.zeros(20)
    out = wp.empty_like(b)
    wp.launch(kernel, dim=len(b), inputs=[b], outputs=[out])
    expected = np.full(20, 2.0, dtype=np.float32)
    assert_np_equal(out.numpy(), expected)


def test_graph_capture(test, device):
    assert warp.context.runtime.driver_version is not None
    if warp.context.runtime.driver_version < (12, 3):
        test.skipTest("Module loading during CUDA graph capture is not supported on driver versions < 12.3")
    a_np = np.arange(10, dtype=np.float32)
    b_np = np.arange(1, 11, dtype=np.float32)
    a = wp.array(a_np, device=device)
    b = wp.array(b_np, device=device)
    with wp.ScopedCapture(device, force_module_load=False) as capture:
        out = wp.map(lambda x, y: wp.abs(2.0 * x - y), a, b)
        out = wp.map(wp.sin, out)
        assert isinstance(out, wp.array)
        out *= 2.0
    expected = np.array(2.0 * np.sin(np.abs(a_np * 2.0 - b_np)), dtype=np.float32)
    assert capture.graph is not None
    wp.capture_launch(capture.graph)
    assert_np_equal(out.numpy(), expected, tol=1e-6)


def test_renamed_warp_module(test, device):
    import warp as uncommon_name

    @wp.func
    def my_func(a: float):
        return uncommon_name.abs(2.0 * a - 10.0)

    a = wp.array(np.arange(10, dtype=np.float32), device=device)
    b = wp.array(np.arange(1, 11, dtype=np.float32), device=device)
    out = wp.map(lambda x, y: uncommon_name.abs(2.0 * x - y), a, b)
    assert isinstance(out, wp.array)
    expected = np.array(np.abs(a.numpy() * 2.0 - b.numpy()), dtype=np.float32)
    assert_np_equal(out.numpy(), expected, tol=1e-6)
    out = wp.map(my_func, a)
    assert isinstance(out, wp.array)
    expected = np.array(np.abs(a.numpy() * 2.0 - 10.0), dtype=np.float32)
    assert_np_equal(out.numpy(), expected, tol=1e-6)


devices = get_test_devices("basic")
cuda_test_devices = get_cuda_test_devices()


class TestMap(unittest.TestCase):
    pass


add_function_test(TestMap, "test_mixed_inputs", test_mixed_inputs, devices=devices)
add_function_test(TestMap, "test_lambda", test_lambda, devices=devices)
add_function_test(TestMap, "test_multiple_return_values", test_multiple_return_values, devices=devices)
add_function_test(TestMap, "test_custom_struct_operator", test_custom_struct_operator, devices=devices)
add_function_test(TestMap, "test_name_clash", test_name_clash, devices=devices)
add_function_test(TestMap, "test_gradient", test_gradient, devices=devices)
add_function_test(TestMap, "test_array_ops", test_array_ops, devices=devices)
add_function_test(TestMap, "test_indexedarrays", test_indexedarrays, devices=devices)
add_function_test(TestMap, "test_broadcasting", test_broadcasting, devices=devices)
add_function_test(TestMap, "test_input_validity", test_input_validity, devices=devices)
add_function_test(TestMap, "test_output_validity", test_output_validity, devices=devices)
add_function_test(TestMap, "test_kernel_creation", test_kernel_creation, devices=devices)
add_function_test(TestMap, "test_graph_capture", test_graph_capture, devices=cuda_test_devices)
add_function_test(TestMap, "test_renamed_warp_module", test_renamed_warp_module, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
