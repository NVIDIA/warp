# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.func
def generic_adder(a: Any, b: Any):
    return a + b


@wp.kernel
def test_generic_adder():
    wp.expect_eq(generic_adder(17, 25), 42)
    wp.expect_eq(generic_adder(7.0, 10.0), 17.0)
    v1 = wp.vec3(1.0, 2.0, 3.0)
    v2 = wp.vec3(10.0, 20.0, 30.0)
    wp.expect_eq(generic_adder(v1, v2), wp.vec3(11.0, 22.0, 33.0))


# regular functions for floats
@wp.func
def specialized_func(a: float, b: float):
    return a * b


@wp.func
def specialized_func(a: float, b: float, c: float):
    return a * b * c


# generic forms
@wp.func
def specialized_func(a: Any, b: Any):
    return a + b


@wp.func
def specialized_func(a: Any, b: Any, c: Any):
    return a + b + c


# specializations for ints
@wp.func
def specialized_func(a: int, b: int):
    return a - b


@wp.func
def specialized_func(a: int, b: int, c: int):
    return a - b - c


@wp.kernel
def test_specialized_func():
    # subtraction with int args
    wp.expect_eq(specialized_func(17, 25), -8)
    wp.expect_eq(specialized_func(17, 25, 10), -18)
    # multiplication with float args
    wp.expect_eq(specialized_func(7.0, 10.0), 70.0)
    wp.expect_eq(specialized_func(7.0, 10.0, 2.0), 140.0)
    # addition with vector args
    v1 = wp.vec3(1.0, 2.0, 3.0)
    v2 = wp.vec3(10.0, 20.0, 30.0)
    v3 = wp.vec3(100.0, 200.0, 300.0)
    wp.expect_eq(specialized_func(v1, v2), wp.vec3(11.0, 22.0, 33.0))
    wp.expect_eq(specialized_func(v1, v2, v3), wp.vec3(111.0, 222.0, 333.0))


# generic array kernel, version 1 (Any)
@wp.kernel
def generic_array_kernel_v1(a: Any, b: Any, c: Any):
    tid = wp.tid()
    sum = a[tid] + b[tid]  # test direct access
    c[tid] = generic_adder(sum, sum)  # test generic function


wp.overload(generic_array_kernel_v1, [wp.array(dtype=int), wp.array(dtype=int), wp.array(dtype=int)])
wp.overload(generic_array_kernel_v1, [wp.array(dtype=float), wp.array(dtype=float), wp.array(dtype=float)])
wp.overload(generic_array_kernel_v1, [wp.array(dtype=wp.vec3), wp.array(dtype=wp.vec3), wp.array(dtype=wp.vec3)])


# generic array kernel, version 2 (generic dtype)
@wp.kernel
def generic_array_kernel_v2(a: wp.array(dtype=Any), b: wp.array(dtype=Any), c: wp.array(dtype=Any)):
    tid = wp.tid()
    sum = a[tid] + b[tid]  # test direct access
    c[tid] = generic_adder(sum, sum)  # test generic function


wp.overload(generic_array_kernel_v2, [wp.array(dtype=int), wp.array(dtype=int), wp.array(dtype=int)])
wp.overload(generic_array_kernel_v2, [wp.array(dtype=float), wp.array(dtype=float), wp.array(dtype=float)])
wp.overload(generic_array_kernel_v2, [wp.array(dtype=wp.vec3), wp.array(dtype=wp.vec3), wp.array(dtype=wp.vec3)])


# generic array kernel, version 3 (unspecified dtype)
@wp.kernel
def generic_array_kernel_v3(a: wp.array(), b: wp.array(), c: wp.array()):
    tid = wp.tid()
    sum = a[tid] + b[tid]  # test direct access
    c[tid] = generic_adder(sum, sum)  # test generic function


wp.overload(generic_array_kernel_v3, [wp.array(dtype=int), wp.array(dtype=int), wp.array(dtype=int)])
wp.overload(generic_array_kernel_v3, [wp.array(dtype=float), wp.array(dtype=float), wp.array(dtype=float)])
wp.overload(generic_array_kernel_v3, [wp.array(dtype=wp.vec3), wp.array(dtype=wp.vec3), wp.array(dtype=wp.vec3)])


def test_generic_array_kernel(test, device):
    with wp.ScopedDevice(device):
        n = 10

        ai = wp.array(data=np.ones(n, dtype=np.int32))
        ci = wp.empty(10, dtype=int)

        af = wp.array(data=np.ones(n, dtype=np.float32))
        cf = wp.empty(10, dtype=float)

        a3 = wp.array(data=np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
        c3 = wp.empty(n, dtype=wp.vec3)

        wp.launch(generic_array_kernel_v1, dim=n, inputs=[af, af, cf])
        wp.launch(generic_array_kernel_v1, dim=n, inputs=[ai, ai, ci])
        wp.launch(generic_array_kernel_v1, dim=n, inputs=[a3, a3, c3])
        assert_np_equal(ci.numpy(), np.full((n,), 4, dtype=np.int32))
        assert_np_equal(cf.numpy(), np.full((n,), 4.0, dtype=np.float32))
        assert_np_equal(c3.numpy(), np.full((n, 3), 4.0, dtype=np.float32))

        wp.launch(generic_array_kernel_v2, dim=n, inputs=[af, af, cf])
        wp.launch(generic_array_kernel_v2, dim=n, inputs=[ai, ai, ci])
        wp.launch(generic_array_kernel_v2, dim=n, inputs=[a3, a3, c3])
        assert_np_equal(ci.numpy(), np.full((n,), 4, dtype=np.int32))
        assert_np_equal(cf.numpy(), np.full((n,), 4.0, dtype=np.float32))
        assert_np_equal(c3.numpy(), np.full((n, 3), 4.0, dtype=np.float32))

        wp.launch(generic_array_kernel_v3, dim=n, inputs=[af, af, cf])
        wp.launch(generic_array_kernel_v3, dim=n, inputs=[ai, ai, ci])
        wp.launch(generic_array_kernel_v3, dim=n, inputs=[a3, a3, c3])
        assert_np_equal(ci.numpy(), np.full((n,), 4, dtype=np.int32))
        assert_np_equal(cf.numpy(), np.full((n,), 4.0, dtype=np.float32))
        assert_np_equal(c3.numpy(), np.full((n, 3), 4.0, dtype=np.float32))


# kernel that adds any scalar value to an array
@wp.kernel
def generic_accumulator_kernel(a: wp.array(dtype=wp.float64), value: Any):
    tid = wp.tid()
    a[tid] = a[tid] + wp.float64(value)


# overload named args
wp.overload(generic_accumulator_kernel, {"value": int})
wp.overload(generic_accumulator_kernel, {"value": float})
wp.overload(generic_accumulator_kernel, {"value": wp.float64})
wp.overload(generic_accumulator_kernel, {"value": wp.bool})


def test_generic_accumulator_kernel(test, device):
    with wp.ScopedDevice(device):
        n = 10
        a = wp.zeros(n, dtype=wp.float64)

        wp.launch(generic_accumulator_kernel, dim=a.size, inputs=[a, 25])
        wp.launch(generic_accumulator_kernel, dim=a.size, inputs=[a, 17.0])
        wp.launch(generic_accumulator_kernel, dim=a.size, inputs=[a, wp.float64(8.0)])
        wp.launch(generic_accumulator_kernel, dim=a.size, inputs=[a, wp.bool(True)])

        assert_np_equal(a.numpy(), np.full((n,), 51.0, dtype=np.float64))


# generic kernel used to automatically generate overloads from launch args
@wp.kernel
def generic_fill(a: wp.array(dtype=Any), value: Any):
    tid = wp.tid()
    a[tid] = value


def test_generic_fill(test, device):
    with wp.ScopedDevice(device):
        n = 10
        ai = wp.zeros(n, dtype=int)
        af = wp.zeros(n, dtype=float)
        a3 = wp.zeros(n, dtype=wp.vec3)
        ab = wp.zeros(n, dtype=wp.bool)

        wp.launch(generic_fill, dim=ai.size, inputs=[ai, 42])
        wp.launch(generic_fill, dim=af.size, inputs=[af, 17.0])
        wp.launch(generic_fill, dim=a3.size, inputs=[a3, wp.vec3(5.0, 5.0, 5.0)])
        wp.launch(generic_fill, dim=ab.size, inputs=[ab, wp.bool(True)])

        assert_np_equal(ai.numpy(), np.full((n,), 42, dtype=np.int32))
        assert_np_equal(af.numpy(), np.full((n,), 17.0, dtype=np.float32))
        assert_np_equal(a3.numpy(), np.full((n, 3), 5.0, dtype=np.float32))
        assert_np_equal(ab.numpy(), np.full((n,), True, dtype=np.bool_))


# generic kernel used to create and launch explicit overloads
@wp.kernel
def generic_fill_v2(a: wp.array(dtype=Any), value: Any):
    tid = wp.tid()
    a[tid] = value


vec3b_type = wp.vec(3, wp.bool)
# create explicit overloads to be launched directly
fill_int = wp.overload(generic_fill_v2, [wp.array(dtype=int), int])
fill_float = wp.overload(generic_fill_v2, [wp.array(dtype=float), float])
fill_vec3 = wp.overload(generic_fill_v2, [wp.array(dtype=wp.vec3), wp.vec3])
fill_vec3b = wp.overload(generic_fill_v2, [wp.array(dtype=vec3b_type), vec3b_type])


def test_generic_fill_overloads(test, device):
    with wp.ScopedDevice(device):
        n = 10
        ai = wp.zeros(n, dtype=int)
        af = wp.zeros(n, dtype=float)
        a3 = wp.zeros(n, dtype=wp.vec3)
        a3b = wp.zeros(n, dtype=vec3b_type)

        wp.launch(fill_int, dim=ai.size, inputs=[ai, 42])
        wp.launch(fill_float, dim=af.size, inputs=[af, 17.0])
        wp.launch(fill_vec3, dim=a3.size, inputs=[a3, wp.vec3(5.0, 5.0, 5.0)])
        wp.launch(fill_vec3b, dim=a3b.size, inputs=[a3b, vec3b_type([True, True, True])])

        assert_np_equal(ai.numpy(), np.full((n,), 42, dtype=np.int32))
        assert_np_equal(af.numpy(), np.full((n,), 17.0, dtype=np.float32))
        assert_np_equal(a3.numpy(), np.full((n, 3), 5.0, dtype=np.float32))
        assert_np_equal(a3b.numpy(), np.full((n, 3), True, dtype=np.bool_))


# generic kernel used to test generic types mixed with specialized types
@wp.func
def generic_conditional_setter_func(a: wp.array(dtype=Any), i: int, value: Any, relative: bool):
    if relative:
        a[i] += value
    else:
        a[i] = value


@wp.kernel
def generic_conditional_setter(a: wp.array(dtype=Any), i: int, value: Any, relative: bool):
    generic_conditional_setter_func(a, i, value, relative)


def test_generic_conditional_setter(test, device):
    with wp.ScopedDevice(device):
        n = 10
        ai = wp.zeros(n, dtype=int)

        wp.launch(generic_conditional_setter, dim=1, inputs=[ai, 1, 42, False])
        wp.launch(generic_conditional_setter, dim=1, inputs=[ai, 1, 5, True])
        ai_true = np.zeros(n, dtype=np.int32)
        ai_true[1] = 47
        assert_np_equal(ai.numpy(), ai_true)


# custom vector/matrix types
my_vec5 = wp.vec(length=5, dtype=wp.float32)
my_mat55 = wp.mat(shape=(5, 5), dtype=wp.float32)


@wp.kernel
def generic_transform(v: Any, m: Any, expected: Any):
    result = wp.mul(m, v)
    wp.expect_eq(result, expected)


# use overload decorator syntax
@wp.overload
def generic_transform(v: wp.vec2, m: wp.mat22, expected: wp.vec2):  # fmt: skip
    ...


@wp.overload
def generic_transform(v: wp.vec3, m: wp.mat33, expected: wp.vec3):  # fmt: skip
    ...


@wp.overload
def generic_transform(v: wp.vec4, m: wp.mat44, expected: wp.vec4):  # fmt: skip
    ...


@wp.overload
def generic_transform(v: my_vec5, m: my_mat55, expected: my_vec5):  # fmt: skip
    ...


def test_generic_transform_kernel(test, device):
    with wp.ScopedDevice(device):
        v2 = wp.vec2(1, 2)
        m22 = wp.mat22(2, 0, 0, 2)
        e2 = wp.vec2(2, 4)

        v3 = wp.vec3(1, 2, 3)
        m33 = wp.mat33(2, 0, 0, 0, 2, 0, 0, 0, 2)
        e3 = wp.vec3(2, 4, 6)

        v4 = wp.vec4(1, 2, 3, 4)
        m44 = wp.mat44(2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2)
        e4 = wp.vec4(2, 4, 6, 8)

        v5 = my_vec5(1, 2, 3, 4, 5)
        m55 = my_mat55(2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2)
        e5 = my_vec5(2, 4, 6, 8, 10)

        wp.launch(generic_transform, dim=1, inputs=[v2, m22, e2])
        wp.launch(generic_transform, dim=1, inputs=[v3, m33, e3])
        wp.launch(generic_transform, dim=1, inputs=[v4, m44, e4])
        wp.launch(generic_transform, dim=1, inputs=[v5, m55, e5])

        wp.synchronize()


@wp.kernel
def generic_transform_array(v: wp.array(), m: wp.array(), result: wp.array()):
    tid = wp.tid()
    result[tid] = wp.mul(m[tid], v[tid])


wp.overload(generic_transform_array, [wp.array(dtype=wp.vec2), wp.array(dtype=wp.mat22), wp.array(dtype=wp.vec2)])
wp.overload(generic_transform_array, [wp.array(dtype=wp.vec3), wp.array(dtype=wp.mat33), wp.array(dtype=wp.vec3)])
wp.overload(generic_transform_array, [wp.array(dtype=wp.vec4), wp.array(dtype=wp.mat44), wp.array(dtype=wp.vec4)])
wp.overload(generic_transform_array, [wp.array(dtype=my_vec5), wp.array(dtype=my_mat55), wp.array(dtype=my_vec5)])


def test_generic_transform_array_kernel(test, device):
    with wp.ScopedDevice(device):
        n = 10

        a2_data = np.tile(np.arange(2, dtype=np.float32), (n, 1))
        a3_data = np.tile(np.arange(3, dtype=np.float32), (n, 1))
        a4_data = np.tile(np.arange(4, dtype=np.float32), (n, 1))
        a5_data = np.tile(np.arange(5, dtype=np.float32), (n, 1))

        m22_data = np.tile((np.identity(2, dtype=np.float32) * 2), (n, 1, 1))
        m33_data = np.tile((np.identity(3, dtype=np.float32) * 2), (n, 1, 1))
        m44_data = np.tile((np.identity(4, dtype=np.float32) * 2), (n, 1, 1))
        m55_data = np.tile((np.identity(5, dtype=np.float32) * 2), (n, 1, 1))

        a2 = wp.array(data=a2_data, dtype=wp.vec2)
        a3 = wp.array(data=a3_data, dtype=wp.vec3)
        a4 = wp.array(data=a4_data, dtype=wp.vec4)
        a5 = wp.array(data=a5_data, dtype=my_vec5)

        m22 = wp.array(data=m22_data, dtype=wp.mat22)
        m33 = wp.array(data=m33_data, dtype=wp.mat33)
        m44 = wp.array(data=m44_data, dtype=wp.mat44)
        m55 = wp.array(data=m55_data, dtype=my_mat55)

        b2 = wp.zeros_like(a2)
        b3 = wp.zeros_like(a3)
        b4 = wp.zeros_like(a4)
        b5 = wp.zeros_like(a5)

        wp.launch(generic_transform_array, dim=n, inputs=[a2, m22, b2])
        wp.launch(generic_transform_array, dim=n, inputs=[a3, m33, b3])
        wp.launch(generic_transform_array, dim=n, inputs=[a4, m44, b4])
        wp.launch(generic_transform_array, dim=n, inputs=[a5, m55, b5])

        assert_np_equal(b2.numpy(), a2_data * 2)
        assert_np_equal(b3.numpy(), a3_data * 2)
        assert_np_equal(b4.numpy(), a4_data * 2)
        assert_np_equal(b5.numpy(), a5_data * 2)


@wp.struct
class Foo:
    x: float
    y: float
    z: float


@wp.struct
class Bar:
    x: wp.vec3
    y: wp.vec3
    z: wp.vec3


@wp.kernel
def test_generic_struct_kernel(s: Any):
    # test member access for generic structs
    wp.expect_eq(s.x + s.y, s.z)


wp.overload(test_generic_struct_kernel, [Foo])
wp.overload(test_generic_struct_kernel, [Bar])


@wp.kernel
def test_generic_type_cast_kernel(a: Any, b: Any):
    a = type(a)(b)
    c = type(generic_adder(b, b))(a)
    wp.expect_eq(b, c)


wp.overload(test_generic_type_cast_kernel, [wp.float32, wp.float64])
wp.overload(test_generic_type_cast_kernel, [wp.float32, wp.int32])
wp.overload(test_generic_type_cast_kernel, [wp.vec3f, wp.vec3d])
wp.overload(test_generic_type_cast_kernel, [wp.mat22f, wp.mat22d])


def test_generic_type_cast(test, device):
    with wp.ScopedDevice(device):
        wp.launch(test_generic_type_cast_kernel, dim=1, inputs=[1.0, 2.0])
        wp.launch(test_generic_type_cast_kernel, dim=1, inputs=[2.0, -5])
        wp.launch(test_generic_type_cast_kernel, dim=1, inputs=[wp.vec3f(1.0, 2.0, 3.0), wp.vec3d(4.0, 5.0, 6.0)])
        wp.launch(test_generic_type_cast_kernel, dim=1, inputs=[wp.mat22f(0.0), wp.mat22d(np.eye(2))])

        wp.synchronize()


@wp.kernel
def test_generic_scalar_construction_kernel(a: wp.array(dtype=Any)):
    zero = type(a[0])(0)
    copy = a.dtype(a[0])
    copy += zero
    wp.expect_eq(copy, a[0])


wp.overload(test_generic_scalar_construction_kernel, [wp.array(dtype=wp.int32)])
wp.overload(test_generic_scalar_construction_kernel, [wp.array(dtype=wp.float64)])


def test_generic_scalar_construction(test, device):
    with wp.ScopedDevice(device):
        wp.launch(test_generic_scalar_construction_kernel, dim=1, inputs=[wp.array([1.0], dtype=wp.int32)])
        wp.launch(test_generic_scalar_construction_kernel, dim=1, inputs=[wp.array([-5], dtype=wp.float64)])

        wp.synchronize()


@wp.kernel
def test_generic_type_construction_kernel(a: wp.array(dtype=Any)):
    zero = type(a[0])()
    copy = type(a).dtype(a[0]) * a.dtype.dtype(1.0)
    copy += zero
    wp.expect_eq(copy, a[0])


wp.overload(test_generic_type_construction_kernel, [wp.array(dtype=wp.vec3f)])
wp.overload(test_generic_type_construction_kernel, [wp.array(dtype=wp.mat22d)])


def test_generic_type_construction(test, device):
    with wp.ScopedDevice(device):
        wp.launch(test_generic_type_construction_kernel, dim=1, inputs=[wp.array([1.0, 2.0, 3.0], dtype=wp.vec3f)])
        wp.launch(test_generic_type_construction_kernel, dim=1, inputs=[wp.array([np.eye(2)], dtype=wp.mat22d)])

        wp.synchronize()


@wp.kernel
def test_generic_struct_construction_kernel(a: Any):
    b = type(a)(a.x, a.y, a.z)
    wp.expect_eq(a.x, b.x)
    wp.expect_eq(a.y, b.y)
    wp.expect_eq(a.z, b.z)


wp.overload(test_generic_struct_construction_kernel, [Foo])
wp.overload(test_generic_struct_construction_kernel, [Bar])


@wp.kernel
def test_generic_type_as_argument_kernel(a: Any):
    vec = wp.vector(length=2, dtype=type(a))
    matrix = wp.identity(n=vec.length, dtype=vec.dtype) * a
    wp.expect_eq(wp.trace(matrix), type(a)(2.0) * a)


wp.overload(test_generic_type_as_argument_kernel, [wp.float32])
wp.overload(test_generic_type_as_argument_kernel, [wp.float64])


def test_generic_type_as_argument(test, device):
    with wp.ScopedDevice(device):
        wp.launch(test_generic_type_as_argument_kernel, dim=1, inputs=[2.0])
        wp.launch(test_generic_type_as_argument_kernel, dim=1, inputs=[-1.0])

        wp.synchronize()


def test_type_operator_misspell(test, device):
    @wp.kernel
    def kernel():
        i = wp.tid()
        _ = typez(i)(0)

    with test.assertRaisesRegex(KeyError, r"Referencing undefined symbol: typez"):
        wp.launch(
            kernel,
            dim=1,
            inputs=[],
            device=device,
        )


def test_type_attribute_error(test, device):
    @wp.kernel
    def kernel():
        a = wp.vec3(0.0)
        _ = a.dtype.shape

    with test.assertRaisesRegex(AttributeError, r"`shape` is not an attribute of '<class 'warp.types.float32'>'"):
        wp.launch(
            kernel,
            dim=1,
            inputs=[],
            device=device,
        )


class TestGenerics(unittest.TestCase):
    pass


devices = get_test_devices()

add_kernel_test(TestGenerics, name="test_generic_adder", kernel=test_generic_adder, dim=1, devices=devices)
add_kernel_test(TestGenerics, name="test_specialized_func", kernel=test_specialized_func, dim=1, devices=devices)

add_function_test(TestGenerics, "test_generic_array_kernel", test_generic_array_kernel, devices=devices)
add_function_test(TestGenerics, "test_generic_accumulator_kernel", test_generic_accumulator_kernel, devices=devices)
add_function_test(TestGenerics, "test_generic_fill", test_generic_fill, devices=devices)
add_function_test(TestGenerics, "test_generic_fill_overloads", test_generic_fill_overloads, devices=devices)
add_function_test(TestGenerics, "test_generic_conditional_setter", test_generic_conditional_setter, devices=devices)
add_function_test(TestGenerics, "test_generic_transform_kernel", test_generic_transform_kernel, devices=devices)
add_function_test(
    TestGenerics, "test_generic_transform_array_kernel", test_generic_transform_array_kernel, devices=devices
)
add_function_test(TestGenerics, "test_generic_type_cast", test_generic_type_cast, devices=devices)
add_function_test(TestGenerics, "test_generic_type_construction", test_generic_type_construction, devices=devices)
add_function_test(TestGenerics, "test_generic_scalar_construction", test_generic_scalar_construction, devices=devices)
add_function_test(TestGenerics, "test_generic_type_as_argument", test_generic_type_as_argument, devices=devices)

foo = Foo()
foo.x = 17.0
foo.y = 25.0
foo.z = 42.0

bar = Bar()
bar.x = wp.vec3(1, 2, 3)
bar.y = wp.vec3(10, 20, 30)
bar.z = wp.vec3(11, 22, 33)

add_kernel_test(
    TestGenerics,
    name="test_generic_struct_kernel",
    kernel=test_generic_struct_kernel,
    dim=1,
    inputs=[foo],
    devices=devices,
)
add_kernel_test(
    TestGenerics,
    name="test_generic_struct_kernel",
    kernel=test_generic_struct_kernel,
    dim=1,
    inputs=[bar],
    devices=devices,
)

add_kernel_test(
    TestGenerics,
    name="test_generic_struct_construction_kernel",
    kernel=test_generic_struct_construction_kernel,
    dim=1,
    inputs=[foo],
    devices=devices,
)
add_kernel_test(
    TestGenerics,
    name="test_generic_struct_construction_kernel",
    kernel=test_generic_struct_construction_kernel,
    dim=1,
    inputs=[bar],
    devices=devices,
)
add_function_test(TestGenerics, "test_type_operator_misspell", test_type_operator_misspell, devices=devices)
add_function_test(TestGenerics, "test_type_attribute_error", test_type_attribute_error, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
