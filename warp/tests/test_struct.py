# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *

import unittest

wp.init()


@wp.struct
class Model:
    dt: float
    gravity: wp.vec3
    m: wp.array(dtype=float)


@wp.struct
class State:
    x: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)


@wp.kernel
def kernel_step(state_in: State, state_out: State, model: Model):
    i = wp.tid()

    state_out.v[i] = state_in.v[i] + model.gravity / model.m[i] * model.dt
    state_out.x[i] = state_in.x[i] + state_out.v[i] * model.dt


@wp.kernel
def kernel_step_with_copy(state_in: State, state_out: State, model: Model):
    i = wp.tid()

    model_rescaled = Model(1.0, model.gravity / model.m[i] * model.dt, model.m)

    state_out_copy = State(state_out.x, state_out.v)
    state_out_copy.v[i] = state_in.v[i] + model_rescaled.gravity
    state_out_copy.x[i] = state_in.x[i] + state_out_copy.v[i] * model.dt


def test_step(test, device):
    dim = 5

    dt = 0.01
    gravity = np.array([0, 0, -9.81])

    m = np.ones(dim)

    m_model = wp.array(m, dtype=float, device=device)

    model = Model()
    model.m = m_model
    model.dt = dt
    model.gravity = wp.vec3(0, 0, -9.81)

    np.random.seed(0)
    x = np.random.normal(size=(dim, 3))
    v = np.random.normal(size=(dim, 3))

    x_expected = x + (v + gravity / m[:, None] * dt) * dt

    x_in = wp.array(x, dtype=wp.vec3, device=device)
    v_in = wp.array(v, dtype=wp.vec3, device=device)

    state_in = State()
    state_in.x = x_in
    state_in.v = v_in

    state_out = State()
    state_out.x = wp.empty_like(x_in)
    state_out.v = wp.empty_like(v_in)

    for step_kernel in [kernel_step, kernel_step_with_copy]:
        with CheckOutput(test):
            wp.launch(step_kernel, dim=dim, inputs=[state_in, state_out, model], device=device)

        assert_np_equal(state_out.x.numpy(), x_expected, tol=1e-6)


@wp.kernel
def kernel_loss(x: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(loss, 0, x[i][0] * x[i][0] + x[i][1] * x[i][1] + x[i][2] * x[i][2])


def test_step_grad(test, device):
    dim = 5

    dt = 0.01
    gravity = np.array([0, 0, -9.81])

    np.random.seed(0)
    m = np.random.rand(dim) + 0.1

    m_model = wp.array(m, dtype=float, device=device, requires_grad=True)

    model = Model()
    model.m = m_model
    model.dt = dt
    model.gravity = wp.vec3(0, 0, -9.81)

    x = np.random.normal(size=(dim, 3))
    v = np.random.normal(size=(dim, 3))

    x_in = wp.array(x, dtype=wp.vec3, device=device, requires_grad=True)
    v_in = wp.array(v, dtype=wp.vec3, device=device, requires_grad=True)

    state_in = State()
    state_in.x = x_in
    state_in.v = v_in

    state_out = State()
    state_out.x = wp.empty_like(x_in, requires_grad=True)
    state_out.v = wp.empty_like(v_in, requires_grad=True)

    loss = wp.empty(1, dtype=float, device=device, requires_grad=True)

    for step_kernel in [kernel_step, kernel_step_with_copy]:
        tape = wp.Tape()

        with tape:
            wp.launch(step_kernel, dim=dim, inputs=[state_in, state_out, model], device=device)
            wp.launch(kernel_loss, dim=dim, inputs=[state_out.x, loss], device=device)

        tape.backward(loss)

        dl_dx = 2 * state_out.x.numpy()
        dl_dv = dl_dx * dt

        dv_dm = -gravity * dt / m[:, None] ** 2
        dl_dm = (dl_dv * dv_dm).sum(-1)

        assert_np_equal(state_out.x.grad.numpy(), dl_dx, tol=1e-6)
        assert_np_equal(state_in.x.grad.numpy(), dl_dx, tol=1e-6)
        assert_np_equal(state_out.v.grad.numpy(), dl_dv, tol=1e-6)
        assert_np_equal(state_in.v.grad.numpy(), dl_dv, tol=1e-6)
        assert_np_equal(model.m.grad.numpy(), dl_dm, tol=1e-6)

        tape.zero()

        assert state_out.x.grad.numpy().sum() == 0.0
        assert state_in.x.grad.numpy().sum() == 0.0
        assert state_out.v.grad.numpy().sum() == 0.0
        assert state_in.v.grad.numpy().sum() == 0.0
        assert model.m.grad.numpy().sum() == 0.0


@wp.struct
class Empty:
    pass


@wp.kernel
def test_empty(input: Empty):
    tid = wp.tid()


@wp.struct
class Uninitialized:
    data: wp.array(dtype=int)


@wp.kernel
def test_uninitialized(input: Uninitialized):
    tid = wp.tid()


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


@wp.kernel
def kernel_nested_struct(foo: Foo):
    tid = wp.tid()
    foo.bar.baz.data[tid] = (
        foo.bar.baz.data[tid] + foo.x + int(foo.bar.y * 100.0) + int(wp.length_sq(foo.bar.baz.z)) + tid * 2
    )


def test_nested_struct(test, device):
    dim = 3

    foo = Foo()
    foo.bar = Bar()
    foo.bar.baz = Baz()
    foo.bar.baz.data = wp.zeros(dim, dtype=int, device=device)
    foo.bar.baz.z = wp.vec3(1, 2, 3)
    foo.bar.y = 1.23
    foo.x = 123

    wp.launch(kernel_nested_struct, dim=dim, inputs=[foo], device=device)

    assert_array_equal(
        foo.bar.baz.data,
        wp.array((260, 262, 264), dtype=int, device=device),
    )


@wp.kernel
def test_struct_instantiate(data: wp.array(dtype=int)):
    baz = Baz(data, wp.vec3(0.0, 0.0, 26.0))
    bar = Bar(baz, 25.0)
    foo = Foo(bar, 24)

    wp.expect_eq(foo.x, 24)
    wp.expect_eq(foo.bar.y, 25.0)
    wp.expect_eq(foo.bar.baz.z[2], 26.0)
    wp.expect_eq(foo.bar.baz.data[0], 1)


@wp.struct
class MathThings:
    v1: wp.vec3
    v2: wp.vec3
    v3: wp.vec3
    m1: wp.mat22
    m2: wp.mat22
    m3: wp.mat22
    m4: wp.mat22
    m5: wp.mat22
    m6: wp.mat22


@wp.kernel
def check_math_conversions(s: MathThings):
    wp.expect_eq(s.v1, wp.vec3(1.0, 2.0, 3.0))
    wp.expect_eq(s.v2, wp.vec3(10.0, 20.0, 30.0))
    wp.expect_eq(s.v3, wp.vec3(100.0, 200.0, 300.0))
    wp.expect_eq(s.m1, wp.mat22(1.0, 2.0, 3.0, 4.0))
    wp.expect_eq(s.m2, wp.mat22(10.0, 20.0, 30.0, 40.0))
    wp.expect_eq(s.m3, wp.mat22(100.0, 200.0, 300.0, 400.0))
    wp.expect_eq(s.m4, wp.mat22(1.0, 2.0, 3.0, 4.0))
    wp.expect_eq(s.m5, wp.mat22(10.0, 20.0, 30.0, 40.0))
    wp.expect_eq(s.m6, wp.mat22(100.0, 200.0, 300.0, 400.0))


def test_struct_math_conversions(test, device):
    s = MathThings()

    # test assigning various containers to vector and matrix attributes
    s.v1 = (1, 2, 3)
    s.v2 = [10, 20, 30]
    s.v3 = np.array([100, 200, 300])
    # 2d containers for matrices
    s.m1 = ((1, 2), (3, 4))
    s.m2 = [[10, 20], [30, 40]]
    s.m3 = np.array([[100, 200], [300, 400]])
    # 1d containers for matrices
    s.m4 = (1, 2, 3, 4)
    s.m5 = [10, 20, 30, 40]
    s.m6 = np.array([100, 200, 300, 400])

    wp.launch(check_math_conversions, dim=1, inputs=[s])


@wp.struct
class TestData:
    value: wp.int32


@wp.func
def GetTestData(value: wp.int32):
    return TestData(value * 2)


@wp.kernel
def test_return_struct(data: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    data[tid] = GetTestData(tid).value

    wp.expect_eq(data[tid], tid * 2)


@wp.struct
class ReturnStruct:
    a: int
    b: int


@wp.func
def test_return_func():
    a = ReturnStruct(1, 2)
    return a


@wp.kernel
def test_return():
    t = test_return_func()
    wp.expect_eq(t.a, 1)
    wp.expect_eq(t.b, 2)


@wp.struct
class DefaultAttribNested:
    f: float


@wp.struct
class DefaultAttribStruct:
    i: int
    d: wp.float64
    v: wp.vec3
    m: wp.mat22
    a: wp.array(dtype=wp.int32)
    s: DefaultAttribNested


@wp.func
def check_default_attributes_func(data: DefaultAttribStruct):
    wp.expect_eq(data.i, wp.int32(0))
    wp.expect_eq(data.d, wp.float64(0))
    wp.expect_eq(data.v, wp.vec3(0.0, 0.0, 0.0))
    wp.expect_eq(data.m, wp.mat22(0.0, 0.0, 0.0, 0.0))
    wp.expect_eq(data.a.shape[0], 0)
    wp.expect_eq(data.s.f, wp.float32(0.0))


@wp.kernel
def check_default_attributes_kernel(data: DefaultAttribStruct):
    check_default_attributes_func(data)


# check structs default initialized in Python correctly
def test_struct_default_attributes_python(test, device):
    s = DefaultAttribStruct()

    wp.launch(check_default_attributes_kernel, dim=1, inputs=[s])


# check structs default initialized in kernels correctly
@wp.kernel
def test_struct_default_attributes_kernel():
    s = DefaultAttribStruct()

    check_default_attributes_func(s)


@wp.struct
class MutableStruct:
    param1: int
    param2: float


@wp.kernel
def test_struct_mutate_attributes_kernel():
    t = MutableStruct()
    t.param1 = 1
    t.param2 = 1.1

    wp.expect_eq(t.param1, 1)
    wp.expect_eq(t.param2, 1.1)


@wp.struct
class InnerStruct:
    i: int


@wp.struct
class ArrayStruct:
    array: wp.array(dtype=InnerStruct)


@wp.kernel
def struct2_reader(test: ArrayStruct):
    k = wp.tid()
    wp.expect_eq(k + 1, test.array[k].i)


def test_nested_array_struct(test, device):
    var1 = InnerStruct()
    var1.i = 1

    var2 = InnerStruct()
    var2.i = 2

    struct = ArrayStruct()
    struct.array = wp.array([var1, var2], dtype=InnerStruct)

    wp.launch(struct2_reader, dim=2, inputs=[struct])


def register(parent):
    devices = get_test_devices()

    class TestStruct(parent):
        pass

    add_function_test(TestStruct, "test_step", test_step, devices=devices)
    add_function_test(TestStruct, "test_step_grad", test_step_grad, devices=devices)
    add_kernel_test(TestStruct, kernel=test_empty, name="test_empty", dim=1, inputs=[Empty()], devices=devices)
    add_kernel_test(
        TestStruct,
        kernel=test_uninitialized,
        name="test_uninitialized",
        dim=1,
        inputs=[Uninitialized()],
        devices=devices,
    )
    add_kernel_test(TestStruct, kernel=test_return, name="test_return", dim=1, inputs=[], devices=devices)
    add_function_test(TestStruct, "test_nested_struct", test_nested_struct, devices=devices)
    add_function_test(TestStruct, "test_nested_array_struct", test_nested_array_struct, devices=devices)
    add_function_test(TestStruct, "test_struct_math_conversions", test_struct_math_conversions, devices=devices)
    add_function_test(
        TestStruct, "test_struct_default_attributes_python", test_struct_default_attributes_python, devices=devices
    )
    add_kernel_test(
        TestStruct,
        name="test_struct_default_attributes",
        kernel=test_struct_default_attributes_kernel,
        dim=1,
        inputs=[],
        devices=devices,
    )

    add_kernel_test(
        TestStruct,
        name="test_struct_mutate_attributes",
        kernel=test_struct_mutate_attributes_kernel,
        dim=1,
        inputs=[],
        devices=devices,
    )

    for device in devices:
        add_kernel_test(
            TestStruct,
            kernel=test_struct_instantiate,
            name="test_struct_instantiate",
            dim=1,
            inputs=[wp.array([1], dtype=int, device=device)],
            devices=[device],
        )
        add_kernel_test(
            TestStruct,
            kernel=test_return_struct,
            name="test_return_struct",
            dim=1,
            inputs=[wp.zeros(10, dtype=int, device=device)],
            devices=[device],
        )

    return TestStruct


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
