# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import warp as wp
from warp.tests.unittest_utils import *

wp.init()


@wp.kernel
def rmw_array_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()

    foos[i] += wp.uint32(1)


def test_rmw_array(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(
        kernel=rmw_array_kernel,
        dim=(10,),
        inputs=[arr],
        device=device,
    )
    wp.synchronize()

    for f in arr.list():
        if f != 1:
            raise AssertionError(f"Unexpected result, got: {f} expected: {1}")


@wp.struct
class RmwFoo:
    field: wp.uint32


@wp.kernel
def rmw_array_struct_kernel(foos: wp.array(dtype=RmwFoo)):
    i = wp.tid()
    foos[i].field += wp.uint32(1)


def test_rmw_array_struct(test, device):
    foos = wp.zeros((10,), dtype=RmwFoo, device=device)

    wp.launch(
        kernel=rmw_array_struct_kernel,
        dim=(10,),
        inputs=[foos],
        device=device,
    )
    wp.synchronize()

    expected = RmwFoo()
    expected.field = 1
    for f in foos.list():
        if f.field != expected.field:
            raise AssertionError(f"Unexpected result, got: {f} expected: {expected}")


@wp.func
def lookup(foos: wp.array(dtype=wp.uint32), index: int):
    return foos[index]


@wp.kernel
def lookup_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()

    x = lookup(foos, i)
    foos[i] = x + wp.uint32(1)


def test_lookup(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(
        kernel=lookup_kernel,
        dim=(10,),
        inputs=[arr],
        device=device,
    )
    wp.synchronize()

    for f in arr.list():
        if f != 1:
            raise AssertionError(f"Unexpected result, got: {f} expected: {1}")


@wp.func
def lookup3(foos: wp.array(dtype=wp.float32), index: int):
    return foos[index]


@wp.kernel
def grad_kernel(foos: wp.array(dtype=wp.float32), bars: wp.array(dtype=wp.float32)):
    i = wp.tid()

    x = lookup3(foos, i)
    bars[i] = x * wp.float32(i) + 1.0


def test_grad(test, device):
    num = 10
    data = np.linspace(20, 20 + num, num, endpoint=False, dtype=np.float32)
    input = wp.array(data, device=device, requires_grad=True)
    output = wp.zeros(num, dtype=wp.float32, device=device)

    ones = wp.array(np.ones(len(output)), dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=grad_kernel,
            dim=(num,),
            inputs=[input],
            outputs=[output],
            device=device,
        )

    tape.backward(grads={output: ones})

    wp.synchronize()

    # test forward results
    for i, f in enumerate(output.list()):
        expected = data[i] * i + 1
        if f != expected:
            raise AssertionError(f"Unexpected result, got: {f} expected: {expected}")

    # test backward results
    for i, f in enumerate(tape.gradients[input].list()):
        expected = i
        if f != expected:
            raise AssertionError(f"Unexpected result, got: {f} expected: {expected}")


@wp.func
def lookup2(foos: wp.array(dtype=wp.uint32), index: int):
    if index % 2 == 0:
        x = foos[index]
        x = wp.uint32(0)
        return x
    else:
        return foos[index]


@wp.kernel
def lookup2_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()

    x = lookup2(foos, i)
    foos[i] = x + wp.uint32(1)


def test_lookup2(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(
        kernel=lookup2_kernel,
        dim=(10,),
        inputs=[arr],
        device=device,
    )
    wp.synchronize()

    for f in arr.list():
        if f != 1:
            raise AssertionError(f"Unexpected result, got: {f} expected: {1}")


@wp.kernel
def unary_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()

    foos[i] = wp.uint32(-1)
    x = -foos[i]
    foos[i] = x


def test_unary(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(
        kernel=unary_kernel,
        dim=(10,),
        inputs=[arr],
        device=device,
    )
    wp.synchronize()

    for f in arr.list():
        if f != 1:
            raise AssertionError(f"Unexpected result, got: {f} expected: {1}")


@wp.kernel
def rvalue_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()

    if foos[i] < wp.uint32(1):
        foos[i] = wp.uint32(1)


def test_rvalue(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(
        kernel=rvalue_kernel,
        dim=(10,),
        inputs=[arr],
        device=device,
    )
    wp.synchronize()

    for f in arr.list():
        if f != 1:
            raise AssertionError(f"Unexpected result, got: {f} expected: {1}")


# Tests, among other things, that assigning a reference to a new variable does
# not create a reference
@wp.kernel
def intermediate_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()

    x = foos[i]
    x = x + wp.uint32(1)
    foos[i] = x


def test_intermediate(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(
        kernel=intermediate_kernel,
        dim=(10,),
        inputs=[arr],
        device=device,
    )
    wp.synchronize()

    for f in arr.list():
        if f != 1:
            raise AssertionError(f"Unexpected result, got: {f} expected: {1}")


@wp.kernel
def array_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    foos[i] = wp.uint32(1)


def test_array_assign(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(
        kernel=array_kernel,
        dim=(10,),
        inputs=[arr],
        device=device,
    )
    wp.synchronize()

    for f in arr.list():
        if f != 1:
            raise AssertionError(f"Unexpected result, got: {f} expected: {1}")


@wp.func
def increment(arg: wp.uint32):
    return arg + wp.uint32(1)


@wp.kernel
def array_call_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    foos[i] = increment(foos[i])


def test_array_call_assign(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(
        kernel=array_kernel,
        dim=(10,),
        inputs=[arr],
        device=device,
    )
    wp.synchronize()

    for f in arr.list():
        if f != 1:
            raise AssertionError(f"Unexpected result, got: {f} expected: {1}")


@wp.struct
class Foo:
    field: wp.uint32


@wp.kernel
def array_struct_kernel(foos: wp.array(dtype=Foo)):
    i = wp.tid()
    foos[i].field = wp.uint32(1)


def test_array_struct_assign(test, device):
    foos = wp.zeros((10,), dtype=Foo, device=device)

    wp.launch(
        kernel=array_struct_kernel,
        dim=(10,),
        inputs=[foos],
        device=device,
    )
    wp.synchronize()

    expected = Foo()
    expected.field = 1
    for f in foos.list():
        if f.field != expected.field:
            raise AssertionError(f"Unexpected result, got: {f} expected: {expected}")


@wp.struct
class Bar:
    field: wp.uint32


@wp.struct
class Baz:
    bar: Bar


@wp.kernel
def array_struct_struct_kernel(foos: wp.array(dtype=Baz)):
    i = wp.tid()
    foos[i].bar.field = wp.uint32(1)


def test_array_struct_struct_assign(test, device):
    foos = wp.zeros((10,), dtype=Baz, device=device)

    wp.launch(
        kernel=array_struct_struct_kernel,
        dim=(10,),
        inputs=[foos],
        device=device,
    )
    wp.synchronize()

    expected = Baz()
    expected.bar.field = 1
    for f in foos.list():
        if f.bar.field != expected.bar.field:
            raise AssertionError(f"Unexpected result, got: {f} expected: {expected}")


@wp.struct
class S:
    a: wp.uint32
    b: wp.float32


@wp.struct
class F:
    x: wp.float32
    s: S
    y: wp.int32


@wp.kernel
def complex_kernel(foos: wp.array(dtype=F)):
    i = wp.tid()
    foos[i].x += wp.float32(1.0)
    foos[i].y = wp.int32(2)
    foos[i].s.b += wp.float32(3.0)
    foos[i].s.a = wp.uint32(foos[i].y)


def test_complex(test, device):
    foos = wp.zeros((10,), dtype=F, device=device)

    wp.launch(
        kernel=complex_kernel,
        dim=(10,),
        inputs=[foos],
        device=device,
    )
    wp.synchronize()

    expected = F()
    expected.x = 1.0
    expected.y = 2
    expected.s.b = 3.0
    expected.s.a = expected.y
    for f in foos.list():
        if f.x != expected.x or f.y != expected.y or f.s.a != expected.s.a or f.s.b != expected.s.b:
            raise AssertionError(f"Unexpected result, got: {f} expected: {expected}")


@wp.struct
class Svec:
    a: wp.uint32
    b: wp.vec2f


@wp.struct
class Fvec:
    x: wp.vec2f
    s: Svec
    y: wp.int32


@wp.kernel
def swizzle_kernel(foos: wp.array(dtype=Fvec)):
    i = wp.tid()

    foos[i].x += wp.vec2f(1.0, 2.0)
    foos[i].y = wp.int32(3)
    foos[i].s.b = wp.vec2f(4.0, 5.0)
    foos[i].s.b.y = wp.float32(6.0)
    foos[i].s.b.x = foos[i].x.y
    foos[i].s.a = wp.uint32(foos[i].y)


def test_swizzle(test, device):
    foos = wp.zeros((10,), dtype=Fvec, device=device)

    wp.launch(
        kernel=swizzle_kernel,
        dim=(10,),
        inputs=[foos],
        device=device,
    )
    wp.synchronize()

    expected = Fvec()
    expected.x = wp.vec2f(1.0, 2.0)
    expected.y = 3
    expected.s.b = wp.vec2f(4.0, 5.0)
    expected.s.b.y = 6.0
    expected.s.b.x = expected.x.y
    expected.s.a = expected.y
    for f in foos.list():
        if f.x != expected.x or f.y != expected.y or f.s.a != expected.s.a or f.s.b != expected.s.b:
            raise AssertionError(f"Unexpected result, got: {f} expected: {expected}")


devices = get_test_devices()


class TestLValue(unittest.TestCase):
    def test_swizzle_error_invalid_attribute(self):
        v = wp.vec3(1, 2, 3)
        with self.assertRaisesRegex(
            AttributeError,
            r"'vec3f' object has no attribute 'foo'$",
        ):
            v.foo

        try:
            v.bar = 123
        except AttributeError:
            self.fail()


add_function_test(TestLValue, "test_rmw_array", test_rmw_array, devices=devices)
add_function_test(TestLValue, "test_rmw_array_struct", test_rmw_array_struct, devices=devices)
add_function_test(TestLValue, "test_lookup", test_lookup, devices=devices)
add_function_test(TestLValue, "test_lookup2", test_lookup2, devices=devices)
add_function_test(TestLValue, "test_grad", test_grad, devices=devices)
add_function_test(TestLValue, "test_unary", test_unary, devices=devices)
add_function_test(TestLValue, "test_rvalue", test_rvalue, devices=devices)
add_function_test(TestLValue, "test_intermediate", test_intermediate, devices=devices)
add_function_test(TestLValue, "test_array_assign", test_array_assign, devices=devices)
add_function_test(TestLValue, "test_array_struct_assign", test_array_struct_assign, devices=devices)
add_function_test(TestLValue, "test_array_struct_struct_assign", test_array_struct_struct_assign, devices=devices)
add_function_test(TestLValue, "test_complex", test_complex, devices=devices)
add_function_test(TestLValue, "test_swizzle", test_swizzle, devices=devices)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
