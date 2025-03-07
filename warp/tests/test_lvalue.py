# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from warp.tests.unittest_utils import *


@wp.kernel
def rmw_array_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()

    foos[i] += wp.uint32(1)


def test_rmw_array(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=rmw_array_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.struct
class RmwFoo:
    field: wp.uint32


@wp.kernel
def rmw_array_struct_kernel(foos: wp.array(dtype=RmwFoo)):
    i = wp.tid()
    foos[i].field += wp.uint32(1)


def test_rmw_array_struct(test, device):
    foos = wp.zeros((10,), dtype=RmwFoo, device=device)

    wp.launch(kernel=rmw_array_struct_kernel, dim=(10,), inputs=[foos], device=device)

    expected = RmwFoo()
    expected.field = 1
    for f in foos.list():
        test.assertEqual(f.field, expected.field)


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

    wp.launch(kernel=lookup_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


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
        wp.launch(kernel=grad_kernel, dim=(num,), inputs=[input], outputs=[output], device=device)

    tape.backward(grads={output: ones})

    # test forward results
    for i, f in enumerate(output.list()):
        test.assertEqual(f, data[i] * i + 1)

    # test backward results
    for i, f in enumerate(tape.gradients[input].list()):
        test.assertEqual(f, i)


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

    wp.launch(kernel=lookup2_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.kernel
def unary_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()

    foos[i] = wp.uint32(-1)
    x = -foos[i]
    foos[i] = x


def test_unary(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=unary_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.kernel
def rvalue_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()

    if foos[i] < wp.uint32(1):
        foos[i] = wp.uint32(1)


def test_rvalue(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=rvalue_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


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

    wp.launch(kernel=intermediate_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.kernel
def array_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    foos[i] = wp.uint32(1)


def test_array_assign(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=array_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.func
def increment(arg: wp.uint32):
    return arg + wp.uint32(1)


@wp.kernel
def array_call_kernel(foos: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    foos[i] = increment(foos[i])


def test_array_call_assign(test, device):
    arr = wp.zeros((10,), dtype=wp.uint32, device=device)

    wp.launch(kernel=array_kernel, dim=(10,), inputs=[arr], device=device)

    assert_np_equal(arr.numpy(), np.ones(10))


@wp.struct
class Foo:
    field: wp.uint32


@wp.kernel
def array_struct_kernel(foos: wp.array(dtype=Foo)):
    i = wp.tid()
    foos[i].field = wp.uint32(1)


def test_array_struct_assign(test, device):
    foos = wp.zeros((10,), dtype=Foo, device=device)

    wp.launch(kernel=array_struct_kernel, dim=(10,), inputs=[foos], device=device)

    expected = Foo()
    expected.field = 1

    test.assertEqual(expected.field, 1)

    for f in foos.list():
        test.assertEqual(f.field, 1)


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

    wp.launch(kernel=array_struct_struct_kernel, dim=(10,), inputs=[foos], device=device)

    expected = Baz()
    expected.bar.field = 1

    test.assertEqual(expected.bar.field, 1)

    for f in foos.list():
        test.assertEqual(f.bar.field, 1)


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

    wp.launch(kernel=complex_kernel, dim=(10,), inputs=[foos], device=device)

    expected = F()
    expected.x = 1.0
    expected.y = 2
    expected.s.b = 3.0
    expected.s.a = expected.y
    for f in foos.list():
        test.assertEqual(f.x, expected.x)
        test.assertEqual(f.y, expected.y)
        test.assertEqual(f.s.a, expected.s.a)
        test.assertEqual(f.s.b, expected.s.b)


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

    wp.launch(kernel=swizzle_kernel, dim=(10,), inputs=[foos], device=device)

    expected = Fvec()
    expected.x = wp.vec2f(1.0, 2.0)
    expected.y = 3
    expected.s.b = wp.vec2f(4.0, 5.0)
    expected.s.b.y = 6.0
    expected.s.b.x = expected.x.y
    expected.s.a = expected.y

    for f in foos.list():
        test.assertEqual(f.x, expected.x)
        test.assertEqual(f.y, expected.y)
        test.assertEqual(f.s.a, expected.s.a)
        test.assertEqual(f.s.b, expected.s.b)


@wp.kernel
def slice_kernel(a: wp.array2d(dtype=wp.vec3), b: wp.array2d(dtype=wp.vec3), c: wp.array2d(dtype=wp.vec3)):
    tid = wp.tid()
    c[tid][0] = a[tid][0] + b[tid][0]


def test_slice(test, device):
    a = wp.full((1, 1), value=1.0, dtype=wp.vec3, requires_grad=True, device=device)
    b = wp.full((1, 1), value=1.0, dtype=wp.vec3, requires_grad=True, device=device)
    c = wp.zeros((1, 1), dtype=wp.vec3, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=slice_kernel, dim=1, inputs=[a, b], outputs=[c], device=device)

    c.grad = wp.full((1, 1), value=1.0, dtype=wp.vec3, device=device)
    tape.backward()

    x = a.grad.list()[0]
    y = b.grad.list()[0]

    expected = wp.vec3(1.0)
    test.assertEqual(x, expected)
    test.assertEqual(y, expected)


devices = get_test_devices()


class TestLValue(unittest.TestCase):
    def test_swizzle_error_invalid_attribute(self):
        v = wp.vec3(1, 2, 3)
        with self.assertRaisesRegex(AttributeError, r"'vec3f' object has no attribute 'foo'$"):
            v.foo  # noqa: B018

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
add_function_test(TestLValue, "test_slice", test_slice, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
