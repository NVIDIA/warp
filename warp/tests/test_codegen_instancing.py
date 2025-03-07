# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

import warp as wp
import warp.tests.aux_test_name_clash1 as name_clash_module_1
import warp.tests.aux_test_name_clash2 as name_clash_module_2
from warp.tests.unittest_utils import *

# =======================================================================


@wp.kernel
def global_kernel(a: wp.array(dtype=int)):
    a[0] = 17


global_kernel_1 = global_kernel


@wp.kernel
def global_kernel(a: wp.array(dtype=int)):
    a[0] = 42


global_kernel_2 = global_kernel


def test_global_kernel_redefine(test, device):
    """Ensure that referenced kernels remain valid and unique, even when redefined."""

    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=int)

        wp.launch(global_kernel, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 42)

        wp.launch(global_kernel_1, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 17)

        wp.launch(global_kernel_2, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 42)


# =======================================================================


@wp.func
def global_func():
    return 17


global_func_1 = global_func


@wp.func
def global_func():
    return 42


global_func_2 = global_func


@wp.kernel
def global_func_kernel(a: wp.array(dtype=int)):
    a[0] = global_func()
    a[1] = global_func_1()
    a[2] = global_func_2()


def test_global_func_redefine(test, device):
    """Ensure that referenced functions remain valid and unique, even when redefined."""

    with wp.ScopedDevice(device):
        a = wp.zeros(3, dtype=int)
        wp.launch(global_func_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([42, 17, 42]))


# =======================================================================


@wp.struct
class GlobalStruct:
    v: float


GlobalStruct1 = GlobalStruct


@wp.struct
class GlobalStruct:
    v: wp.vec2


GlobalStruct2 = GlobalStruct


@wp.kernel
def global_struct_args_kernel(s0: GlobalStruct, s1: GlobalStruct1, s2: GlobalStruct2, a: wp.array(dtype=float)):
    a[0] = s0.v[0]
    a[1] = s0.v[1]
    a[2] = s1.v
    a[3] = s2.v[0]
    a[4] = s2.v[1]


def test_global_struct_args_redefine(test, device):
    """Ensure that referenced structs remain valid and unique, even when redefined."""
    with wp.ScopedDevice(device):
        s0 = GlobalStruct()
        s1 = GlobalStruct1()
        s2 = GlobalStruct2()
        s0.v = wp.vec2(1.0, 2.0)
        s1.v = 3.0
        s2.v = wp.vec2(4.0, 5.0)

        a = wp.zeros(5, dtype=float)

        wp.launch(global_struct_args_kernel, dim=1, inputs=[s0, s1, s2, a])

        assert_np_equal(a.numpy(), np.array([1, 2, 3, 4, 5], dtype=np.float32))


@wp.kernel
def global_struct_ctor_kernel(a: wp.array(dtype=float)):
    s0 = GlobalStruct()
    s1 = GlobalStruct1()
    s2 = GlobalStruct2()
    s0.v = wp.vec2(1.0, 2.0)
    s1.v = 3.0
    s2.v = wp.vec2(4.0, 5.0)
    a[0] = s0.v[0]
    a[1] = s0.v[1]
    a[2] = s1.v
    a[3] = s2.v[0]
    a[4] = s2.v[1]


def test_global_struct_ctor_redefine(test, device):
    """Ensure that referenced structs remain valid and unique, even when redefined."""
    with wp.ScopedDevice(device):
        a = wp.zeros(5, dtype=float)
        wp.launch(global_struct_ctor_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([1, 2, 3, 4, 5], dtype=np.float32))


# =======================================================================


# "primary" (first) overload
@wp.func
def global_func_po(x: int):
    return x * x


# "secondary" overload
@wp.func
def global_func_po(x: float):
    return x * x


# redefine primary overload
@wp.func
def global_func_po(x: int):
    return x * x * x


@wp.kernel
def global_overload_primary_kernel(a: wp.array(dtype=float)):
    # use primary (int) overload
    a[0] = float(global_func_po(2))
    # use secondary (float) overload
    a[1] = global_func_po(2.0)


def test_global_overload_primary_redefine(test, device):
    """Ensure that redefining a primary overload works and doesn't affect secondary overloads."""
    with wp.ScopedDevice(device):
        a = wp.zeros(2, dtype=float)
        wp.launch(global_overload_primary_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([8, 4], dtype=np.float32))


# =======================================================================


# "primary" (first) overload
@wp.func
def global_func_so(x: int):
    return x * x


# "secondary" overload
@wp.func
def global_func_so(x: float):
    return x * x


# redefine secondary overload
@wp.func
def global_func_so(x: float):
    return x * x * x


@wp.kernel
def global_overload_secondary_kernel(a: wp.array(dtype=float)):
    # use primary (int) overload
    a[0] = float(global_func_so(2))
    # use secondary (float) overload
    a[1] = global_func_so(2.0)


def test_global_overload_secondary_redefine(test, device):
    """Ensure that redefining a secondary overload works."""
    with wp.ScopedDevice(device):
        a = wp.zeros(2, dtype=float)
        wp.launch(global_overload_secondary_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([4, 8], dtype=np.float32))


# =======================================================================


@wp.kernel
def global_generic_kernel(x: Any, a: wp.array(dtype=Any)):
    a[0] = x * x


global_generic_kernel_1 = global_generic_kernel


@wp.kernel
def global_generic_kernel(x: Any, a: wp.array(dtype=Any)):
    a[0] = x * x * x


global_generic_kernel_2 = global_generic_kernel


def test_global_generic_kernel_redefine(test, device):
    """Ensure that referenced generic kernels remain valid and unique, even when redefined."""

    with wp.ScopedDevice(device):
        ai = wp.zeros(1, dtype=int)
        af = wp.zeros(1, dtype=float)

        wp.launch(global_generic_kernel, dim=1, inputs=[2, ai])
        wp.launch(global_generic_kernel, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 8)
        test.assertEqual(af.numpy()[0], 8.0)

        wp.launch(global_generic_kernel_1, dim=1, inputs=[2, ai])
        wp.launch(global_generic_kernel_1, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 4)
        test.assertEqual(af.numpy()[0], 4.0)

        wp.launch(global_generic_kernel_2, dim=1, inputs=[2, ai])
        wp.launch(global_generic_kernel_2, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 8)
        test.assertEqual(af.numpy()[0], 8.0)


# =======================================================================


@wp.func
def global_generic_func(x: Any):
    return x * x


global_generic_func_1 = global_generic_func


@wp.func
def global_generic_func(x: Any):
    return x * x * x


global_generic_func_2 = global_generic_func


@wp.kernel
def global_generic_func_kernel(ai: wp.array(dtype=int), af: wp.array(dtype=float)):
    ai[0] = global_generic_func(2)
    af[0] = global_generic_func(2.0)

    ai[1] = global_generic_func_1(2)
    af[1] = global_generic_func_1(2.0)

    ai[2] = global_generic_func_2(2)
    af[2] = global_generic_func_2(2.0)


def test_global_generic_func_redefine(test, device):
    """Ensure that referenced generic functions remain valid and unique, even when redefined."""

    with wp.ScopedDevice(device):
        ai = wp.zeros(3, dtype=int)
        af = wp.zeros(3, dtype=float)
        wp.launch(global_generic_func_kernel, dim=1, inputs=[ai, af])
        assert_np_equal(ai.numpy(), np.array([8, 4, 8], dtype=np.int32))
        assert_np_equal(af.numpy(), np.array([8, 4, 8], dtype=np.float32))


# =======================================================================


def create_kernel_simple():
    # not a closure
    @wp.kernel
    def k(a: wp.array(dtype=int)):
        a[0] = 17

    return k


simple_kernel_1 = create_kernel_simple()
simple_kernel_2 = create_kernel_simple()


def test_create_kernel_simple(test, device):
    """Test creating multiple identical simple (non-closure) kernels."""
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=int)

        wp.launch(simple_kernel_1, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 17)

        wp.launch(simple_kernel_2, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 17)


# =======================================================================


def create_func_simple():
    # not a closure
    @wp.func
    def f():
        return 17

    return f


simple_func_1 = create_func_simple()
simple_func_2 = create_func_simple()


@wp.kernel
def simple_func_kernel(a: wp.array(dtype=int)):
    a[0] = simple_func_1()
    a[1] = simple_func_2()


def test_create_func_simple(test, device):
    """Test creating multiple identical simple (non-closure) functions."""
    with wp.ScopedDevice(device):
        a = wp.zeros(2, dtype=int)
        wp.launch(simple_func_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([17, 17]))


# =======================================================================


def create_struct_simple():
    @wp.struct
    class S:
        x: int

    return S


SimpleStruct1 = create_struct_simple()
SimpleStruct2 = create_struct_simple()


@wp.kernel
def simple_struct_args_kernel(s1: SimpleStruct1, s2: SimpleStruct2, a: wp.array(dtype=int)):
    a[0] = s1.x
    a[1] = s2.x


def test_create_struct_simple_args(test, device):
    """Test creating multiple identical structs and passing them as arguments."""
    with wp.ScopedDevice(device):
        s1 = SimpleStruct1()
        s2 = SimpleStruct2()
        s1.x = 17
        s2.x = 42
        a = wp.zeros(2, dtype=int)
        wp.launch(simple_struct_args_kernel, dim=1, inputs=[s1, s2, a])
        assert_np_equal(a.numpy(), np.array([17, 42]))


@wp.kernel
def simple_struct_ctor_kernel(a: wp.array(dtype=int)):
    s1 = SimpleStruct1()
    s2 = SimpleStruct2()
    s1.x = 17
    s2.x = 42
    a[0] = s1.x
    a[1] = s2.x


def test_create_struct_simple_ctor(test, device):
    """Test creating multiple identical structs and constructing them in kernels."""
    with wp.ScopedDevice(device):
        a = wp.zeros(2, dtype=int)
        wp.launch(simple_struct_ctor_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([17, 42]))


# =======================================================================


def create_generic_kernel_simple():
    # not a closure
    @wp.kernel
    def k(x: Any, a: wp.array(dtype=Any)):
        a[0] = x * x

    return k


simple_generic_kernel_1 = create_generic_kernel_simple()
simple_generic_kernel_2 = create_generic_kernel_simple()


def test_create_generic_kernel_simple(test, device):
    """Test creating multiple identical simple (non-closure) generic kernels."""
    with wp.ScopedDevice(device):
        ai = wp.zeros(1, dtype=int)
        af = wp.zeros(1, dtype=float)

        wp.launch(simple_generic_kernel_1, dim=1, inputs=[2, ai])
        wp.launch(simple_generic_kernel_1, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 4)
        test.assertEqual(af.numpy()[0], 4.0)

        wp.launch(simple_generic_kernel_2, dim=1, inputs=[2, ai])
        wp.launch(simple_generic_kernel_2, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 4)
        test.assertEqual(af.numpy()[0], 4.0)


# =======================================================================


def create_generic_func_simple():
    # not a closure
    @wp.func
    def f(x: Any):
        return x * x

    return f


simple_generic_func_1 = create_generic_func_simple()
simple_generic_func_2 = create_generic_func_simple()


@wp.kernel
def simple_generic_func_kernel(
    ai: wp.array(dtype=int),
    af: wp.array(dtype=float),
):
    ai[0] = simple_generic_func_1(2)
    af[0] = simple_generic_func_1(2.0)

    ai[1] = simple_generic_func_2(2)
    af[1] = simple_generic_func_2(2.0)


def test_create_generic_func_simple(test, device):
    """Test creating multiple identical simple (non-closure) generic functions."""
    with wp.ScopedDevice(device):
        ai = wp.zeros(2, dtype=int)
        af = wp.zeros(2, dtype=float)
        wp.launch(simple_generic_func_kernel, dim=1, inputs=[ai, af])
        assert_np_equal(ai.numpy(), np.array([4, 4], dtype=np.int32))
        assert_np_equal(af.numpy(), np.array([4, 4], dtype=np.float32))


# =======================================================================


def create_kernel_cond(cond):
    if cond:

        @wp.kernel
        def k(a: wp.array(dtype=int)):
            a[0] = 17
    else:

        @wp.kernel
        def k(a: wp.array(dtype=int)):
            a[0] = 42

    return k


cond_kernel_1 = create_kernel_cond(True)
cond_kernel_2 = create_kernel_cond(False)


def test_create_kernel_cond(test, device):
    """Test conditionally creating different simple (non-closure) kernels."""
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=int)

        wp.launch(cond_kernel_1, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 17)

        wp.launch(cond_kernel_2, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 42)


# =======================================================================


def create_func_cond(cond):
    if cond:

        @wp.func
        def f():
            return 17
    else:

        @wp.func
        def f():
            return 42

    return f


cond_func_1 = create_func_cond(True)
cond_func_2 = create_func_cond(False)


@wp.kernel
def cond_func_kernel(a: wp.array(dtype=int)):
    a[0] = cond_func_1()
    a[1] = cond_func_2()


def test_create_func_cond(test, device):
    """Test conditionally creating different simple (non-closure) functions."""
    with wp.ScopedDevice(device):
        a = wp.zeros(2, dtype=int)
        wp.launch(cond_func_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([17, 42]))


# =======================================================================


def create_struct_cond(cond):
    if cond:

        @wp.struct
        class S:
            v: float
    else:

        @wp.struct
        class S:
            v: wp.vec2

    return S


CondStruct1 = create_struct_cond(True)
CondStruct2 = create_struct_cond(False)


@wp.kernel
def cond_struct_args_kernel(s1: CondStruct1, s2: CondStruct2, a: wp.array(dtype=float)):
    a[0] = s1.v
    a[1] = s2.v[0]
    a[2] = s2.v[1]


def test_create_struct_cond_args(test, device):
    """Test conditionally creating different structs and passing them as arguments."""
    with wp.ScopedDevice(device):
        s1 = CondStruct1()
        s2 = CondStruct2()
        s1.v = 1.0
        s2.v = wp.vec2(2.0, 3.0)
        a = wp.zeros(3, dtype=float)
        wp.launch(cond_struct_args_kernel, dim=1, inputs=[s1, s2, a])
        assert_np_equal(a.numpy(), np.array([1, 2, 3], dtype=np.float32))


@wp.kernel
def cond_struct_ctor_kernel(a: wp.array(dtype=float)):
    s1 = CondStruct1()
    s2 = CondStruct2()
    s1.v = 1.0
    s2.v = wp.vec2(2.0, 3.0)
    a[0] = s1.v
    a[1] = s2.v[0]
    a[2] = s2.v[1]


def test_create_struct_cond_ctor(test, device):
    """Test conditionally creating different structs and passing them as arguments."""
    with wp.ScopedDevice(device):
        a = wp.zeros(3, dtype=float)
        wp.launch(cond_struct_ctor_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([1, 2, 3], dtype=np.float32))


# =======================================================================


def create_generic_kernel_cond(cond):
    if cond:

        @wp.kernel
        def k(x: Any, a: wp.array(dtype=Any)):
            a[0] = x * x
    else:

        @wp.kernel
        def k(x: Any, a: wp.array(dtype=Any)):
            a[0] = x * x * x

    return k


cond_generic_kernel_1 = create_generic_kernel_cond(True)
cond_generic_kernel_2 = create_generic_kernel_cond(False)


def test_create_generic_kernel_cond(test, device):
    """Test creating different simple (non-closure) generic kernels."""
    with wp.ScopedDevice(device):
        ai = wp.zeros(1, dtype=int)
        af = wp.zeros(1, dtype=float)

        wp.launch(cond_generic_kernel_1, dim=1, inputs=[2, ai])
        wp.launch(cond_generic_kernel_1, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 4)
        test.assertEqual(af.numpy()[0], 4.0)

        wp.launch(cond_generic_kernel_2, dim=1, inputs=[2, ai])
        wp.launch(cond_generic_kernel_2, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 8)
        test.assertEqual(af.numpy()[0], 8.0)


# =======================================================================


def create_generic_func_cond(cond):
    if cond:

        @wp.func
        def f(x: Any):
            return x * x
    else:

        @wp.func
        def f(x: Any):
            return x * x * x

    return f


cond_generic_func_1 = create_generic_func_cond(True)
cond_generic_func_2 = create_generic_func_cond(False)


@wp.kernel
def cond_generic_func_kernel(
    ai: wp.array(dtype=int),
    af: wp.array(dtype=float),
):
    ai[0] = cond_generic_func_1(2)
    af[0] = cond_generic_func_1(2.0)

    ai[1] = cond_generic_func_2(2)
    af[1] = cond_generic_func_2(2.0)


def test_create_generic_func_cond(test, device):
    """Test creating different simple (non-closure) generic functions."""
    with wp.ScopedDevice(device):
        ai = wp.zeros(2, dtype=int)
        af = wp.zeros(2, dtype=float)
        wp.launch(cond_generic_func_kernel, dim=1, inputs=[ai, af])
        assert_np_equal(ai.numpy(), np.array([4, 8], dtype=np.int32))
        assert_np_equal(af.numpy(), np.array([4, 8], dtype=np.float32))


# =======================================================================


def create_kernel_closure(value: int):
    # closure
    @wp.kernel
    def k(a: wp.array(dtype=int)):
        a[0] = value

    return k


closure_kernel_1 = create_kernel_closure(17)
closure_kernel_2 = create_kernel_closure(42)


def test_create_kernel_closure(test, device):
    """Test creating kernel closures."""
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=int)

        wp.launch(closure_kernel_1, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 17)

        wp.launch(closure_kernel_2, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 42)


# =======================================================================


def create_func_closure(value: int):
    # closure
    @wp.func
    def f():
        return value

    return f


closure_func_1 = create_func_closure(17)
closure_func_2 = create_func_closure(42)


@wp.kernel
def closure_func_kernel(a: wp.array(dtype=int)):
    a[0] = closure_func_1()
    a[1] = closure_func_2()


def test_create_func_closure(test, device):
    """Test creating function closures."""
    with wp.ScopedDevice(device):
        a = wp.zeros(2, dtype=int)
        wp.launch(closure_func_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([17, 42]))


# =======================================================================


def create_func_closure_overload(value: int):
    @wp.func
    def f():
        return value

    @wp.func
    def f(x: int):
        return value * x

    # return overloaded closure function
    return f


closure_func_overload_1 = create_func_closure_overload(2)
closure_func_overload_2 = create_func_closure_overload(3)


@wp.kernel
def closure_func_overload_kernel(a: wp.array(dtype=int)):
    a[0] = closure_func_overload_1()
    a[1] = closure_func_overload_1(2)
    a[2] = closure_func_overload_2()
    a[3] = closure_func_overload_2(2)


def test_create_func_closure_overload(test, device):
    """Test creating overloaded function closures."""
    with wp.ScopedDevice(device):
        a = wp.zeros(4, dtype=int)
        wp.launch(closure_func_overload_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([2, 4, 3, 6]))


# =======================================================================


def create_func_closure_overload_selfref(value: int):
    @wp.func
    def f():
        return value

    @wp.func
    def f(x: int):
        # reference another overload
        return f() * x

    # return overloaded closure function
    return f


closure_func_overload_selfref_1 = create_func_closure_overload_selfref(2)
closure_func_overload_selfref_2 = create_func_closure_overload_selfref(3)


@wp.kernel
def closure_func_overload_selfref_kernel(a: wp.array(dtype=int)):
    a[0] = closure_func_overload_selfref_1()
    a[1] = closure_func_overload_selfref_1(2)
    a[2] = closure_func_overload_selfref_2()
    a[3] = closure_func_overload_selfref_2(2)


def test_create_func_closure_overload_selfref(test, device):
    """Test creating overloaded function closures with self-referencing overloads."""
    with wp.ScopedDevice(device):
        a = wp.zeros(4, dtype=int)
        wp.launch(closure_func_overload_selfref_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([2, 4, 3, 6]))


# =======================================================================


def create_func_closure_nonoverload(dtype, value):
    @wp.func
    def f(x: dtype):
        return x * value

    return f


# functions created in different scopes should NOT be overloads of each other
# (i.e., creating new functions with the same signature should not replace previous ones)
closure_func_nonoverload_1 = create_func_closure_nonoverload(int, 2)
closure_func_nonoverload_2 = create_func_closure_nonoverload(float, 2.0)
closure_func_nonoverload_3 = create_func_closure_nonoverload(int, 3)
closure_func_nonoverload_4 = create_func_closure_nonoverload(float, 3.0)


@wp.kernel
def closure_func_nonoverload_kernel(
    ai: wp.array(dtype=int),
    af: wp.array(dtype=float),
):
    ai[0] = closure_func_nonoverload_1(2)
    af[0] = closure_func_nonoverload_2(2.0)
    ai[1] = closure_func_nonoverload_3(2)
    af[1] = closure_func_nonoverload_4(2.0)


def test_create_func_closure_nonoverload(test, device):
    """Test creating function closures that are not overloads of each other (overloads are grouped by scope, not globally)."""
    with wp.ScopedDevice(device):
        ai = wp.zeros(2, dtype=int)
        af = wp.zeros(2, dtype=float)
        wp.launch(closure_func_nonoverload_kernel, dim=1, inputs=[ai, af])
        assert_np_equal(ai.numpy(), np.array([4, 6], dtype=np.int32))
        assert_np_equal(af.numpy(), np.array([4, 6], dtype=np.float32))


# =======================================================================


def create_fk_closure(a, b):
    # closure
    @wp.func
    def f():
        return a

    # closure
    @wp.kernel
    def k(a: wp.array(dtype=int)):
        a[0] = f() + b

    return f, k


fk_closure_func_1, fk_closure_kernel_1 = create_fk_closure(10, 7)
fk_closure_func_2, fk_closure_kernel_2 = create_fk_closure(40, 2)


# use generated functions in a new kernel
@wp.kernel
def fk_closure_combine_kernel(a: wp.array(dtype=int)):
    a[0] = fk_closure_func_1() + fk_closure_func_2()


def test_create_fk_closure(test, device):
    """Test creating function and kernel closures together, then reusing the functions in another kernel."""
    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=int)

        wp.launch(fk_closure_kernel_1, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 17)

        wp.launch(fk_closure_kernel_2, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 42)

        wp.launch(fk_closure_combine_kernel, dim=1, inputs=[a])
        test.assertEqual(a.numpy()[0], 50)


# =======================================================================


def create_generic_kernel_closure(value):
    @wp.kernel
    def k(x: Any, a: wp.array(dtype=Any)):
        a[0] = x * type(x)(value)

    return k


generic_closure_kernel_1 = create_generic_kernel_closure(2)
generic_closure_kernel_2 = create_generic_kernel_closure(3)


def test_create_generic_kernel_closure(test, device):
    """Test creating generic closure kernels."""
    with wp.ScopedDevice(device):
        ai = wp.zeros(1, dtype=int)
        af = wp.zeros(1, dtype=float)

        wp.launch(generic_closure_kernel_1, dim=1, inputs=[2, ai])
        wp.launch(generic_closure_kernel_1, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 4)
        test.assertEqual(af.numpy()[0], 4.0)

        wp.launch(generic_closure_kernel_2, dim=1, inputs=[2, ai])
        wp.launch(generic_closure_kernel_2, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 6)
        test.assertEqual(af.numpy()[0], 6.0)


# =======================================================================


def create_generic_kernel_overload_closure(value, dtype):
    @wp.kernel
    def k(x: Any, a: wp.array(dtype=Any)):
        a[0] = x * type(x)(value)

    # return only the overload, not the generic kernel
    return wp.overload(k, [dtype, wp.array(dtype=dtype)])


generic_closure_kernel_overload_i1 = create_generic_kernel_overload_closure(2, int)
generic_closure_kernel_overload_i2 = create_generic_kernel_overload_closure(3, int)
generic_closure_kernel_overload_f1 = create_generic_kernel_overload_closure(2, float)
generic_closure_kernel_overload_f2 = create_generic_kernel_overload_closure(3, float)


def test_create_generic_kernel_overload_closure(test, device):
    """Test creating generic closure kernels, but return only overloads, not the generic kernels themselves."""
    with wp.ScopedDevice(device):
        ai = wp.zeros(1, dtype=int)
        af = wp.zeros(1, dtype=float)

        wp.launch(generic_closure_kernel_overload_i1, dim=1, inputs=[2, ai])
        wp.launch(generic_closure_kernel_overload_f1, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 4)
        test.assertEqual(af.numpy()[0], 4.0)

        wp.launch(generic_closure_kernel_overload_i2, dim=1, inputs=[2, ai])
        wp.launch(generic_closure_kernel_overload_f2, dim=1, inputs=[2.0, af])
        test.assertEqual(ai.numpy()[0], 6)
        test.assertEqual(af.numpy()[0], 6.0)


# =======================================================================


def create_generic_func_closure(value):
    @wp.func
    def f(x: Any):
        return x * type(x)(value)

    return f


generic_closure_func_1 = create_generic_func_closure(2)
generic_closure_func_2 = create_generic_func_closure(3)


@wp.kernel
def closure_generic_func_kernel(
    ai: wp.array(dtype=int),
    af: wp.array(dtype=float),
):
    ai[0] = generic_closure_func_1(2)
    af[0] = generic_closure_func_1(2.0)

    ai[1] = generic_closure_func_2(2)
    af[1] = generic_closure_func_2(2.0)


def test_create_generic_func_closure(test, device):
    """Test creating generic closure functions."""
    with wp.ScopedDevice(device):
        ai = wp.zeros(2, dtype=int)
        af = wp.zeros(2, dtype=float)
        wp.launch(closure_generic_func_kernel, dim=1, inputs=[ai, af])
        assert_np_equal(ai.numpy(), np.array([4, 6], dtype=np.int32))
        assert_np_equal(af.numpy(), np.array([4, 6], dtype=np.float32))


# =======================================================================


def create_generic_func_closure_overload(value):
    @wp.func
    def f(x: Any):
        return x * type(x)(value)

    @wp.func
    def f(x: Any, y: Any):
        return f(x + y)

    # return overloaded generic closure function
    return f


generic_closure_func_overload_1 = create_generic_func_closure_overload(2)
generic_closure_func_overload_2 = create_generic_func_closure_overload(3)


@wp.kernel
def generic_closure_func_overload_kernel(
    ai: wp.array(dtype=int),
    af: wp.array(dtype=float),
):
    ai[0] = generic_closure_func_overload_1(1)  # 1 * 2 = 2
    ai[1] = generic_closure_func_overload_2(1)  # 1 * 3 = 3
    ai[2] = generic_closure_func_overload_1(1, 2)  # (1 + 2) * 2 = 6
    ai[3] = generic_closure_func_overload_2(1, 2)  # (1 + 2) * 3 = 9

    af[0] = generic_closure_func_overload_1(1.0)  # 1 * 2 = 2
    af[1] = generic_closure_func_overload_2(1.0)  # 1 * 3 = 3
    af[2] = generic_closure_func_overload_1(1.0, 2.0)  # (1 + 2) * 2 = 6
    af[3] = generic_closure_func_overload_2(1.0, 2.0)  # (1 + 2) * 3 = 9


def test_create_generic_func_closure_overload(test, device):
    """Test creating overloaded generic function closures."""
    with wp.ScopedDevice(device):
        ai = wp.zeros(4, dtype=int)
        af = wp.zeros(4, dtype=float)
        wp.launch(generic_closure_func_overload_kernel, dim=1, inputs=[ai, af])
        assert_np_equal(ai.numpy(), np.array([2, 3, 6, 9], dtype=np.int32))
        assert_np_equal(af.numpy(), np.array([2, 3, 6, 9], dtype=np.float32))


# =======================================================================


def create_type_closure_scalar(scalar_type):
    @wp.kernel
    def k(input: float, expected: float):
        x = scalar_type(input)
        wp.expect_eq(float(x), expected)

    return k


type_closure_kernel_int = create_type_closure_scalar(int)
type_closure_kernel_float = create_type_closure_scalar(float)
type_closure_kernel_uint8 = create_type_closure_scalar(wp.uint8)


def test_type_closure_scalar(test, device):
    with wp.ScopedDevice(device):
        wp.launch(type_closure_kernel_int, dim=1, inputs=[-1.5, -1.0])
        wp.launch(type_closure_kernel_float, dim=1, inputs=[-1.5, -1.5])

        # FIXME: a problem with type conversions breaks this case
        # wp.launch(type_closure_kernel_uint8, dim=1, inputs=[-1.5, 255.0])


# =======================================================================


def create_type_closure_vector(vec_type):
    @wp.kernel
    def k(expected: float):
        v = vec_type(1.0)
        wp.expect_eq(wp.length_sq(v), expected)

    return k


type_closure_kernel_vec2 = create_type_closure_vector(wp.vec2)
type_closure_kernel_vec3 = create_type_closure_vector(wp.vec3)


def test_type_closure_vector(test, device):
    with wp.ScopedDevice(device):
        wp.launch(type_closure_kernel_vec2, dim=1, inputs=[2.0])
        wp.launch(type_closure_kernel_vec3, dim=1, inputs=[3.0])


# =======================================================================


@wp.struct
class ClosureStruct1:
    v: float


@wp.struct
class ClosureStruct2:
    v: wp.vec2


@wp.func
def closure_struct_func(s: ClosureStruct1):
    return 17.0


@wp.func
def closure_struct_func(s: ClosureStruct2):
    return 42.0


def create_type_closure_struct(struct_type):
    @wp.kernel
    def k(expected: float):
        s = struct_type()
        result = closure_struct_func(s)
        wp.expect_eq(result, expected)

    return k


type_closure_kernel_struct1 = create_type_closure_struct(ClosureStruct1)
type_closure_kernel_struct2 = create_type_closure_struct(ClosureStruct2)


def test_type_closure_struct(test, device):
    with wp.ScopedDevice(device):
        wp.launch(type_closure_kernel_struct1, dim=1, inputs=[17.0])
        wp.launch(type_closure_kernel_struct2, dim=1, inputs=[42.0])


# =======================================================================


@wp.kernel
def name_clash_func_kernel(a: wp.array(dtype=int)):
    a[0] = name_clash_module_1.same_func()
    a[1] = name_clash_module_2.same_func()
    a[2] = name_clash_module_1.different_func()
    a[3] = name_clash_module_2.different_func()


def test_name_clash_func(test, device):
    """Test using identically named functions from different modules"""
    with wp.ScopedDevice(device):
        a = wp.zeros(4, dtype=int)
        wp.launch(name_clash_func_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([99, 99, 17, 42]))


# =======================================================================


@wp.kernel
def name_clash_structs_args_kernel(
    s1: name_clash_module_1.SameStruct,
    s2: name_clash_module_2.SameStruct,
    d1: name_clash_module_1.DifferentStruct,
    d2: name_clash_module_2.DifferentStruct,
    a: wp.array(dtype=float),
):
    a[0] = s1.x
    a[1] = s2.x
    a[2] = d1.v
    a[3] = d2.v[0]
    a[4] = d2.v[1]


def test_name_clash_struct_args(test, device):
    with wp.ScopedDevice(device):
        s1 = name_clash_module_1.SameStruct()
        s2 = name_clash_module_2.SameStruct()
        d1 = name_clash_module_1.DifferentStruct()
        d2 = name_clash_module_2.DifferentStruct()
        s1.x = 1.0
        s2.x = 2.0
        d1.v = 3.0
        d2.v = wp.vec2(4.0, 5.0)
        a = wp.zeros(5, dtype=float)
        wp.launch(name_clash_structs_args_kernel, dim=1, inputs=[s1, s2, d1, d2, a])
        assert_np_equal(a.numpy(), np.array([1, 2, 3, 4, 5], dtype=np.float32))


# =======================================================================


@wp.kernel
def name_clash_structs_ctor_kernel(
    a: wp.array(dtype=float),
):
    s1 = name_clash_module_1.SameStruct()
    s2 = name_clash_module_2.SameStruct()
    d1 = name_clash_module_1.DifferentStruct()
    d2 = name_clash_module_2.DifferentStruct()

    s1.x = 1.0
    s2.x = 2.0
    d1.v = 3.0
    d2.v = wp.vec2(4.0, 5.0)

    a[0] = s1.x
    a[1] = s2.x
    a[2] = d1.v
    a[3] = d2.v[0]
    a[4] = d2.v[1]


def test_name_clash_struct_ctor(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(5, dtype=float)
        wp.launch(name_clash_structs_ctor_kernel, dim=1, inputs=[a])
        assert_np_equal(a.numpy(), np.array([1, 2, 3, 4, 5], dtype=np.float32))


# =======================================================================


def test_create_kernel_loop(test, device):
    """
    Test creating a kernel in a loop.  The kernel is always the same,
    so the module hash doesn't change and the module shouldn't be reloaded.
    This test ensures that the kernel hooks are found for new duplicate kernels.
    """

    with wp.ScopedDevice(device):
        for _ in range(5):

            @wp.kernel
            def k():
                pass

            wp.launch(k, dim=1)
            wp.synchronize_device()


# =======================================================================


def test_module_mark_modified(test, device):
    """Test that Module.mark_modified() forces module rehashing and reloading."""

    with wp.ScopedDevice(device):

        @wp.kernel
        def k(expected: int):
            wp.expect_eq(C, expected)

        C = 17
        wp.launch(k, dim=1, inputs=[17])
        wp.synchronize_device()

        # redefine constant and force rehashing on next launch
        C = 42
        k.module.mark_modified()

        wp.launch(k, dim=1, inputs=[42])
        wp.synchronize_device()


# =======================================================================


def test_garbage_collection(test, device):
    """Test that dynamically generated kernels without user references are not retained in the module."""

    # use a helper module with a known kernel count
    import warp.tests.aux_test_instancing_gc as gc_test_module

    with wp.ScopedDevice(device):
        a = wp.zeros(1, dtype=int)

        for i in range(10):
            # create a unique kernel on each iteration
            k = gc_test_module.create_kernel_closure(i)

            # import gc
            # gc.collect()

            # since we don't keep references to the previous kernels,
            # they should be garbage-collected and not appear in the module
            k.module.load(device=device)
            test.assertEqual(len(k.module.live_kernels), 1)

            # test the kernel
            wp.launch(k, dim=1, inputs=[a])
            test.assertEqual(a.numpy()[0], i)


# =======================================================================


class TestCodeGenInstancing(unittest.TestCase):
    pass


devices = get_test_devices()

# global redefinitions with retained references
add_function_test(
    TestCodeGenInstancing, func=test_global_kernel_redefine, name="test_global_kernel_redefine", devices=devices
)
add_function_test(
    TestCodeGenInstancing, func=test_global_func_redefine, name="test_global_func_redefine", devices=devices
)
add_function_test(
    TestCodeGenInstancing,
    func=test_global_struct_args_redefine,
    name="test_global_struct_args_redefine",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_global_struct_ctor_redefine,
    name="test_global_struct_ctor_redefine",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_global_overload_primary_redefine,
    name="test_global_overload_primary_redefine",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_global_overload_secondary_redefine,
    name="test_global_overload_secondary_redefine",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_global_generic_kernel_redefine,
    name="test_global_generic_kernel_redefine",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_global_generic_func_redefine,
    name="test_global_generic_func_redefine",
    devices=devices,
)

# create identical simple kernels, functions, and structs
add_function_test(
    TestCodeGenInstancing, func=test_create_kernel_simple, name="test_create_kernel_simple", devices=devices
)
add_function_test(TestCodeGenInstancing, func=test_create_func_simple, name="test_create_func_simple", devices=devices)
add_function_test(
    TestCodeGenInstancing, func=test_create_struct_simple_args, name="test_create_struct_simple_args", devices=devices
)
add_function_test(
    TestCodeGenInstancing, func=test_create_struct_simple_ctor, name="test_create_struct_simple_ctor", devices=devices
)
add_function_test(
    TestCodeGenInstancing,
    func=test_create_generic_kernel_simple,
    name="test_create_generic_kernel_simple",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing, func=test_create_generic_func_simple, name="test_create_generic_func_simple", devices=devices
)

# create different simple kernels, functions, and structs
add_function_test(TestCodeGenInstancing, func=test_create_kernel_cond, name="test_create_kernel_cond", devices=devices)
add_function_test(TestCodeGenInstancing, func=test_create_func_cond, name="test_create_func_cond", devices=devices)
add_function_test(
    TestCodeGenInstancing, func=test_create_struct_cond_args, name="test_create_struct_cond_args", devices=devices
)
add_function_test(
    TestCodeGenInstancing, func=test_create_struct_cond_ctor, name="test_create_struct_cond_ctor", devices=devices
)
add_function_test(
    TestCodeGenInstancing, func=test_create_generic_kernel_cond, name="test_create_generic_kernel_cond", devices=devices
)
add_function_test(
    TestCodeGenInstancing, func=test_create_generic_func_cond, name="test_create_generic_func_cond", devices=devices
)

# closure kernels and functions
add_function_test(
    TestCodeGenInstancing, func=test_create_kernel_closure, name="test_create_kernel_closure", devices=devices
)
add_function_test(
    TestCodeGenInstancing, func=test_create_func_closure, name="test_create_func_closure", devices=devices
)
add_function_test(
    TestCodeGenInstancing,
    func=test_create_func_closure_overload,
    name="test_create_func_closure_overload",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_create_func_closure_overload_selfref,
    name="test_create_func_closure_overload_selfref",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_create_func_closure_nonoverload,
    name="test_create_func_closure_nonoverload",
    devices=devices,
)
add_function_test(TestCodeGenInstancing, func=test_create_fk_closure, name="test_create_fk_closure", devices=devices)
add_function_test(
    TestCodeGenInstancing,
    func=test_create_generic_kernel_closure,
    name="test_create_generic_kernel_closure",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_create_generic_kernel_overload_closure,
    name="test_create_generic_kernel_overload_closure",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_create_generic_func_closure,
    name="test_create_generic_func_closure",
    devices=devices,
)
add_function_test(
    TestCodeGenInstancing,
    func=test_create_generic_func_closure_overload,
    name="test_create_generic_func_closure_overload",
    devices=devices,
)

# type closures
add_function_test(
    TestCodeGenInstancing, func=test_type_closure_scalar, name="test_type_closure_scalar", devices=devices
)
add_function_test(
    TestCodeGenInstancing, func=test_type_closure_vector, name="test_type_closure_vector", devices=devices
)
add_function_test(
    TestCodeGenInstancing, func=test_type_closure_struct, name="test_type_closure_struct", devices=devices
)

# test name clashes between modules
add_function_test(TestCodeGenInstancing, func=test_name_clash_func, name="test_name_clash_func", devices=devices)
add_function_test(
    TestCodeGenInstancing, func=test_name_clash_struct_args, name="test_name_clash_struct_args", devices=devices
)
add_function_test(
    TestCodeGenInstancing, func=test_name_clash_struct_ctor, name="test_name_clash_struct_ctor", devices=devices
)

# miscellaneous tests
add_function_test(TestCodeGenInstancing, func=test_create_kernel_loop, name="test_create_kernel_loop", devices=devices)
add_function_test(
    TestCodeGenInstancing, func=test_module_mark_modified, name="test_module_mark_modified", devices=devices
)
add_function_test(TestCodeGenInstancing, func=test_garbage_collection, name="test_garbage_collection", devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
