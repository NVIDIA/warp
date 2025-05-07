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
from typing import Any, Tuple

import warp as wp
from warp.tests.unittest_utils import *


@wp.struct
class BasicsStruct:
    origin: wp.vec3
    scale: float


@wp.kernel
def test_basics():
    tid = wp.tid()
    s = BasicsStruct(wp.vec3(1.1, 2.2, 3.3), 1.23)

    t = (1, 2.2, wp.vec3(1.1, 2.2, 3.3), wp.mat22(1.1, 2.2, 3.3, 4.4), s, tid)
    wp.expect_eq(len(t), 6)
    wp.expect_eq(wp.static(len(t)), 6)
    wp.expect_eq(t[0], 1)
    wp.expect_eq(t[1], 2.2)
    wp.expect_eq(t[2], wp.vec3(1.1, 2.2, 3.3))
    wp.expect_eq(t[3], wp.mat22(1.1, 2.2, 3.3, 4.4))
    wp.expect_eq(t[4].origin, wp.vec3(1.1, 2.2, 3.3))
    wp.expect_eq(t[4].scale, 1.23)
    wp.expect_eq(t[5], wp.tid())

    t0, t1, t2, t3, t4, t5 = t
    wp.expect_eq(t0, 1)
    wp.expect_eq(t1, 2.2)
    wp.expect_eq(t2, wp.vec3(1.1, 2.2, 3.3))
    wp.expect_eq(t3, wp.mat22(1.1, 2.2, 3.3, 4.4))
    wp.expect_eq(t4.origin, wp.vec3(1.1, 2.2, 3.3))
    wp.expect_eq(t4.scale, 1.23)
    wp.expect_eq(t5, wp.tid())


@wp.kernel
def test_builtin_with_multiple_return():
    expected_axis = wp.vec3(0.26726124, 0.53452247, 0.80178368)
    expected_angle = 1.50408018
    q = wp.quat(1.0, 2.0, 3.0, 4.0)

    t = wp.quat_to_axis_angle(q)
    wp.expect_eq(len(t), 2)
    wp.expect_eq(wp.static(len(t)), 2)

    axis_1 = t[0]
    angle_1 = t[1]
    wp.expect_near(axis_1[0], expected_axis[0])
    wp.expect_near(axis_1[1], expected_axis[1])
    wp.expect_near(axis_1[2], expected_axis[2])
    wp.expect_near(angle_1, expected_angle)

    axis_2, angle_2 = t
    wp.expect_near(axis_2[0], expected_axis[0])
    wp.expect_near(axis_2[1], expected_axis[1])
    wp.expect_near(axis_2[2], expected_axis[2])
    wp.expect_near(angle_2, expected_angle)

    axis_3, angle_3 = wp.quat_to_axis_angle(q)
    wp.expect_near(axis_3[0], expected_axis[0])
    wp.expect_near(axis_3[1], expected_axis[1])
    wp.expect_near(axis_3[2], expected_axis[2])
    wp.expect_near(angle_3, expected_angle)


@wp.func
def user_func_with_multiple_return(x: int, y: float) -> Tuple[int, float]:
    return (x * 123, y * 1.23)


@wp.kernel
def test_user_func_with_multiple_return():
    t = user_func_with_multiple_return(4, wp.pow(2.0, 3.0))
    wp.expect_eq(len(t), 2)
    wp.expect_eq(wp.static(len(t)), 2)

    x_1 = t[0]
    y_1 = t[1]
    wp.expect_eq(x_1, 492)
    wp.expect_near(y_1, 9.84)

    x_2, y_2 = t
    wp.expect_eq(x_2, 492)
    wp.expect_near(y_2, 9.84)

    x_3, y_3 = user_func_with_multiple_return(4, wp.pow(2.0, 3.0))
    wp.expect_eq(x_3, 492)
    wp.expect_near(y_3, 9.84)


@wp.func
def user_func_with_tuple_arg(values: Tuple[wp.vec3, float]) -> float:
    wp.expect_eq(len(values), 2)
    wp.expect_eq(wp.static(len(values)), 2)
    return wp.length(values[0]) * values[1]


@wp.kernel
def test_user_func_with_tuple_arg():
    t = (wp.vec3(1.0, 2.0, 3.0), wp.pow(2.0, 4.0))
    wp.expect_eq(len(t), 2)
    wp.expect_eq(wp.static(len(t)), 2)

    x_1 = user_func_with_tuple_arg(t)
    wp.expect_near(x_1, 59.86652)

    x_2 = user_func_with_tuple_arg((t[0], t[1]))
    wp.expect_near(x_2, 59.86652)

    x_3 = user_func_with_tuple_arg((wp.vec3(1.0, 2.0, 3.0), wp.pow(2.0, 4.0)))
    wp.expect_near(x_3, 59.86652)


@wp.func
def loop_user_func(values: Tuple[int, int, int]):
    out = wp.int32(0)
    for i in range(wp.static(len(values))):
        out += values[i]

    for i in range(len(values)):
        out += values[i] * 2

    return out


@wp.kernel
def test_loop():
    t = (1, 2, 3)
    res = loop_user_func(t)
    wp.expect_eq(res, 18)


@wp.func
def loop_variadic_any_user_func(values: Any):
    out = wp.int32(0)
    for i in range(wp.static(len(values))):
        out += values[i]

    for i in range(len(values)):
        out += values[i] * 2

    return out


@wp.kernel
def test_loop_variadic_any():
    t1 = (1,)
    res = loop_variadic_any_user_func(t1)
    wp.expect_eq(res, 3)

    t2 = (2, 3)
    res = loop_variadic_any_user_func(t2)
    wp.expect_eq(res, 15)

    t3 = (3, 4, 5)
    res = loop_variadic_any_user_func(t3)
    wp.expect_eq(res, 36)

    t4 = (4, 5, 6, 7)
    res = loop_variadic_any_user_func(t4)
    wp.expect_eq(res, 66)


@wp.func
def loop_variadic_ellipsis_user_func(values: Tuple[int, ...]):
    out = wp.int32(0)
    for i in range(wp.static(len(values))):
        out += values[i]

    return out


@wp.kernel
def test_loop_variadic_ellipsis():
    t1 = (1,)
    res = loop_variadic_ellipsis_user_func(t1)
    wp.expect_eq(res, 1)

    t2 = (2, 3)
    res = loop_variadic_ellipsis_user_func(t2)
    wp.expect_eq(res, 5)

    t3 = (3, 4, 5)
    res = loop_variadic_ellipsis_user_func(t3)
    wp.expect_eq(res, 12)

    t4 = (4, 5, 6, 7)
    res = loop_variadic_ellipsis_user_func(t4)
    wp.expect_eq(res, 22)


devices = get_test_devices()


class TestTuple(unittest.TestCase):
    pass


add_kernel_test(TestTuple, name="test_basics", kernel=test_basics, dim=3, devices=devices)
add_kernel_test(
    TestTuple,
    name="test_builtin_with_multiple_return",
    kernel=test_builtin_with_multiple_return,
    dim=1,
    devices=devices,
)
add_kernel_test(
    TestTuple,
    name="test_user_func_with_multiple_return",
    kernel=test_user_func_with_multiple_return,
    dim=1,
    devices=devices,
)
add_kernel_test(
    TestTuple,
    name="test_user_func_with_tuple_arg",
    kernel=test_user_func_with_tuple_arg,
    dim=1,
    devices=devices,
)
add_kernel_test(
    TestTuple,
    name="test_loop",
    kernel=test_loop,
    dim=1,
    devices=devices,
)
add_kernel_test(
    TestTuple,
    name="test_loop_variadic_any",
    kernel=test_loop_variadic_any,
    dim=1,
    devices=devices,
)
add_kernel_test(
    TestTuple,
    name="test_loop_variadic_ellipsis",
    kernel=test_loop_variadic_ellipsis,
    dim=1,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
