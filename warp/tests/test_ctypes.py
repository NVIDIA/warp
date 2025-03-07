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

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def add_vec2(dest: wp.array(dtype=wp.vec2), c: wp.vec2):
    tid = wp.tid()
    dest[tid] = c


@wp.kernel
def transform_vec2(dest_right: wp.array(dtype=wp.vec2), dest_left: wp.array(dtype=wp.vec2), m: wp.mat22, v: wp.vec2):
    tid = wp.tid()

    dest_right[tid] = wp.mul(m, v)
    dest_left[tid] = wp.mul(v, m)


@wp.kernel
def add_vec3(dest: wp.array(dtype=wp.vec3), c: wp.vec3):
    tid = wp.tid()
    dest[tid] = c


@wp.kernel
def transform_vec3(dest_right: wp.array(dtype=wp.vec3), dest_left: wp.array(dtype=wp.vec3), m: wp.mat33, v: wp.vec3):
    tid = wp.tid()

    dest_right[tid] = wp.mul(m, v)
    dest_left[tid] = wp.mul(v, m)


@wp.kernel
def transform_multiply(xforms: wp.array(dtype=wp.transform), a: wp.transform):
    tid = wp.tid()

    xforms[tid] = wp.transform_multiply(xforms[tid], a)


def test_vec2_arg(test, device, n):
    dest = wp.zeros(n=n, dtype=wp.vec2, device=device)
    c = np.array((1.0, 2.0))

    wp.launch(add_vec2, dim=n, inputs=[dest, c], device=device)

    # ensure type can round-trip from Python->GPU->Python
    assert_np_equal(dest.numpy(), np.tile(c, (n, 1)))


def test_vec2_transform(test, device, n):
    dest_right = wp.zeros(n=n, dtype=wp.vec2, device=device)
    dest_left = wp.zeros(n=n, dtype=wp.vec2, device=device)
    c = np.array((1.0, 2.0))
    m = np.array(((3.0, -1.0), (2.5, 4.0)))

    wp.launch(transform_vec2, dim=n, inputs=[dest_right, dest_left, m, c], device=device)
    assert_np_equal(dest_right.numpy(), np.tile(m @ c, (n, 1)))
    assert_np_equal(dest_left.numpy(), np.tile(c @ m, (n, 1)))


def test_vec3_arg(test, device, n):
    dest = wp.zeros(n=n, dtype=wp.vec3, device=device)
    c = np.array((1.0, 2.0, 3.0))

    wp.launch(add_vec3, dim=n, inputs=[dest, c], device=device)
    assert_np_equal(dest.numpy(), np.tile(c, (n, 1)))


def test_vec3_transform(test, device, n):
    dest_right = wp.zeros(n=n, dtype=wp.vec3, device=device)
    dest_left = wp.zeros(n=n, dtype=wp.vec3, device=device)
    c = np.array((1.0, 2.0, 3.0))
    m = np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)))

    wp.launch(transform_vec3, dim=n, inputs=[dest_right, dest_left, m, c], device=device)
    assert_np_equal(dest_right.numpy(), np.tile(m @ c, (n, 1)))
    assert_np_equal(dest_left.numpy(), np.tile(c @ m, (n, 1)))


def test_transform_multiply(test, device, n):
    a = wp.transform((0.0, 1.0, 0.0), wp.quat_identity())

    x = []
    for _i in range(10):
        x.append(wp.transform_identity())

    xforms = wp.array(x, dtype=wp.transform, device=device)
    wp.launch(transform_multiply, dim=n, inputs=[xforms, a], device=device)


transformf = wp.types.transformation(dtype=wp.float32)


@wp.kernel
def test_transformation_constructor():
    a = wp.transformation(wp.vec3(0.0), wp.quat_identity())
    b = transformf(wp.vec3(0.0), wp.quat_identity())
    c = wp.transform_identity(dtype=wp.float64)


spatial_vector = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def test_spatial_vector_constructor():
    a = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))


# construct kernel + test harness for given matrix / vector types
def make_matrix_test(dim, matrix, vector):
    def test_matrix_kernel(
        a: wp.array(dtype=matrix),
        b: wp.array(dtype=matrix),
        c: wp.array(dtype=matrix),
        x: wp.array(dtype=vector),
        result_m: wp.array(dtype=matrix),
        result_i: wp.array(dtype=matrix),
        result_d: wp.array(dtype=float),
        result_x: wp.array(dtype=vector),
    ):
        tid = wp.tid()

        m = a[tid] * b[tid] + c[tid] * 2.0

        result_m[tid] = m
        result_x[tid] = m * x[tid]

        result_d[tid] = wp.determinant(m)

        invm = wp.inverse(m)
        result_i[tid] = m * invm

    # register a custom kernel (no decorator) function
    # this lets us register the same function definition
    # against multiple symbols, with different arg types
    kernel = wp.Kernel(func=test_matrix_kernel, key=f"test_mat{dim}{dim}_kernel")

    def test_matrix(test, device):
        rng = np.random.default_rng(42)

        n = 1024

        a = rng.random(size=(n, dim, dim), dtype=float)
        b = rng.random(size=(n, dim, dim), dtype=float)
        c = rng.random(size=(n, dim, dim), dtype=float)
        x = rng.random(size=(n, dim, 1), dtype=float)

        a_array = wp.array(a, dtype=matrix, device=device)
        b_array = wp.array(b, dtype=matrix, device=device)
        c_array = wp.array(c, dtype=matrix, device=device)

        x_array = wp.array(x, dtype=vector, device=device)

        result_m_array = wp.zeros_like(a_array)
        result_i_array = wp.zeros_like(a_array)
        result_x_array = wp.zeros_like(x_array)
        result_d_array = wp.zeros(n, dtype=float, device=device)

        wp.launch(
            kernel,
            n,
            inputs=[a_array, b_array, c_array, x_array, result_m_array, result_i_array, result_d_array, result_x_array],
            device=device,
        )

        # numpy reference result
        result_m = np.matmul(a, b) + c * 2.0
        result_x = np.matmul(result_m, x)
        result_i = np.array([np.eye(dim)] * n)
        result_d = np.linalg.det(result_m)

        assert_np_equal(result_m_array.numpy(), result_m, tol=1.0e-5)
        assert_np_equal(result_i_array.numpy(), result_i, tol=1.0e-3)
        assert_np_equal(result_d_array.numpy(), result_d, tol=1.0e-3)
        assert_np_equal(result_x_array.numpy(), result_x, tol=1.0e-5)

    return test_matrix


# generate test functions for matrix types
test_mat22 = make_matrix_test(2, wp.mat22, wp.vec2)
test_mat33 = make_matrix_test(3, wp.mat33, wp.vec3)
test_mat44 = make_matrix_test(4, wp.mat44, wp.vec4)


def test_scalar_array(test, device):
    scalar_list = (0.0, 1.0, 2.0)
    scalar_array = wp.array(scalar_list, device=device)

    assert_np_equal(np.array(scalar_list), scalar_array.numpy())


def test_vector_array(test, device):
    vector_list = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]

    vector_array = wp.array(vector_list, dtype=wp.vec3, device=device)

    assert_np_equal(np.array(vector_list), vector_array.numpy())


@wp.kernel
def test_vector_arg_types(v2: wp.vec2, v3: wp.vec3, v4: wp.vec4, m22: wp.mat22, m33: wp.mat33, m44: wp.mat44):
    wp.expect_eq(v2, wp.vec2(1.0, 2.0))
    wp.expect_eq(v3, wp.vec3(1.0, 2.0, 3.0))
    wp.expect_eq(v4, wp.vec4(1.0, 2.0, 3.0, 4.0))

    wp.expect_eq(m22, wp.mat22(1.0, 2.0, 3.0, 4.0))

    wp.expect_eq(m33, wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))

    wp.expect_eq(m44, wp.mat44(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0))


@wp.kernel
def test_scalar_arg_types(
    i8: wp.int8,
    u8: wp.uint8,
    i16: wp.int16,
    u16: wp.uint16,
    i32: wp.int32,
    u32: wp.uint32,
    i64: wp.int64,
    u64: wp.uint64,
    f32: wp.float32,
    f64: wp.float64,
):
    wp.expect_eq(int(i8), -64)
    wp.expect_eq(int(u8), 255)
    wp.expect_eq(int(i16), -64)
    wp.expect_eq(int(u16), 255)
    wp.expect_eq(int(i32), -64)
    wp.expect_eq(int(u32), 255)
    wp.expect_eq(int(i64), -64)
    wp.expect_eq(int(u64), 255)
    wp.expect_eq(int(f32), 3)
    wp.expect_eq(int(f64), 3)
    wp.expect_eq(float(f32), 3.14159)
    wp.expect_eq(float(f64), 3.14159)


@wp.kernel
def test_scalar_array_types_load(
    i8: wp.array(dtype=wp.int8),
    u8: wp.array(dtype=wp.uint8),
    i16: wp.array(dtype=wp.int16),
    u16: wp.array(dtype=wp.uint16),
    i32: wp.array(dtype=wp.int32),
    u32: wp.array(dtype=wp.uint32),
    i64: wp.array(dtype=wp.int64),
    u64: wp.array(dtype=wp.uint64),
    f32: wp.array(dtype=wp.float32),
    f64: wp.array(dtype=wp.float64),
):
    tid = wp.tid()

    wp.expect_eq(int(i8[tid]), tid)
    wp.expect_eq(int(u8[tid]), tid)
    wp.expect_eq(int(i16[tid]), tid)
    wp.expect_eq(int(u16[tid]), tid)
    wp.expect_eq(int(i32[tid]), tid)
    wp.expect_eq(int(u32[tid]), tid)
    wp.expect_eq(int(i64[tid]), tid)
    wp.expect_eq(int(u64[tid]), tid)
    wp.expect_eq(float(f32[tid]), float(tid))
    wp.expect_eq(float(f64[tid]), float(tid))


@wp.kernel
def test_scalar_array_types_store(
    i8: wp.array(dtype=wp.int8),
    u8: wp.array(dtype=wp.uint8),
    i16: wp.array(dtype=wp.int16),
    u16: wp.array(dtype=wp.uint16),
    i32: wp.array(dtype=wp.int32),
    u32: wp.array(dtype=wp.uint32),
    i64: wp.array(dtype=wp.int64),
    u64: wp.array(dtype=wp.uint64),
    f32: wp.array(dtype=wp.float32),
    f64: wp.array(dtype=wp.float64),
):
    tid = wp.tid()

    i8[tid] = wp.int8(tid)
    u8[tid] = wp.uint8(tid)
    i16[tid] = wp.int16(tid)
    u16[tid] = wp.uint16(tid)
    i32[tid] = wp.int32(tid)
    u32[tid] = wp.uint32(tid)
    i64[tid] = wp.int64(tid)
    u64[tid] = wp.uint64(tid)
    f32[tid] = wp.float32(tid)
    f64[tid] = wp.float64(tid)

    # check round-trip
    wp.expect_eq(int(i8[tid]), tid)
    wp.expect_eq(int(u8[tid]), tid)
    wp.expect_eq(int(i16[tid]), tid)
    wp.expect_eq(int(u16[tid]), tid)
    wp.expect_eq(int(i32[tid]), tid)
    wp.expect_eq(int(u32[tid]), tid)
    wp.expect_eq(int(i64[tid]), tid)
    wp.expect_eq(int(u64[tid]), tid)
    wp.expect_eq(float(f32[tid]), float(tid))
    wp.expect_eq(float(f64[tid]), float(tid))


@wp.kernel
def test_type_conversions():
    # below tests auto-generated by the following snippet:

    # scalar_types_all = [*wp.types.scalar_types, int, float]
    # for t in scalar_types_all:
    #     for u in scalar_types_all:

    #         def prefix(t):
    #             if t == int or t == float:
    #                 return t.__name__
    #             else:
    #                 return "wp." + t.__name__

    #         print(f"wp.expect_eq({prefix(t)}(2.0), {prefix(t)}({prefix(u)}(2.0)))")

    wp.expect_eq(wp.int8(2.0), wp.int8(wp.int8(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.uint8(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.int16(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.uint16(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.int32(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.uint32(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.int64(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.uint64(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.float16(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.float32(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(wp.float64(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(int(2.0)))
    wp.expect_eq(wp.int8(2.0), wp.int8(float(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.int8(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.uint8(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.int16(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.uint16(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.int32(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.uint32(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.int64(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.uint64(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.float16(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.float32(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(wp.float64(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(int(2.0)))
    wp.expect_eq(wp.uint8(2.0), wp.uint8(float(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.int8(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.uint8(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.int16(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.uint16(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.int32(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.uint32(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.int64(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.uint64(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.float16(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.float32(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(wp.float64(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(int(2.0)))
    wp.expect_eq(wp.int16(2.0), wp.int16(float(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.int8(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.uint8(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.int16(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.uint16(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.int32(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.uint32(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.int64(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.uint64(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.float16(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.float32(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(wp.float64(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(int(2.0)))
    wp.expect_eq(wp.uint16(2.0), wp.uint16(float(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.int8(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.uint8(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.int16(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.uint16(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.int32(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.uint32(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.int64(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.uint64(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.float16(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.float32(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(wp.float64(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(int(2.0)))
    wp.expect_eq(wp.int32(2.0), wp.int32(float(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.int8(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.uint8(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.int16(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.uint16(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.int32(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.uint32(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.int64(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.uint64(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.float16(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.float32(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(wp.float64(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(int(2.0)))
    wp.expect_eq(wp.uint32(2.0), wp.uint32(float(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.int8(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.uint8(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.int16(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.uint16(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.int32(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.uint32(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.int64(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.uint64(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.float16(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.float32(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(wp.float64(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(int(2.0)))
    wp.expect_eq(wp.int64(2.0), wp.int64(float(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.int8(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.uint8(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.int16(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.uint16(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.int32(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.uint32(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.int64(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.uint64(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.float16(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.float32(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(wp.float64(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(int(2.0)))
    wp.expect_eq(wp.uint64(2.0), wp.uint64(float(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.int8(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.uint8(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.int16(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.uint16(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.int32(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.uint32(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.int64(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.uint64(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.float16(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.float32(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(wp.float64(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(int(2.0)))
    wp.expect_eq(wp.float16(2.0), wp.float16(float(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.int8(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.uint8(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.int16(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.uint16(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.int32(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.uint32(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.int64(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.uint64(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.float16(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.float32(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(wp.float64(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(int(2.0)))
    wp.expect_eq(wp.float32(2.0), wp.float32(float(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.int8(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.uint8(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.int16(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.uint16(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.int32(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.uint32(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.int64(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.uint64(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.float16(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.float32(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(wp.float64(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(int(2.0)))
    wp.expect_eq(wp.float64(2.0), wp.float64(float(2.0)))
    wp.expect_eq(int(2.0), int(wp.int8(2.0)))
    wp.expect_eq(int(2.0), int(wp.uint8(2.0)))
    wp.expect_eq(int(2.0), int(wp.int16(2.0)))
    wp.expect_eq(int(2.0), int(wp.uint16(2.0)))
    wp.expect_eq(int(2.0), int(wp.int32(2.0)))
    wp.expect_eq(int(2.0), int(wp.uint32(2.0)))
    wp.expect_eq(int(2.0), int(wp.int64(2.0)))
    wp.expect_eq(int(2.0), int(wp.uint64(2.0)))
    wp.expect_eq(int(2.0), int(wp.float16(2.0)))
    wp.expect_eq(int(2.0), int(wp.float32(2.0)))
    wp.expect_eq(int(2.0), int(wp.float64(2.0)))
    wp.expect_eq(int(2.0), int(int(2.0)))
    wp.expect_eq(int(2.0), int(float(2.0)))
    wp.expect_eq(float(2.0), float(wp.int8(2.0)))
    wp.expect_eq(float(2.0), float(wp.uint8(2.0)))
    wp.expect_eq(float(2.0), float(wp.int16(2.0)))
    wp.expect_eq(float(2.0), float(wp.uint16(2.0)))
    wp.expect_eq(float(2.0), float(wp.int32(2.0)))
    wp.expect_eq(float(2.0), float(wp.uint32(2.0)))
    wp.expect_eq(float(2.0), float(wp.int64(2.0)))
    wp.expect_eq(float(2.0), float(wp.uint64(2.0)))
    wp.expect_eq(float(2.0), float(wp.float16(2.0)))
    wp.expect_eq(float(2.0), float(wp.float32(2.0)))
    wp.expect_eq(float(2.0), float(wp.float64(2.0)))
    wp.expect_eq(float(2.0), float(int(2.0)))
    wp.expect_eq(float(2.0), float(float(2.0)))


def test_scalar_array_types(test, device, load, store):
    dim = 64

    i8 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.int8), device=device)
    u8 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.uint8), device=device)
    i16 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.int16), device=device)
    u16 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.uint16), device=device)
    i32 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.int32), device=device)
    u32 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.uint32), device=device)
    i64 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.int64), device=device)
    u64 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.uint64), device=device)
    f32 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.float32), device=device)
    f64 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.float64), device=device)

    if load:
        wp.launch(
            test_scalar_array_types_load,
            dim=dim,
            inputs=[i8, u8, i16, u16, i32, u32, i64, u64, f32, f64],
            device=device,
        )

    if store:
        wp.launch(
            test_scalar_array_types_store,
            dim=dim,
            inputs=[i8, u8, i16, u16, i32, u32, i64, u64, f32, f64],
            device=device,
        )


@wp.kernel
def test_transform_matrix():
    r = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5)
    t = wp.vec3(0.25, 0.5, -0.75)
    s = wp.vec3(2.0, 0.5, 0.75)

    m = wp.mat44(t, r, s)

    p = wp.vec3(1.0, 2.0, 3.0)

    r_0 = wp.quat_rotate(r, wp.cw_mul(s, p)) + t
    r_1 = wp.transform_point(m, p)

    r_2 = wp.transform_vector(m, p)

    wp.expect_near(r_0, r_1, 1.0e-4)
    wp.expect_near(r_2, r_0 - t, 1.0e-4)


devices = get_test_devices()


class TestCTypes(unittest.TestCase):
    pass


inputs = [
    wp.vec2(1.0, 2.0),
    wp.vec3(1.0, 2.0, 3.0),
    wp.vec4(1.0, 2.0, 3.0, 4.0),
    wp.mat22(1.0, 2.0, 3.0, 4.0),
    wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
    wp.mat44(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
]

add_function_test(TestCTypes, "test_mat22", test_mat22, devices=devices)
add_function_test(TestCTypes, "test_mat33", test_mat33, devices=devices)
add_function_test(TestCTypes, "test_mat44", test_mat44, devices=devices)
add_kernel_test(
    TestCTypes, name="test_transformation_constructor", kernel=test_transformation_constructor, dim=1, devices=devices
)
add_kernel_test(
    TestCTypes, name="test_spatial_vector_constructor", kernel=test_spatial_vector_constructor, dim=1, devices=devices
)
add_kernel_test(
    TestCTypes,
    name="test_scalar_arg_types",
    kernel=test_scalar_arg_types,
    dim=1,
    inputs=[-64, 255, -64, 255, -64, 255, -64, 255, 3.14159, 3.14159],
    devices=devices,
)
add_kernel_test(
    TestCTypes,
    name="test_scalar_arg_types_explicit",
    kernel=test_scalar_arg_types,
    dim=1,
    inputs=[
        wp.int8(-64),
        wp.uint8(255),
        wp.int16(-64),
        wp.uint16(255),
        wp.int32(-64),
        wp.uint32(255),
        wp.int64(-64),
        wp.uint64(255),
        wp.float32(3.14159),
        wp.float64(3.14159),
    ],
    devices=devices,
)
add_kernel_test(
    TestCTypes, name="test_vector_arg_types", kernel=test_vector_arg_types, dim=1, inputs=inputs, devices=devices
)
add_kernel_test(TestCTypes, name="test_type_convesrions", kernel=test_type_conversions, dim=1, devices=devices)

add_function_test(
    TestCTypes, "test_scalar_array_load", test_scalar_array_types, devices=devices, load=True, store=False
)
add_function_test(
    TestCTypes, "test_scalar_array_store", test_scalar_array_types, devices=devices, load=False, store=True
)
add_function_test(TestCTypes, "test_vec2_arg", test_vec2_arg, devices=devices, n=8)
add_function_test(TestCTypes, "test_vec2_transform", test_vec2_transform, devices=devices, n=8)
add_function_test(TestCTypes, "test_vec3_arg", test_vec3_arg, devices=devices, n=8)
add_function_test(TestCTypes, "test_vec3_transform", test_vec3_transform, devices=devices, n=8)
add_function_test(TestCTypes, "test_transform_multiply", test_transform_multiply, devices=devices, n=8)
add_kernel_test(TestCTypes, name="test_transform_matrix", kernel=test_transform_matrix, dim=1, devices=devices)
add_function_test(TestCTypes, "test_scalar_array", test_scalar_array, devices=devices)
add_function_test(TestCTypes, "test_vector_array", test_vector_array, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
