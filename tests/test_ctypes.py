# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp

import unittest
import test_base

wp.init()

@wp.kernel
def add_vec2(dest: wp.array(dtype=wp.vec2),
             c: wp.vec2):

    tid = wp.tid()
    wp.store(dest, tid, c)


@wp.kernel
def transform_vec2(dest: wp.array(dtype=wp.vec2),
    m: wp.mat22,
    v: wp.vec2):

    tid = wp.tid()

    p = wp.mul(m, v)
    wp.store(dest, tid, p)


@wp.kernel
def add_vec3(dest: wp.array(dtype=wp.vec3),
             c: wp.vec3):

    tid = wp.tid()
    wp.store(dest, tid, c)

@wp.kernel
def transform_vec3(dest: wp.array(dtype=wp.vec3),
    m: wp.mat33,
    v: wp.vec3):

    tid = wp.tid()

    p = wp.mul(m, v)
    wp.store(dest, tid, p)


@wp.kernel
def transform_multiply(xforms: wp.array(dtype=wp.transform),
    a: wp.transform):

    tid = wp.tid()

    xforms[tid] = wp.transform_multiply(xforms[tid], a)


def test_vec2_arg(test, device, n):
    
    dest = wp.zeros(n=n, dtype=wp.vec2, device=device)
    c = np.array((1.0, 2.0))

    wp.launch(add_vec2, dim=n, inputs=[dest, c], device=device)

    # ensure type can round-trip from Python->GPU->Python
    test.assertTrue(np.array_equal(dest.numpy(), np.tile(c, (n, 1))))

def test_vec2_transform(test, device, n):

    dest = wp.zeros(n=n, dtype=wp.vec2, device=device)
    c = np.array((1.0, 2.0))
    m = np.array(((3.0, -1.0),
                  (2.5, 4.0)))

    wp.launch(transform_vec2, dim=n, inputs=[dest, m, c], device=device)
    test.assertTrue(np.array_equal(dest.numpy(), np.tile(m@c, (n, 1))))

def test_vec3_arg(test, device, n):
        
    dest = wp.zeros(n=n, dtype=wp.vec3, device=device)
    c = np.array((1.0, 2.0, 3.0))

    wp.launch(add_vec3, dim=n, inputs=[dest, c], device=device)
    test.assertTrue(np.array_equal(dest.numpy(), np.tile(c, (n, 1))))

def test_vec3_transform(test, device, n):

    dest = wp.zeros(n=n, dtype=wp.vec3, device=device)
    c = np.array((1.0, 2.0, 3.0))
    m = np.array(((1.0, 2.0, 3.0),
                  (4.0, 5.0, 6.0),
                  (7.0, 8.0, 9.0)))

    wp.launch(transform_vec3, dim=n, inputs=[dest, m, c], device=device)
    test.assertTrue(np.array_equal(dest.numpy(), np.tile(m@c, (n, 1))))

def test_transform_multiply(test, device, n):

    a = wp.transform((0.0, 1.0, 0.0), wp.quat_identity())

    x = []
    for i in range(10):
        x.append(wp.transform_identity())

    xforms = wp.array(x, dtype=wp.transform, device=device)
    wp.launch(transform_multiply, dim=n, inputs=[xforms, a], device=device)


def test_scalar_array(test, device):

    scalar_list = (0.0, 1.0, 2.0)
    scalar_array = wp.array(scalar_list, device=device)

    test.assert_np_equal(np.array(scalar_list), scalar_array.numpy())

def test_vector_array(test, device):

    vector_list = [ (0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0),
                    (2.0, 2.0, 2.0) ]

    vector_array = wp.array(vector_list, dtype=wp.vec3, device=device)

    test.assert_np_equal(np.array(vector_list), vector_array.numpy())
    


@wp.kernel
def test_vector_arg_types(v2: wp.vec2,
                          v3: wp.vec3,
                          v4: wp.vec4,
                          m22: wp.mat22,
                          m33: wp.mat33,
                          m44: wp.mat44):

    wp.expect_eq(v2, wp.vec2(1.0, 2.0))
    wp.expect_eq(v3, wp.vec3(1.0, 2.0, 3.0))
    wp.expect_eq(v4, wp.vec4(1.0, 2.0, 3.0, 4.0))
    
    wp.expect_eq(m22, wp.mat22(1.0, 2.0, 
                               3.0, 4.0))
    
    wp.expect_eq(m33, wp.mat33(1.0, 2.0, 3.0,
                               4.0, 5.0, 6.0,
                               7.0, 8.0, 9.0))

    wp.expect_eq(m44, wp.mat44(1.0, 2.0, 3.0, 4.0,
                               5.0, 6.0, 7.0, 8.0,
                               9.0, 10.0, 11.0, 12.0,
                               13.0, 14.0, 15.0, 16.0))


@wp.kernel
def test_scalar_arg_types(i8: wp.int8,
                          u8: wp.uint8,
                          i16: wp.int16,
                          u16: wp.uint16,
                          i32: wp.int32,                  
                          u32: wp.uint32,
                          i64: wp.int64,
                          u64: wp.uint64,
                          f32: wp.float32,
                          f64: wp.float64):
                  
    wp.expect_eq(int(i8), -64)
    wp.expect_eq(int(u8),  255)
    wp.expect_eq(int(i16), -64)
    wp.expect_eq(int(u16),  255)
    wp.expect_eq(int(i32), -64)
    wp.expect_eq(int(u32), 255)
    wp.expect_eq(int(i64), -64)
    wp.expect_eq(int(u64), 255)
    wp.expect_eq(int(f32), 3)
    wp.expect_eq(int(f64), 3)
    wp.expect_eq(float(f32), 3.14159)
    wp.expect_eq(float(f64), 3.14159)


@wp.kernel
def test_scalar_array_types_load(i8: wp.array(dtype=wp.int8),
                                 u8: wp.array(dtype=wp.uint8),
                                 i16: wp.array(dtype=wp.int16),
                                 u16: wp.array(dtype=wp.uint16),
                                 i32: wp.array(dtype=wp.int32),
                                 u32: wp.array(dtype=wp.uint32),
                                 i64: wp.array(dtype=wp.int64),
                                 u64: wp.array(dtype=wp.uint64),
                                 f32: wp.array(dtype=wp.float32),
                                 f64: wp.array(dtype=wp.float64)):
                  
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
def test_scalar_array_types_store(i8: wp.array(dtype=wp.int8),
                                  u8: wp.array(dtype=wp.uint8),
                                  i16: wp.array(dtype=wp.int16),
                                  u16: wp.array(dtype=wp.uint16),                                  
                                  i32: wp.array(dtype=wp.int32),
                                  u32: wp.array(dtype=wp.uint32),
                                  i64: wp.array(dtype=wp.int64),
                                  u64: wp.array(dtype=wp.uint64),
                                  f32: wp.array(dtype=wp.float32),
                                  f64: wp.array(dtype=wp.float64)):
                  
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


def test_scalar_array_types(test, device, load, store):

    dim = 64

    i8 = wp.array(np.linspace(0, dim, dim,  endpoint=False, dtype=np.int8), device=device)
    u8 = wp.array(np.linspace(0, dim, dim,  endpoint=False, dtype=np.uint8), device=device)
    i16 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.int16), device=device)
    u16 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.uint16), device=device)
    i32 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.int32), device=device)
    u32 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.uint32), device=device)
    i64 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.int64), device=device)
    u64 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.uint64), device=device)
    f32 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.float32), device=device)
    f64 = wp.array(np.linspace(0, dim, dim, endpoint=False, dtype=np.float64), device=device)

    if load:
        wp.launch(test_scalar_array_types_load, dim=dim, inputs=[i8, u8, i16, u16, i32, u32, i64, u64, f32, f64], device=device)

    if store:
        wp.launch(test_scalar_array_types_store, dim=dim, inputs=[i8, u8, i16, u16, i32, u32, i64, u64, f32, f64], device=device)


devices = wp.get_devices()

class TestCTypes(test_base.TestBase):
    pass


inputs = [wp.vec2(1.0, 2.0),
          wp.vec3(1.0, 2.0, 3.0),
          wp.vec4(1.0, 2.0, 3.0, 4.0),
          wp.mat22(1.0, 2.0,
                   3.0, 4.0),
          wp.mat33(1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0),
          wp.mat44(1.0, 2.0, 3.0, 4.0,
                   5.0, 6.0, 7.0, 8.0,
                   9.0, 10.0, 11.0, 12.0,
                   13.0, 14.0, 15.0, 16.0)]


TestCTypes.add_kernel_test(name="test_scalar_arg_types", kernel=test_scalar_arg_types, dim=1, inputs=[-64, 255, -64, 255, -64, 255, -64, 255, 3.14159, 3.14159], devices=devices)
TestCTypes.add_kernel_test(name="test_vector_arg_types", kernel=test_vector_arg_types, dim=1, inputs=inputs, devices=devices)
TestCTypes.add_function_test("test_scalar_array_load", test_scalar_array_types, devices=devices, load=True, store=False)
TestCTypes.add_function_test("test_scalar_array_store", test_scalar_array_types, devices=devices, load=False, store=True)
TestCTypes.add_function_test("test_vec2_arg", test_vec2_arg, devices=devices, n=8)
TestCTypes.add_function_test("test_vec2_transform", test_vec2_transform, devices=devices, n=8)
TestCTypes.add_function_test("test_vec3_arg", test_vec3_arg, devices=devices, n=8)
TestCTypes.add_function_test("test_vec3_transform", test_vec3_transform, devices=devices, n=8)
TestCTypes.add_function_test("test_transform_multiply", test_transform_multiply, devices=devices, n=8)
TestCTypes.add_function_test("test_scalar_array", test_scalar_array, devices=devices)
TestCTypes.add_function_test("test_vector_array", test_vector_array, devices=devices)


if __name__ == '__main__':
    unittest.main(verbosity=2)