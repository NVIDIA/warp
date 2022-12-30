# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *

wp.init()

np_signed_int_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.byte,
]

np_unsigned_int_types = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ubyte,
]

np_int_types = np_signed_int_types + np_unsigned_int_types

np_float_types = [
    np.float16,
    np.float32,
    np.float64
]

np_scalar_types = np_int_types + np_float_types

def randvals(shape,dtype):
    if dtype in np_float_types:
        return np.random.randn(*shape).astype(dtype)
    elif dtype in [np.int8,np.uint8,np.byte,np.ubyte]:
        return np.random.randint(1,3,size=shape,dtype=dtype)
    return np.random.randint(1,5,size=shape,dtype=dtype)

def getkernel(func,suffix=""):
    
    module = wp.get_module(func.__name__ + "_" + suffix)
    return wp.Kernel(func=func, key=func.__name__ + "_" + suffix, module=module)

def test_arrays(test, device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 1.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)
    
    v2_np = randvals((10,2),dtype)
    v3_np = randvals((10,3),dtype)
    v4_np = randvals((10,4),dtype)
    v5_np = randvals((10,5),dtype)
    
    v2 = wp.array(v2_np, dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(v3_np, dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(v4_np, dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(v5_np, dtype=vec5, requires_grad=True, device=device)

    assert_np_equal(v2.numpy(), v2_np, tol=1.e-6)
    assert_np_equal(v3.numpy(), v3_np, tol=1.e-6)
    assert_np_equal(v4.numpy(), v4_np, tol=1.e-6)
    assert_np_equal(v5.numpy(), v5_np, tol=1.e-6)

    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    
    v2 = wp.array(v2_np, dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(v3_np, dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(v4_np, dtype=vec4, requires_grad=True, device=device)
    
    assert_np_equal(v2.numpy(), v2_np, tol=1.e-6)
    assert_np_equal(v3.numpy(), v3_np, tol=1.e-6)
    assert_np_equal(v4.numpy(), v4_np, tol=1.e-6)


def test_constructors(test, device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)
    
    def check_scalar_constructor(
        input: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = vec2(wptype(2.0) * input[0])
        v3result = vec3(wptype(3.0) * input[0])
        v4result = vec4(wptype(4.0) * input[0])
        v5result = vec5(wptype(5.0) * input[0])

        v2[0] = v2result
        v3[0] = v3result
        v4[0] = v4result
        v5[0] = v5result

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]

    input = wp.array(randvals([1],dtype), requires_grad=True, device=device)
    kernel = getkernel(check_scalar_constructor,suffix=dtype.__name__)
    v2 = wp.zeros(1, dtype=vec2, device=device)
    v3 = wp.zeros(1, dtype=vec3, device=device)
    v4 = wp.zeros(1, dtype=vec4, device=device)
    v5 = wp.zeros(1, dtype=vec5, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[ input ], outputs=[v2,v3,v4,v5,v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)

    if dtype in np_float_types:
        for l in [v20,v21]:
            tape.backward(loss=l)
            test.assertEqual(tape.gradients[input].numpy()[0],2.0)
            tape.zero()

        for l in [v30,v31,v32]:
            tape.backward(loss=l)
            test.assertEqual(tape.gradients[input].numpy()[0],3.0)
            tape.zero()

        for l in [v40,v41,v42,v43]:
            tape.backward(loss=l)
            test.assertEqual(tape.gradients[input].numpy()[0],4.0)
            tape.zero()

        for l in [v50,v51,v52,v53,v54]:
            tape.backward(loss=l)
            test.assertEqual(tape.gradients[input].numpy()[0],5.0)
            tape.zero()

    val = input.numpy()[0]
    assert_np_equal(v2.numpy()[0], 2.0 * np.array([val,val]), tol=1.e-6)
    assert_np_equal(v3.numpy()[0], 3.0 * np.array([val,val,val]), tol=1.e-6)
    assert_np_equal(v4.numpy()[0], 4.0 * np.array([val,val,val,val]), tol=1.e-6)
    assert_np_equal(v5.numpy()[0], 5.0 * np.array([val,val,val,val,val]), tol=1.e-6)
    
    assert_np_equal(v20.numpy()[0], 2.0 * val, tol=1.e-6)
    assert_np_equal(v21.numpy()[0], 2.0 * val, tol=1.e-6)
    assert_np_equal(v30.numpy()[0], 3.0 * val, tol=1.e-6)
    assert_np_equal(v31.numpy()[0], 3.0 * val, tol=1.e-6)
    assert_np_equal(v32.numpy()[0], 3.0 * val, tol=1.e-6)
    assert_np_equal(v40.numpy()[0], 4.0 * val, tol=1.e-6)
    assert_np_equal(v41.numpy()[0], 4.0 * val, tol=1.e-6)
    assert_np_equal(v42.numpy()[0], 4.0 * val, tol=1.e-6)
    assert_np_equal(v43.numpy()[0], 4.0 * val, tol=1.e-6)
    assert_np_equal(v50.numpy()[0], 5.0 * val, tol=1.e-6)
    assert_np_equal(v51.numpy()[0], 5.0 * val, tol=1.e-6)
    assert_np_equal(v52.numpy()[0], 5.0 * val, tol=1.e-6)
    assert_np_equal(v53.numpy()[0], 5.0 * val, tol=1.e-6)
    assert_np_equal(v54.numpy()[0], 5.0 * val, tol=1.e-6)

    
    
    def check_vector_constructors(
        input: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = vec2(wptype(2.0) * input[0],wptype(3.0) * input[1])
        v3result = vec3(wptype(4.0) * input[2],wptype(5.0) * input[3],wptype(6.0) * input[4])
        v4result = vec4(wptype(7.0) * input[5],wptype(8.0) * input[6],wptype(9.0) * input[7],wptype(10.0) * input[8])
        v5result = vec5(wptype(11.0) * input[9],wptype(12.0) * input[10],wptype(13.0) * input[11],wptype(14.0) * input[12],wptype(15.0) * input[13])

        v2[0] = v2result
        v3[0] = v3result
        v4[0] = v4result
        v5[0] = v5result

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]

    input = wp.array(randvals([14],dtype), requires_grad=True, device=device)
    kernel = getkernel(check_vector_constructors,suffix=dtype.__name__)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[ input ], outputs=[v2,v3,v4,v5,v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)
    
    if dtype in np_float_types:
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54]):
            tape.backward(loss=l)
            grad = tape.gradients[input].numpy()
            expected_grad = np.zeros_like(grad)
            expected_grad[i] = i+2
            assert_np_equal(grad,expected_grad, tol=tol)
            tape.zero()

    assert_np_equal(v2.numpy()[0,0], 2.0 * input.numpy()[0], tol=tol)
    assert_np_equal(v2.numpy()[0,1], 3.0 * input.numpy()[1], tol=tol)
    assert_np_equal(v3.numpy()[0,0], 4.0 * input.numpy()[2], tol=tol)
    assert_np_equal(v3.numpy()[0,1], 5.0 * input.numpy()[3], tol=tol)
    assert_np_equal(v3.numpy()[0,2], 6.0 * input.numpy()[4], tol=tol)
    assert_np_equal(v4.numpy()[0,0], 7.0 * input.numpy()[5], tol=tol)
    assert_np_equal(v4.numpy()[0,1], 8.0 * input.numpy()[6], tol=tol)
    assert_np_equal(v4.numpy()[0,2], 9.0 * input.numpy()[7], tol=tol)
    assert_np_equal(v4.numpy()[0,3], 10.0 * input.numpy()[8], tol=tol)
    assert_np_equal(v5.numpy()[0,0], 11.0 * input.numpy()[9], tol=tol)
    assert_np_equal(v5.numpy()[0,1], 12.0 * input.numpy()[10], tol=tol)
    assert_np_equal(v5.numpy()[0,2], 13.0 * input.numpy()[11], tol=tol)
    assert_np_equal(v5.numpy()[0,3], 14.0 * input.numpy()[12], tol=tol)
    assert_np_equal(v5.numpy()[0,4], 15.0 * input.numpy()[13], tol=tol)
    
    assert_np_equal(v20.numpy()[0], 2.0 * input.numpy()[0], tol=tol)
    assert_np_equal(v21.numpy()[0], 3.0 * input.numpy()[1], tol=tol)
    assert_np_equal(v30.numpy()[0], 4.0 * input.numpy()[2], tol=tol)
    assert_np_equal(v31.numpy()[0], 5.0 * input.numpy()[3], tol=tol)
    assert_np_equal(v32.numpy()[0], 6.0 * input.numpy()[4], tol=tol)
    assert_np_equal(v40.numpy()[0], 7.0 * input.numpy()[5], tol=tol)
    assert_np_equal(v41.numpy()[0], 8.0 * input.numpy()[6], tol=tol)
    assert_np_equal(v42.numpy()[0], 9.0 * input.numpy()[7], tol=tol)
    assert_np_equal(v43.numpy()[0], 10.0 * input.numpy()[8], tol=tol)
    assert_np_equal(v50.numpy()[0], 11.0 * input.numpy()[9], tol=tol)
    assert_np_equal(v51.numpy()[0], 12.0 * input.numpy()[10], tol=tol)
    assert_np_equal(v52.numpy()[0], 13.0 * input.numpy()[11], tol=tol)
    assert_np_equal(v53.numpy()[0], 14.0 * input.numpy()[12], tol=tol)
    assert_np_equal(v54.numpy()[0], 15.0 * input.numpy()[13], tol=tol)


def test_indexing(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)
    
    def check_indexing(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),

        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),

        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v20[0] = wptype(2.0) * v2[0][0]
        v21[0] = wptype(3.0) * v2[0][1]

        v30[0] = wptype(4.0) * v3[0][0]
        v31[0] = wptype(5.0) * v3[0][1]
        v32[0] = wptype(6.0) * v3[0][2]

        v40[0] = wptype(7.0) * v4[0][0]
        v41[0] = wptype(8.0) * v4[0][1]
        v42[0] = wptype(9.0) * v4[0][2]
        v43[0] = wptype(10.0) * v4[0][3]

        v50[0] = wptype(11.0) * v5[0][0]
        v51[0] = wptype(12.0) * v5[0][1]
        v52[0] = wptype(13.0) * v5[0][2]
        v53[0] = wptype(14.0) * v5[0][3]
        v54[0] = wptype(15.0) * v5[0][4]

    kernel = getkernel(check_indexing,suffix=dtype.__name__)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[ v2,v3,v4,v5 ], outputs=[v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)
    
    if dtype in np_float_types:
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54]):
            tape.backward(loss=l)
            allgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [v2,v3,v4,v5] ])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = i+2
            assert_np_equal(allgrads,expected_grads, tol=tol)
            tape.zero()
    
    assert_np_equal(v20.numpy()[0], 2.0 * v2.numpy()[0,0], tol=tol)
    assert_np_equal(v21.numpy()[0], 3.0 * v2.numpy()[0,1], tol=tol)
    assert_np_equal(v30.numpy()[0], 4.0 * v3.numpy()[0,0], tol=tol)
    assert_np_equal(v31.numpy()[0], 5.0 * v3.numpy()[0,1], tol=tol)
    assert_np_equal(v32.numpy()[0], 6.0 * v3.numpy()[0,2], tol=tol)
    assert_np_equal(v40.numpy()[0], 7.0 * v4.numpy()[0,0], tol=tol)
    assert_np_equal(v41.numpy()[0], 8.0 * v4.numpy()[0,1], tol=tol)
    assert_np_equal(v42.numpy()[0], 9.0 * v4.numpy()[0,2], tol=tol)
    assert_np_equal(v43.numpy()[0], 10.0 * v4.numpy()[0,3], tol=tol)
    assert_np_equal(v50.numpy()[0], 11.0 * v5.numpy()[0,0], tol=tol)
    assert_np_equal(v51.numpy()[0], 12.0 * v5.numpy()[0,1], tol=tol)
    assert_np_equal(v52.numpy()[0], 13.0 * v5.numpy()[0,2], tol=tol)
    assert_np_equal(v53.numpy()[0], 14.0 * v5.numpy()[0,3], tol=tol)
    assert_np_equal(v54.numpy()[0], 15.0 * v5.numpy()[0,4], tol=tol)
    

def test_equality(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 1.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    v20 =  wp.array([1.0,2.0], dtype=vec2, requires_grad=True, device=device)
    v21 =  wp.array([1.0,3.0], dtype=vec2, requires_grad=True, device=device)
    v22 =  wp.array([3.0,2.0], dtype=vec2, requires_grad=True, device=device)

    v30 =  wp.array([1.0,2.0,3.0], dtype=vec3, requires_grad=True, device=device)
    v31 =  wp.array([-1.0,2.0,3.0], dtype=vec3, requires_grad=True, device=device)
    v32 =  wp.array([1.0,-2.0,3.0], dtype=vec3, requires_grad=True, device=device)
    v33 =  wp.array([1.0,2.0,-3.0], dtype=vec3, requires_grad=True, device=device)

    v40 =  wp.array([1.0,2.0,3.0,4.0], dtype=vec4, requires_grad=True, device=device)
    v41 =  wp.array([-1.0,2.0,3.0,4.0], dtype=vec4, requires_grad=True, device=device)
    v42 =  wp.array([1.0,-2.0,3.0,4.0], dtype=vec4, requires_grad=True, device=device)
    v43 =  wp.array([1.0,2.0,-3.0,4.0], dtype=vec4, requires_grad=True, device=device)
    v44 =  wp.array([1.0,2.0,3.0,-4.0], dtype=vec4, requires_grad=True, device=device)
    
    v50 =  wp.array([1.0,2.0,3.0,4.0,5.0], dtype=vec5, requires_grad=True, device=device)
    v51 =  wp.array([-1.0,2.0,3.0,4.0,5.0], dtype=vec5, requires_grad=True, device=device)
    v52 =  wp.array([1.0,-2.0,3.0,4.0,5.0], dtype=vec5, requires_grad=True, device=device)
    v53 =  wp.array([1.0,2.0,-3.0,4.0,5.0], dtype=vec5, requires_grad=True, device=device)
    v54 =  wp.array([1.0,2.0,3.0,-4.0,5.0], dtype=vec5, requires_grad=True, device=device)
    v55 =  wp.array([1.0,2.0,3.0,4.0,-5.0], dtype=vec5, requires_grad=True, device=device)

    def check_equality(
        v20: wp.array(dtype=vec2),
        v21: wp.array(dtype=vec2),
        v22: wp.array(dtype=vec2),
        
        v30: wp.array(dtype=vec3),
        v31: wp.array(dtype=vec3),
        v32: wp.array(dtype=vec3),
        v33: wp.array(dtype=vec3),
        
        v40: wp.array(dtype=vec4),
        v41: wp.array(dtype=vec4),
        v42: wp.array(dtype=vec4),
        v43: wp.array(dtype=vec4),
        v44: wp.array(dtype=vec4),
        
        v50: wp.array(dtype=vec5),
        v51: wp.array(dtype=vec5),
        v52: wp.array(dtype=vec5),
        v53: wp.array(dtype=vec5),
        v54: wp.array(dtype=vec5),
        v55: wp.array(dtype=vec5),
    ):
        wp.expect_eq( v20[0], v20[0] )
        wp.expect_neq( v21[0], v20[0] )
        wp.expect_neq( v22[0], v20[0] )

        wp.expect_eq( v30[0], v30[0] )
        wp.expect_neq( v31[0], v30[0] )
        wp.expect_neq( v32[0], v30[0] )
        wp.expect_neq( v33[0], v30[0] )

        wp.expect_eq( v40[0], v40[0] )
        wp.expect_neq( v41[0], v40[0] )
        wp.expect_neq( v42[0], v40[0] )
        wp.expect_neq( v43[0], v40[0] )
        wp.expect_neq( v44[0], v40[0] )

        wp.expect_eq( v50[0], v50[0] )
        wp.expect_neq( v51[0], v50[0] )
        wp.expect_neq( v52[0], v50[0] )
        wp.expect_neq( v53[0], v50[0] )
        wp.expect_neq( v54[0], v50[0] )
        wp.expect_neq( v55[0], v50[0] )

    kernel = getkernel(check_equality,suffix=dtype.__name__)
    wp.launch(kernel, dim=1, inputs=[
        v20,v21,v22,
        v30,v31,v32,v33,
        v40,v41,v42,v43,v44,
        v50,v51,v52,v53,v54,v55,
        ], outputs=[], device=device
    )

def test_negation(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_negation(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v2out: wp.array(dtype=vec2),
        v3out: wp.array(dtype=vec3),
        v4out: wp.array(dtype=vec4),
        v5out: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        v2result = -v2load
        v3result = -v3load
        v4result = -v4load
        v5result = -v5load

        v2out[0] = v2result
        v3out[0] = v3result
        v4out[0] = v4result
        v5out[0] = v5result

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]

    kernel = getkernel(check_negation,suffix=dtype.__name__)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5_np = randvals((1,5),dtype)
    v5 = wp.array(v5_np, dtype=vec5, requires_grad=True, device=device)

    v2out = wp.zeros(1, dtype=vec2, device=device)
    v3out = wp.zeros(1, dtype=vec3, device=device)
    v4out = wp.zeros(1, dtype=vec4, device=device)
    v5out = wp.zeros(1, dtype=vec5, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[v2,v3,v4,v5 ], outputs=[v2out,v3out,v4out,v5out,v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)

    if dtype in np_float_types:
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54]):
            tape.backward(loss=l)
            allgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [v2,v3,v4,v5] ])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = [-2,-2,-3,-3,-3,-4,-4,-4,-4,-5,-5,-5,-5,-5][i]
            assert_np_equal(allgrads,expected_grads, tol=tol)
            tape.zero()

    assert_np_equal(v2out.numpy()[0], -2.0 * v2.numpy()[0], tol=tol)
    assert_np_equal(v3out.numpy()[0], -3.0 * v3.numpy()[0], tol=tol)
    assert_np_equal(v4out.numpy()[0], -4.0 * v4.numpy()[0], tol=tol)
    assert_np_equal(v5out.numpy()[0], -5.0 * v5.numpy()[0], tol=tol)
    

def test_scalar_multiplication(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_mul(
        s: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(6.0) * v5[0]

        sload = wptype(5.0) * s[0]

        v2result = sload * v2load
        v3result = sload * v3load
        v4result = sload * v4load
        v5result = sload * v5load

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]
    
    s = wp.array(randvals([1],dtype), requires_grad=True, device=device)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_mul,suffix=dtype.__name__), dim=1, inputs=[s,v2,v3,v4,v5,], outputs=[v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)

    assert_np_equal(v20.numpy()[0], 2.0*5 * s.numpy()[0] * v2.numpy()[0,0], tol=tol)
    assert_np_equal(v21.numpy()[0], 2.0*5 * s.numpy()[0] * v2.numpy()[0,1], tol=tol)

    assert_np_equal(v30.numpy()[0], 3.0*5 * s.numpy()[0] * v3.numpy()[0,0], tol=10*tol)
    assert_np_equal(v31.numpy()[0], 3.0*5 * s.numpy()[0] * v3.numpy()[0,1], tol=10*tol)
    assert_np_equal(v32.numpy()[0], 3.0*5 * s.numpy()[0] * v3.numpy()[0,2], tol=10*tol)

    assert_np_equal(v40.numpy()[0], 4.0*5 * s.numpy()[0] * v4.numpy()[0,0], tol=10*tol)
    assert_np_equal(v41.numpy()[0], 4.0*5 * s.numpy()[0] * v4.numpy()[0,1], tol=10*tol)
    assert_np_equal(v42.numpy()[0], 4.0*5 * s.numpy()[0] * v4.numpy()[0,2], tol=10*tol)
    assert_np_equal(v43.numpy()[0], 4.0*5 * s.numpy()[0] * v4.numpy()[0,3], tol=10*tol)

    assert_np_equal(v50.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,0], tol=10*tol)
    assert_np_equal(v51.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,1], tol=10*tol)
    assert_np_equal(v52.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,2], tol=10*tol)
    assert_np_equal(v53.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,3], tol=10*tol)
    assert_np_equal(v54.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,4], tol=10*tol)

    incmps = np.concatenate([ v.numpy()[0] for v in [v2,v3,v4,v5] ])


    if dtype in np_float_types:
        multfactors = [2*5,2*5,3*5,3*5,3*5, 4*5,4*5,4*5,4*5, 6*5,6*5,6*5,6*5,6*5,]
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43]):
            tape.backward(loss=l)
            sgrad = tape.gradients[s].numpy()[0]
            assert_np_equal(sgrad,multfactors[i] * incmps[i], tol=10*tol)
            allgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [v2,v3,v4] ])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = s.numpy()[0] * multfactors[i]
            assert_np_equal(allgrads,expected_grads, tol=10*tol)
            tape.zero()


def test_scalar_multiplication_rightmul(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_rightmul(
        s: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(6.0) * v5[0]

        sload = wptype(5.0) * s[0]

        v2result = v2load * sload
        v3result = v3load * sload
        v4result = v4load * sload
        v5result = v5load * sload

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]
    
    s = wp.array(randvals([1],dtype), requires_grad=True, device=device)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_rightmul,suffix=dtype.__name__), dim=1, inputs=[s,v2,v3,v4,v5,], outputs=[v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)

    assert_np_equal(v20.numpy()[0], 2.0*5 * s.numpy()[0] * v2.numpy()[0,0], tol=tol)
    assert_np_equal(v21.numpy()[0], 2.0*5 * s.numpy()[0] * v2.numpy()[0,1], tol=tol)

    assert_np_equal(v30.numpy()[0], 3.0*5 * s.numpy()[0] * v3.numpy()[0,0], tol=10*tol)
    assert_np_equal(v31.numpy()[0], 3.0*5 * s.numpy()[0] * v3.numpy()[0,1], tol=10*tol)
    assert_np_equal(v32.numpy()[0], 3.0*5 * s.numpy()[0] * v3.numpy()[0,2], tol=10*tol)

    assert_np_equal(v40.numpy()[0], 4.0*5 * s.numpy()[0] * v4.numpy()[0,0], tol=10*tol)
    assert_np_equal(v41.numpy()[0], 4.0*5 * s.numpy()[0] * v4.numpy()[0,1], tol=10*tol)
    assert_np_equal(v42.numpy()[0], 4.0*5 * s.numpy()[0] * v4.numpy()[0,2], tol=10*tol)
    assert_np_equal(v43.numpy()[0], 4.0*5 * s.numpy()[0] * v4.numpy()[0,3], tol=10*tol)

    assert_np_equal(v50.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,0], tol=10*tol)
    assert_np_equal(v51.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,1], tol=10*tol)
    assert_np_equal(v52.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,2], tol=10*tol)
    assert_np_equal(v53.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,3], tol=10*tol)
    assert_np_equal(v54.numpy()[0], 6.0*5 * s.numpy()[0] * v5.numpy()[0,4], tol=10*tol)

    incmps = np.concatenate([ v.numpy()[0] for v in [v2,v3,v4,v5] ])

    if dtype in np_float_types:
        multfactors = [2*5,2*5,3*5,3*5,3*5, 4*5,4*5,4*5,4*5, 6*5,6*5,6*5,6*5,6*5,]
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43]):
            tape.backward(loss=l)
            sgrad = tape.gradients[s].numpy()[0]
            assert_np_equal(sgrad,multfactors[i] * incmps[i], tol=10*tol)
            allgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [v2,v3,v4] ])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = s.numpy()[0] * multfactors[i]
            assert_np_equal(allgrads,expected_grads, tol=10*tol)
            tape.zero()

def test_cw_multiplication(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_cw_mul(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        s2load = wptype(1.0) * s2[0]
        s3load = wptype(2.0) * s3[0]
        s4load = wptype(3.0) * s4[0]
        s5load = wptype(4.0) * s5[0]

        v2result = wp.cw_mul(s2load, v2load)
        v3result = wp.cw_mul(s3load, v3load)
        v4result = wp.cw_mul(s4load, v4load)
        v5result = wp.cw_mul(s5load, v5load)

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]
    
    s2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_cw_mul,suffix=dtype.__name__), dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)

    assert_np_equal(v20.numpy()[0], 2.0 * s2.numpy()[0,0] * v2.numpy()[0,0], tol=10*tol)
    assert_np_equal(v21.numpy()[0], 2.0 * s2.numpy()[0,1] * v2.numpy()[0,1], tol=10*tol)

    assert_np_equal(v30.numpy()[0], 6.0 * s3.numpy()[0,0] * v3.numpy()[0,0], tol=10*tol)
    assert_np_equal(v31.numpy()[0], 6.0 * s3.numpy()[0,1] * v3.numpy()[0,1], tol=10*tol)
    assert_np_equal(v32.numpy()[0], 6.0 * s3.numpy()[0,2] * v3.numpy()[0,2], tol=10*tol)

    assert_np_equal(v40.numpy()[0], 12.0 * s4.numpy()[0,0] * v4.numpy()[0,0], tol=10*tol)
    assert_np_equal(v41.numpy()[0], 12.0 * s4.numpy()[0,1] * v4.numpy()[0,1], tol=10*tol)
    assert_np_equal(v42.numpy()[0], 12.0 * s4.numpy()[0,2] * v4.numpy()[0,2], tol=10*tol)
    assert_np_equal(v43.numpy()[0], 12.0 * s4.numpy()[0,3] * v4.numpy()[0,3], tol=10*tol)

    assert_np_equal(v50.numpy()[0], 20.0 * s5.numpy()[0,0] * v5.numpy()[0,0], tol=10*tol)
    assert_np_equal(v51.numpy()[0], 20.0 * s5.numpy()[0,1] * v5.numpy()[0,1], tol=10*tol)
    assert_np_equal(v52.numpy()[0], 20.0 * s5.numpy()[0,2] * v5.numpy()[0,2], tol=10*tol)
    assert_np_equal(v53.numpy()[0], 20.0 * s5.numpy()[0,3] * v5.numpy()[0,3], tol=10*tol)
    assert_np_equal(v54.numpy()[0], 20.0 * s5.numpy()[0,4] * v5.numpy()[0,4], tol=10*tol)

    incmps = np.concatenate([ v.numpy()[0] for v in [v2,v3,v4,v5] ])
    scmps = np.concatenate([ v.numpy()[0] for v in [s2,s3,s4,s5] ])
    multfactors = [2.0,2.0,6.0,6.0,6.0,12.0,12.0,12.0,12.0,20.0,20.0,20.0,20.0,20.0]
    
    if dtype in np_float_types:
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54]):
            tape.backward(loss=l)
            sgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [s2,s3,s4,s5] ])
            expected_grads = np.zeros_like(sgrads)
            expected_grads[i] = incmps[i] * multfactors[i]
            assert_np_equal(sgrads,expected_grads, tol=10*tol)
            
            allgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [v2,v3,v4,v5] ])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = scmps[i] * multfactors[i]
            assert_np_equal(allgrads,expected_grads, tol=10*tol)
            
            tape.zero()


def test_scalar_division(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_div(
        s: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(6.0) * v5[0]

        sload = wptype(5.0) * s[0]

        v2result = v2load / sload
        v3result = v3load / sload
        v4result = v4load / sload
        v5result = v5load / sload

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]
    
    s = wp.array(randvals([1],dtype), requires_grad=True, device=device)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_div,suffix=dtype.__name__), dim=1, inputs=[s,v2,v3,v4,v5,], outputs=[v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)

    if dtype in np_int_types:

        assert_np_equal(v20.numpy()[0], 2 * v2.numpy()[0,0] // ( 5 * s.numpy()[0]), tol=tol)
        assert_np_equal(v21.numpy()[0], 2 * v2.numpy()[0,1] // ( 5 * s.numpy()[0]), tol=tol)

        assert_np_equal(v30.numpy()[0], 3 * v3.numpy()[0,0] // ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v31.numpy()[0], 3 * v3.numpy()[0,1] // ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v32.numpy()[0], 3 * v3.numpy()[0,2] // ( 5 * s.numpy()[0]), tol=10*tol)

        assert_np_equal(v40.numpy()[0], 4 * v4.numpy()[0,0] // ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v41.numpy()[0], 4 * v4.numpy()[0,1] // ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v42.numpy()[0], 4 * v4.numpy()[0,2] // ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v43.numpy()[0], 4 * v4.numpy()[0,3] // ( 5 * s.numpy()[0]), tol=10*tol)

        assert_np_equal(v50.numpy()[0], 6 * v5.numpy()[0,0] // ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v51.numpy()[0], 6 * v5.numpy()[0,1] // ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v52.numpy()[0], 6 * v5.numpy()[0,2] // ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v53.numpy()[0], 6 * v5.numpy()[0,3] // ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v54.numpy()[0], 6 * v5.numpy()[0,4] // ( 5 * s.numpy()[0]), tol=10*tol)

    else:
            
        assert_np_equal(v20.numpy()[0], 2 * v2.numpy()[0,0] / ( 5 * s.numpy()[0]), tol=tol)
        assert_np_equal(v21.numpy()[0], 2 * v2.numpy()[0,1] / ( 5 * s.numpy()[0]), tol=tol)

        assert_np_equal(v30.numpy()[0], 3 * v3.numpy()[0,0] / ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v31.numpy()[0], 3 * v3.numpy()[0,1] / ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v32.numpy()[0], 3 * v3.numpy()[0,2] / ( 5 * s.numpy()[0]), tol=10*tol)

        assert_np_equal(v40.numpy()[0], 4 * v4.numpy()[0,0] / ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v41.numpy()[0], 4 * v4.numpy()[0,1] / ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v42.numpy()[0], 4 * v4.numpy()[0,2] / ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v43.numpy()[0], 4 * v4.numpy()[0,3] / ( 5 * s.numpy()[0]), tol=10*tol)

        assert_np_equal(v50.numpy()[0], 6 * v5.numpy()[0,0] / ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v51.numpy()[0], 6 * v5.numpy()[0,1] / ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v52.numpy()[0], 6 * v5.numpy()[0,2] / ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v53.numpy()[0], 6 * v5.numpy()[0,3] / ( 5 * s.numpy()[0]), tol=10*tol)
        assert_np_equal(v54.numpy()[0], 6 * v5.numpy()[0,4] / ( 5 * s.numpy()[0]), tol=10*tol)

    incmps = np.concatenate([ v.numpy()[0] for v in [v2,v3,v4,v5] ])

    if dtype in np_float_types:
        multfactors = [2.0/5,2.0/5,3.0/5,3.0/5,3.0/5,4.0/5,4.0/5,4.0/5,4.0/5,6.0/5,6.0/5,6.0/5,6.0/5,6.0/5]
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54]):
            
            tape.backward(loss=l)
            sgrad = tape.gradients[s].numpy()[0]

            # d/ds v/s = -v/s^2
            assert_np_equal(sgrad,-multfactors[i] * incmps[i] / (s.numpy()[0]*s.numpy()[0]), tol=10*tol)

            allgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [v2,v3,v4,v5] ])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = multfactors[i] / s.numpy()[0]
            
            # d/dv v/s = 1/s
            assert_np_equal(allgrads,expected_grads, tol=tol)
            tape.zero()

def test_cw_division(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_cw_div(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        s2load = wptype(1.0) * s2[0]
        s3load = wptype(2.0) * s3[0]
        s4load = wptype(3.0) * s4[0]
        s5load = wptype(4.0) * s5[0]

        v2result = wp.cw_div(v2load, s2load)
        v3result = wp.cw_div(v3load, s3load)
        v4result = wp.cw_div(v4load, s4load)
        v5result = wp.cw_div(v5load, s5load)

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]
    
    s2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_cw_div,suffix=dtype.__name__), dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)

    if dtype in np_int_types:
        assert_np_equal(v20.numpy()[0], (2.0 * v2.numpy()[0,0]) // (1.0 * s2.numpy()[0,0]), tol=tol)
        assert_np_equal(v21.numpy()[0], (2.0 * v2.numpy()[0,1]) // (1.0 * s2.numpy()[0,1]), tol=tol)

        assert_np_equal(v30.numpy()[0], (3.0 * v3.numpy()[0,0]) // (2.0 * s3.numpy()[0,0]), tol=tol)
        assert_np_equal(v31.numpy()[0], (3.0 * v3.numpy()[0,1]) // (2.0 * s3.numpy()[0,1]), tol=tol)
        assert_np_equal(v32.numpy()[0], (3.0 * v3.numpy()[0,2]) // (2.0 * s3.numpy()[0,2]), tol=tol)

        assert_np_equal(v40.numpy()[0], (4.0 * v4.numpy()[0,0]) // (3.0 * s4.numpy()[0,0]), tol=tol)
        assert_np_equal(v41.numpy()[0], (4.0 * v4.numpy()[0,1]) // (3.0 * s4.numpy()[0,1]), tol=tol)
        assert_np_equal(v42.numpy()[0], (4.0 * v4.numpy()[0,2]) // (3.0 * s4.numpy()[0,2]), tol=tol)
        assert_np_equal(v43.numpy()[0], (4.0 * v4.numpy()[0,3]) // (3.0 * s4.numpy()[0,3]), tol=tol)

        assert_np_equal(v50.numpy()[0], (5.0 * v5.numpy()[0,0]) // (4.0 * s5.numpy()[0,0]), tol=tol)
        assert_np_equal(v51.numpy()[0], (5.0 * v5.numpy()[0,1]) // (4.0 * s5.numpy()[0,1]), tol=tol)
        assert_np_equal(v52.numpy()[0], (5.0 * v5.numpy()[0,2]) // (4.0 * s5.numpy()[0,2]), tol=tol)
        assert_np_equal(v53.numpy()[0], (5.0 * v5.numpy()[0,3]) // (4.0 * s5.numpy()[0,3]), tol=tol)
        assert_np_equal(v54.numpy()[0], (5.0 * v5.numpy()[0,4]) // (4.0 * s5.numpy()[0,4]), tol=tol)
    else:
        assert_np_equal(v20.numpy()[0], (2.0 * v2.numpy()[0,0]) / (1.0 * s2.numpy()[0,0]), tol=tol)
        assert_np_equal(v21.numpy()[0], (2.0 * v2.numpy()[0,1]) / (1.0 * s2.numpy()[0,1]), tol=tol)

        assert_np_equal(v30.numpy()[0], (3.0 * v3.numpy()[0,0]) / (2.0 * s3.numpy()[0,0]), tol=tol)
        assert_np_equal(v31.numpy()[0], (3.0 * v3.numpy()[0,1]) / (2.0 * s3.numpy()[0,1]), tol=tol)
        assert_np_equal(v32.numpy()[0], (3.0 * v3.numpy()[0,2]) / (2.0 * s3.numpy()[0,2]), tol=tol)

        assert_np_equal(v40.numpy()[0], (4.0 * v4.numpy()[0,0]) / (3.0 * s4.numpy()[0,0]), tol=tol)
        assert_np_equal(v41.numpy()[0], (4.0 * v4.numpy()[0,1]) / (3.0 * s4.numpy()[0,1]), tol=tol)
        assert_np_equal(v42.numpy()[0], (4.0 * v4.numpy()[0,2]) / (3.0 * s4.numpy()[0,2]), tol=tol)
        assert_np_equal(v43.numpy()[0], (4.0 * v4.numpy()[0,3]) / (3.0 * s4.numpy()[0,3]), tol=tol)

        assert_np_equal(v50.numpy()[0], (5.0 * v5.numpy()[0,0]) / (4.0 * s5.numpy()[0,0]), tol=tol)
        assert_np_equal(v51.numpy()[0], (5.0 * v5.numpy()[0,1]) / (4.0 * s5.numpy()[0,1]), tol=tol)
        assert_np_equal(v52.numpy()[0], (5.0 * v5.numpy()[0,2]) / (4.0 * s5.numpy()[0,2]), tol=tol)
        assert_np_equal(v53.numpy()[0], (5.0 * v5.numpy()[0,3]) / (4.0 * s5.numpy()[0,3]), tol=tol)
        assert_np_equal(v54.numpy()[0], (5.0 * v5.numpy()[0,4]) / (4.0 * s5.numpy()[0,4]), tol=tol)

    if dtype in np_float_types:
        incmps = np.concatenate([ v.numpy()[0] for v in [v2,v3,v4,v5] ])
        scmps = np.concatenate([ v.numpy()[0] for v in [s2,s3,s4,s5] ])
        multfactors = [2.0/1.0,2.0/1.0,3.0/2.0,3.0/2.0,3.0/2.0,4.0/3.0,4.0/3.0,4.0/3.0,4.0/3.0,5.0/4.0,5.0/4.0,5.0/4.0,5.0/4.0,5.0/4.0]
        
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54]):
            tape.backward(loss=l)
            sgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [s2,s3,s4,s5] ])
            expected_grads = np.zeros_like(sgrads)
            
            # d/ds v/s = -v/s^2
            expected_grads[i] = -incmps[i] * multfactors[i] / (scmps[i] * scmps[i])
            assert_np_equal(sgrads,expected_grads, tol=20*tol)
            
            allgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [v2,v3,v4,v5] ])
            expected_grads = np.zeros_like(allgrads)
            
            # d/dv v/s = 1/s
            expected_grads[i] = multfactors[i] / scmps[i]
            assert_np_equal(allgrads,expected_grads, tol=tol)
            
            tape.zero()


def test_addition(test,device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_add(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        s2load = wptype(1.0) * s2[0]
        s3load = wptype(2.0) * s3[0]
        s4load = wptype(3.0) * s4[0]
        s5load = wptype(4.0) * s5[0]

        v2result = v2load + s2load
        v3result = v3load + s3load
        v4result = v4load + s4load
        v5result = v5load + s5load

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]
    
    s2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_add,suffix=dtype.__name__), dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)

    assert_np_equal(v20.numpy()[0], 2.0 * v2.numpy()[0,0] + 1.0 * s2.numpy()[0,0], tol=tol)
    assert_np_equal(v21.numpy()[0], 2.0 * v2.numpy()[0,1] + 1.0 * s2.numpy()[0,1], tol=tol)

    assert_np_equal(v30.numpy()[0], 3.0 * v3.numpy()[0,0] + 2.0 * s3.numpy()[0,0], tol=tol)
    assert_np_equal(v31.numpy()[0], 3.0 * v3.numpy()[0,1] + 2.0 * s3.numpy()[0,1], tol=tol)
    assert_np_equal(v32.numpy()[0], 3.0 * v3.numpy()[0,2] + 2.0 * s3.numpy()[0,2], tol=tol)

    assert_np_equal(v40.numpy()[0], 4.0 * v4.numpy()[0,0] + 3.0 * s4.numpy()[0,0], tol=tol)
    assert_np_equal(v41.numpy()[0], 4.0 * v4.numpy()[0,1] + 3.0 * s4.numpy()[0,1], tol=tol)
    assert_np_equal(v42.numpy()[0], 4.0 * v4.numpy()[0,2] + 3.0 * s4.numpy()[0,2], tol=tol)
    assert_np_equal(v43.numpy()[0], 4.0 * v4.numpy()[0,3] + 3.0 * s4.numpy()[0,3], tol=tol)

    assert_np_equal(v50.numpy()[0], 5.0 * v5.numpy()[0,0] + 4.0 * s5.numpy()[0,0], tol=tol)
    assert_np_equal(v51.numpy()[0], 5.0 * v5.numpy()[0,1] + 4.0 * s5.numpy()[0,1], tol=tol)
    assert_np_equal(v52.numpy()[0], 5.0 * v5.numpy()[0,2] + 4.0 * s5.numpy()[0,2], tol=tol)
    assert_np_equal(v53.numpy()[0], 5.0 * v5.numpy()[0,3] + 4.0 * s5.numpy()[0,3], tol=tol)
    assert_np_equal(v54.numpy()[0], 5.0 * v5.numpy()[0,4] + 4.0 * s5.numpy()[0,4], tol=2*tol)

    if dtype in np_float_types:
        infactors = [2,2,3,3,3,4,4,4,4,5,5,5,5,5]
        sfactors = [1.0,1.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,4.0]
        
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54]):
            tape.backward(loss=l)
            sgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [s2,s3,s4,s5] ])
            expected_grads = np.zeros_like(sgrads)
            
            expected_grads[i] = sfactors[i]
            assert_np_equal(sgrads,expected_grads, tol=10*tol)
            
            allgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [v2,v3,v4,v5] ])
            expected_grads = np.zeros_like(allgrads)
            
            # d/dv v/s = 1/s
            expected_grads[i] = infactors[i]
            assert_np_equal(allgrads,expected_grads, tol=tol)
            
            tape.zero()

def test_subtraction_unsigned(test,device,dtype):
    
    np.random.seed(123)
    
    tol = {
        np.float16: 1.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    #print("vec2:",vec2)
    #asdfas

    def check_subtraction_unsigned():

        wp.expect_eq(
            vec2(wptype(3),wptype(4)) - vec2(wptype(1),wptype(2)),
            vec2(wptype(2),wptype(2))
        )
        wp.expect_eq(
            vec3(wptype(3),wptype(4),wptype(4),) - vec3(wptype(1),wptype(2),wptype(3)),
            vec3(wptype(2),wptype(2),wptype(1))
        )
        wp.expect_eq(
            vec4(wptype(3),wptype(4),wptype(4),wptype(5),) - vec4(wptype(1),wptype(2),wptype(3),wptype(4)),
            vec4(wptype(2),wptype(2),wptype(1),wptype(1))
        )
        wp.expect_eq(
            vec5(wptype(3),wptype(4),wptype(4),wptype(5),wptype(4),) - vec5(wptype(1),wptype(2),wptype(3),wptype(4),wptype(4)),
            vec5(wptype(2),wptype(2),wptype(1),wptype(1),wptype(0))
        )

    kernel = getkernel(check_subtraction_unsigned,suffix=dtype.__name__)
    wp.launch(kernel, dim=1, inputs=[], outputs=[], device=device)


def test_subtraction(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_subtraction(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        s2load = wptype(1.0) * s2[0]
        s3load = wptype(2.0) * s3[0]
        s4load = wptype(3.0) * s4[0]
        s5load = wptype(4.0) * s5[0]

        v2result = v2load - s2load
        v3result = v3load - s3load
        v4result = v4load - s4load
        v5result = v5load - s5load

        v20[0] = v2result[0]
        v21[0] = v2result[1]

        v30[0] = v3result[0]
        v31[0] = v3result[1]
        v32[0] = v3result[2]

        v40[0] = v4result[0]
        v41[0] = v4result[1]
        v42[0] = v4result[2]
        v43[0] = v4result[3]

        v50[0] = v5result[0]
        v51[0] = v5result[1]
        v52[0] = v5result[2]
        v53[0] = v5result[3]
        v54[0] = v5result[4]
    
    s2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_subtraction,suffix=dtype.__name__), dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54], device=device)

    assert_np_equal(v20.numpy()[0], 2.0 * v2.numpy()[0,0] - 1.0 * s2.numpy()[0,0], tol=tol)
    assert_np_equal(v21.numpy()[0], 2.0 * v2.numpy()[0,1] - 1.0 * s2.numpy()[0,1], tol=tol)

    assert_np_equal(v30.numpy()[0], 3.0 * v3.numpy()[0,0] - 2.0 * s3.numpy()[0,0], tol=tol)
    assert_np_equal(v31.numpy()[0], 3.0 * v3.numpy()[0,1] - 2.0 * s3.numpy()[0,1], tol=tol)
    assert_np_equal(v32.numpy()[0], 3.0 * v3.numpy()[0,2] - 2.0 * s3.numpy()[0,2], tol=tol)

    assert_np_equal(v40.numpy()[0], 4.0 * v4.numpy()[0,0] - 3.0 * s4.numpy()[0,0], tol=2*tol)
    assert_np_equal(v41.numpy()[0], 4.0 * v4.numpy()[0,1] - 3.0 * s4.numpy()[0,1], tol=2*tol)
    assert_np_equal(v42.numpy()[0], 4.0 * v4.numpy()[0,2] - 3.0 * s4.numpy()[0,2], tol=2*tol)
    assert_np_equal(v43.numpy()[0], 4.0 * v4.numpy()[0,3] - 3.0 * s4.numpy()[0,3], tol=2*tol)

    assert_np_equal(v50.numpy()[0], 5.0 * v5.numpy()[0,0] - 4.0 * s5.numpy()[0,0], tol=tol)
    assert_np_equal(v51.numpy()[0], 5.0 * v5.numpy()[0,1] - 4.0 * s5.numpy()[0,1], tol=tol)
    assert_np_equal(v52.numpy()[0], 5.0 * v5.numpy()[0,2] - 4.0 * s5.numpy()[0,2], tol=tol)
    assert_np_equal(v53.numpy()[0], 5.0 * v5.numpy()[0,3] - 4.0 * s5.numpy()[0,3], tol=tol)
    assert_np_equal(v54.numpy()[0], 5.0 * v5.numpy()[0,4] - 4.0 * s5.numpy()[0,4], tol=tol)

    if dtype in np_float_types:
        infactors = [2,2,3,3,3,4,4,4,4,5,5,5,5,5]
        sfactors = [1.0,1.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,4.0,4.0,4.0,4.0,4.0]
        
        for i,l in enumerate([v20,v21,v30,v31,v32,v40,v41,v42,v43,v50,v51,v52,v53,v54]):
            tape.backward(loss=l)
            sgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [s2,s3,s4,s5] ])
            expected_grads = np.zeros_like(sgrads)
            
            expected_grads[i] = -sfactors[i]
            assert_np_equal(sgrads,expected_grads, tol=10*tol)
            
            allgrads = np.concatenate([ tape.gradients[v].numpy()[0] for v in [v2,v3,v4,v5] ])
            expected_grads = np.zeros_like(allgrads)
            
            # d/dv v/s = 1/s
            expected_grads[i] = infactors[i]
            assert_np_equal(allgrads,expected_grads, tol=tol)
            
            tape.zero()


def test_dotproduct(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_dot(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        dot2: wp.array(dtype=wptype),
        dot3: wp.array(dtype=wptype),
        dot4: wp.array(dtype=wptype),
        dot5: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(2.0) * v4[0]
        v5load = wptype(3.0) * v5[0]

        s2load = wptype(3.0) * s2[0]
        s3load = wptype(2.0) * s3[0]
        s4load = wptype(3.0) * s4[0]
        s5load = wptype(2.0) * s5[0]

        dot2[0] = wp.dot(v2load,s2load)
        dot3[0] = wp.dot(v3load,s3load)
        dot4[0] = wp.dot(v4load,s4load)
        dot5[0] = wp.dot(v5load,s5load)

    s2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    dot2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot5 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_dot,suffix=dtype.__name__), dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[dot2,dot3,dot4,dot5], device=device)

    assert_np_equal(dot2.numpy()[0], 2.0 * 3.0 * (v2.numpy() * s2.numpy()).sum(), tol=10*tol)
    assert_np_equal(dot3.numpy()[0], 3.0 * 2.0 * (v3.numpy() * s3.numpy()).sum(), tol=10*tol)
    assert_np_equal(dot4.numpy()[0], 2.0 * 3.0 * (v4.numpy() * s4.numpy()).sum(), tol=10*tol)
    assert_np_equal(dot5.numpy()[0], 3.0 * 2.0 * (v5.numpy() * s5.numpy()).sum(), tol=10*tol)
    
    if dtype in np_float_types:
        tape.backward(loss=dot2)
        sgrads = tape.gradients[s2].numpy()[0]
        expected_grads = 2.0 * 3.0 * v2.numpy()[0]
        assert_np_equal(sgrads,expected_grads, tol=10*tol)

        vgrads = tape.gradients[v2].numpy()[0]
        expected_grads = 2.0 * 3.0 * s2.numpy()[0]
        assert_np_equal(vgrads,expected_grads, tol=tol)
        
        tape.zero()
        
        tape.backward(loss=dot3)
        sgrads = tape.gradients[s3].numpy()[0]
        expected_grads = 3.0 * 2.0 * v3.numpy()[0]
        assert_np_equal(sgrads,expected_grads, tol=10*tol)

        vgrads = tape.gradients[v3].numpy()[0]
        expected_grads = 3.0 * 2.0 * s3.numpy()[0]
        assert_np_equal(vgrads,expected_grads, tol=tol)
        
        tape.zero()
        
        tape.backward(loss=dot4)
        sgrads = tape.gradients[s4].numpy()[0]
        expected_grads = 2.0 * 3.0 * v4.numpy()[0]
        assert_np_equal(sgrads,expected_grads, tol=10*tol)

        vgrads = tape.gradients[v4].numpy()[0]
        expected_grads = 2.0 * 3.0 * s4.numpy()[0]
        assert_np_equal(vgrads,expected_grads, tol=tol)
        
        tape.zero()
        
        tape.backward(loss=dot5)
        sgrads = tape.gradients[s5].numpy()[0]
        expected_grads = 3.0 * 2.0 * v5.numpy()[0]
        assert_np_equal(sgrads,expected_grads, tol=10*tol)

        vgrads = tape.gradients[v5].numpy()[0]
        expected_grads = 3.0 * 2.0 * s5.numpy()[0]
        assert_np_equal(vgrads,expected_grads, tol=10*tol)
        
        tape.zero()


def test_length(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-7,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_length(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        l2: wp.array(dtype=wptype),
        l3: wp.array(dtype=wptype),
        l4: wp.array(dtype=wptype),
        l5: wp.array(dtype=wptype),
        l22: wp.array(dtype=wptype),
        l23: wp.array(dtype=wptype),
        l24: wp.array(dtype=wptype),
        l25: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        l2[0] = wp.length(v2load)
        l3[0] = wp.length(v3load)
        l4[0] = wp.length(v4load)
        l5[0] = wp.length(v5load)

        l22[0] = wp.length_sq(v2load)
        l23[0] = wp.length_sq(v3load)
        l24[0] = wp.length_sq(v4load)
        l25[0] = wp.length_sq(v5load)


    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    
    l2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l5 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    
    l22 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l23 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l24 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l25 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_length,suffix=dtype.__name__), dim=1, inputs=[v2,v3,v4,v5,], outputs=[l2,l3,l4,l5,l22,l23,l24,l25], device=device)

    assert_np_equal(l2.numpy()[0], 2.0 * np.linalg.norm(v2.numpy()), tol=10*tol)
    assert_np_equal(l3.numpy()[0], 3.0 * np.linalg.norm(v3.numpy()), tol=10*tol)
    assert_np_equal(l4.numpy()[0], 4.0 * np.linalg.norm(v4.numpy()), tol=10*tol)
    assert_np_equal(l5.numpy()[0], 5.0 * np.linalg.norm(v5.numpy()), tol=10*tol)

    assert_np_equal(l22.numpy()[0], 4.0 * np.linalg.norm(v2.numpy())**2, tol=10*tol)
    assert_np_equal(l23.numpy()[0], 9.0 * np.linalg.norm(v3.numpy())**2, tol=10*tol)
    assert_np_equal(l24.numpy()[0], 16.0 * np.linalg.norm(v4.numpy())**2, tol=10*tol)
    assert_np_equal(l25.numpy()[0], 25.0 * np.linalg.norm(v5.numpy())**2, tol=10*tol)
    
    
    tape.backward(loss=l2)
    grad = tape.gradients[v2].numpy()[0]
    expected_grad = 2.0 * v2.numpy()[0] / np.linalg.norm(v2.numpy())
    assert_np_equal(grad,expected_grad, tol=10*tol)
    tape.zero()

    tape.backward(loss=l3)
    grad = tape.gradients[v3].numpy()[0]
    expected_grad = 3.0 * v3.numpy()[0] / np.linalg.norm(v3.numpy())
    assert_np_equal(grad,expected_grad, tol=10*tol)
    tape.zero()
    
    tape.backward(loss=l4)
    grad = tape.gradients[v4].numpy()[0]
    expected_grad = 4.0 * v4.numpy()[0] / np.linalg.norm(v4.numpy())
    assert_np_equal(grad,expected_grad, tol=10*tol)
    tape.zero()
    
    tape.backward(loss=l5)
    grad = tape.gradients[v5].numpy()[0]
    expected_grad = 5.0 * v5.numpy()[0] / np.linalg.norm(v5.numpy())
    assert_np_equal(grad,expected_grad, tol=10*tol)
    tape.zero()
    
    tape.backward(loss=l22)
    grad = tape.gradients[v2].numpy()[0]
    expected_grad = 2 * 2.0 * 2.0 * v2.numpy()[0]
    assert_np_equal(grad,expected_grad, tol=10*tol)
    tape.zero()

    tape.backward(loss=l23)
    grad = tape.gradients[v3].numpy()[0]
    expected_grad = 2 * 3.0 * 3.0 * v3.numpy()[0]
    assert_np_equal(grad,expected_grad, tol=10*tol)
    tape.zero()
    
    tape.backward(loss=l24)
    grad = tape.gradients[v4].numpy()[0]
    expected_grad = 2 * 4.0 * 4.0 * v4.numpy()[0]
    assert_np_equal(grad,expected_grad, tol=10*tol)
    tape.zero()
    
    tape.backward(loss=l25)
    grad = tape.gradients[v5].numpy()[0]
    expected_grad = 2 * 5.0 * 5.0 * v5.numpy()[0]
    assert_np_equal(grad,expected_grad, tol=10*tol)
    tape.zero()
    
def test_normalize(test,device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_normalize(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        n20: wp.array(dtype=wptype),
        n21: wp.array(dtype=wptype),
        n30: wp.array(dtype=wptype),
        n31: wp.array(dtype=wptype),
        n32: wp.array(dtype=wptype),
        n40: wp.array(dtype=wptype),
        n41: wp.array(dtype=wptype),
        n42: wp.array(dtype=wptype),
        n43: wp.array(dtype=wptype),
        n50: wp.array(dtype=wptype),
        n51: wp.array(dtype=wptype),
        n52: wp.array(dtype=wptype),
        n53: wp.array(dtype=wptype),
        n54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        n2 = wp.normalize(v2load)
        n3 = wp.normalize(v3load)
        n4 = wp.normalize(v4load)
        n5 = wp.normalize(v5load)

        n20[0] = n2[0]
        n21[0] = n2[1]

        n30[0] = n3[0]
        n31[0] = n3[1]
        n32[0] = n3[2]

        n40[0] = n4[0]
        n41[0] = n4[1]
        n42[0] = n4[2]
        n43[0] = n4[3]

        n50[0] = n5[0]
        n51[0] = n5[1]
        n52[0] = n5[2]
        n53[0] = n5[3]
        n54[0] = n5[4]

    def check_normalize_alt(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        n20: wp.array(dtype=wptype),
        n21: wp.array(dtype=wptype),
        n30: wp.array(dtype=wptype),
        n31: wp.array(dtype=wptype),
        n32: wp.array(dtype=wptype),
        n40: wp.array(dtype=wptype),
        n41: wp.array(dtype=wptype),
        n42: wp.array(dtype=wptype),
        n43: wp.array(dtype=wptype),
        n50: wp.array(dtype=wptype),
        n51: wp.array(dtype=wptype),
        n52: wp.array(dtype=wptype),
        n53: wp.array(dtype=wptype),
        n54: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        n2 = v2load / wp.length(v2load)
        n3 = v3load / wp.length(v3load)
        n4 = v4load / wp.length(v4load)
        n5 = v5load / wp.length(v5load)

        n20[0] = n2[0]
        n21[0] = n2[1]

        n30[0] = n3[0]
        n31[0] = n3[1]
        n32[0] = n3[2]

        n40[0] = n4[0]
        n41[0] = n4[1]
        n42[0] = n4[2]
        n43[0] = n4[3]

        n50[0] = n5[0]
        n51[0] = n5[1]
        n52[0] = n5[2]
        n53[0] = n5[3]
        n54[0] = n5[4]


    # I've already tested the things I'm using in check_normalize_alt, so I'll just
    # make sure the two are giving the same results/gradients
    v2 = wp.array(randvals((1,2),dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1,4),dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1,5),dtype), dtype=vec5, requires_grad=True, device=device)
    
    n20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    
    n20_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n21_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n30_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n31_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n32_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n40_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n41_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n42_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n43_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n50_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n51_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n52_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n53_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n54_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    outputs0 = [n20,n21,n30,n31,n32,n40,n41,n42,n43,n50,n51,n52,n53,n54,]
    tape0 = wp.Tape()
    with tape0:
        wp.launch(getkernel(check_normalize,suffix=dtype.__name__), dim=1, inputs=[v2,v3,v4,v5,], outputs=outputs0, device=device)

    outputs1=[n20_alt,n21_alt,n30_alt,n31_alt,n32_alt,n40_alt,n41_alt,n42_alt,n43_alt,n50_alt,n51_alt,n52_alt,n53_alt,n54_alt,]
    tape1 = wp.Tape()
    with tape1:
        wp.launch(getkernel(check_normalize_alt,suffix=dtype.__name__), dim=1, inputs=[v2,v3,v4,v5,], outputs=outputs1, device=device)

    for ncmp,ncmpalt in zip(outputs0,outputs1):
        assert_np_equal(ncmp.numpy()[0], ncmpalt.numpy()[0], tol=10*tol)

    invecs = [v2,v2,v3,v3,v3,v4,v4,v4,v4,v5,v5,v5,v5,v5,]
    for ncmp,ncmpalt,v in zip(outputs0,outputs1,invecs):
        tape0.backward(loss=ncmp)
        tape1.backward(loss=ncmpalt)
        assert_np_equal(tape0.gradients[v].numpy()[0],tape1.gradients[v].numpy()[0], tol=10*tol)
        tape0.zero()
        tape1.zero()

def test_crossproduct(test,device,dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    def check_cross(
        s3: wp.array(dtype=vec3),
        v3: wp.array(dtype=vec3),
        c0: wp.array(dtype=wptype),
        c1: wp.array(dtype=wptype),
        c2: wp.array(dtype=wptype),
    ):
        v3load = wptype(3.0) * v3[0]
        s3load = wptype(2.0) * s3[0]

        c = wp.cross(s3load,v3load)

        c0[0] = c[0]
        c1[0] = c[1]
        c2[0] = c[2]

    s3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    v3 = wp.array(randvals((1,3),dtype), dtype=vec3, requires_grad=True, device=device)
    c0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    c1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    c2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(getkernel(check_cross,suffix=dtype.__name__), dim=1, inputs=[s3,v3,], outputs=[c0,c1,c2], device=device)

    result = np.cross(2.0 * s3.numpy(), 3.0 * v3.numpy())[0]
    assert_np_equal(c0.numpy()[0], result[0], tol=10*tol)
    assert_np_equal(c1.numpy()[0], result[1], tol=10*tol)
    assert_np_equal(c2.numpy()[0], result[2], tol=10*tol)

    if dtype in np_float_types:

        # c.x = sy vz - sz vy
        # c.y = sz vx - sx vz
        # c.z = sx vy - sy vx

        # ( d/dsx d/dsy d/dsz )c.x = ( 0 vz -vy )
        # ( d/dsx d/dsy d/dsz )c.y = ( -vz 0 vx )
        # ( d/dsx d/dsy d/dsz )c.z = ( vy -vx 0 )

        # ( d/dvx d/dvy d/dvz )c.x = (0 -sz sy)
        # ( d/dvx d/dvy d/dvz )c.y = (sz 0 -sx)
        # ( d/dvx d/dvy d/dvz )c.z = (-sy sx 0)
        
        tape.backward(loss=c0)
        assert_np_equal(tape.gradients[s3].numpy(), 3.0 * 2.0 * np.array([0,v3.numpy()[0,2],-v3.numpy()[0,1]]), tol=10*tol)
        assert_np_equal(tape.gradients[v3].numpy(), 3.0 * 2.0 * np.array([0,-s3.numpy()[0,2],s3.numpy()[0,1]]), tol=10*tol)
        tape.zero()
        
        tape.backward(loss=c1)
        assert_np_equal(tape.gradients[s3].numpy(), 3.0 * 2.0 * np.array([-v3.numpy()[0,2],0,v3.numpy()[0,0]]), tol=10*tol)
        assert_np_equal(tape.gradients[v3].numpy(), 3.0 * 2.0 * np.array([s3.numpy()[0,2],0,-s3.numpy()[0,0]]), tol=10*tol)
        tape.zero()
        
        tape.backward(loss=c2)
        assert_np_equal(tape.gradients[s3].numpy(), 3.0 * 2.0 * np.array([v3.numpy()[0,1],-v3.numpy()[0,0],0]), tol=10*tol)
        assert_np_equal(tape.gradients[v3].numpy(), 3.0 * 2.0 * np.array([-s3.numpy()[0,1],s3.numpy()[0,0],0]), tol=10*tol)
        tape.zero()


def register(parent):

    devices = wp.get_devices()

    class TestVec(parent):
        pass
    
    for dtype in np_unsigned_int_types:
        add_function_test(TestVec, f"test_subtraction_unsigned_{dtype.__name__}", test_subtraction_unsigned, devices=devices, dtype=dtype)

    for dtype in np_signed_int_types + np_float_types:
        add_function_test(TestVec, f"test_negation_{dtype.__name__}", test_negation, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_crossproduct_{dtype.__name__}", test_crossproduct, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_subtraction_{dtype.__name__}", test_subtraction, devices=devices, dtype=dtype)

    for dtype in np_scalar_types:
        add_function_test(TestVec, f"test_arrays_{dtype.__name__}", test_arrays, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_constructors_{dtype.__name__}", test_constructors, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_indexing_{dtype.__name__}", test_indexing, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_equality_{dtype.__name__}", test_equality, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_scalar_multiplication_{dtype.__name__}", test_scalar_multiplication, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_scalar_multiplication_rightmul_{dtype.__name__}", test_scalar_multiplication_rightmul, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_cw_multiplication_{dtype.__name__}", test_cw_multiplication, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_scalar_division_{dtype.__name__}", test_scalar_division, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_cw_division_{dtype.__name__}", test_cw_division, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_addition_{dtype.__name__}", test_addition, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_dotproduct_{dtype.__name__}", test_dotproduct, devices=devices, dtype=dtype)

    for dtype in np_float_types:
        add_function_test(TestVec, f"test_length_{dtype.__name__}", test_length, devices=devices, dtype=dtype)
        add_function_test(TestVec, f"test_normalize_{dtype.__name__}", test_normalize, devices=devices, dtype=dtype)

    return TestVec

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
