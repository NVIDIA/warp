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

def get_select_kernel(dtype):
    
    def output_select_kernel_fn(
        input: wp.array(dtype=dtype),
        index: int,
        out: wp.array(dtype=dtype),
    ):
        out[0] = input[index]
    
    return getkernel(output_select_kernel_fn,suffix=dtype.__name__)

def test_arrays(test, device,dtype):

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)
    mat32 = wp.mat(shape=(3,2),type=wptype)

    
    np.random.seed(123)
    
    v2_np = randvals([10,2,2],dtype)
    v3_np = randvals([10,3,3],dtype)
    v4_np = randvals([10,4,4],dtype)
    v5_np = randvals([10,5,5],dtype)
    v32_np = randvals([10,3,2],dtype)
    
    v2 = wp.array(v2_np, dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(v3_np, dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(v4_np, dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(v5_np, dtype=mat55, requires_grad=True, device=device)
    v32 = wp.array(v32_np, dtype=mat32, requires_grad=True, device=device)

    assert_np_equal(v2.numpy(), v2_np, tol=1.e-6)
    assert_np_equal(v3.numpy(), v3_np, tol=1.e-6)
    assert_np_equal(v4.numpy(), v4_np, tol=1.e-6)
    assert_np_equal(v5.numpy(), v5_np, tol=1.e-6)
    assert_np_equal(v32.numpy(), v32_np, tol=1.e-6)

    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    
    v2 = wp.array(v2_np, dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(v3_np, dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(v4_np, dtype=mat44, requires_grad=True, device=device)
    
    assert_np_equal(v2.numpy(), v2_np, tol=1.e-6)
    assert_np_equal(v3.numpy(), v3_np, tol=1.e-6)
    assert_np_equal(v4.numpy(), v4_np, tol=1.e-6)

def test_constructors(test, device,dtype):

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_scalar_mat_constructor(
        input: wp.array(dtype=wptype),
        outcomponents: wp.array(dtype=wptype),
    ):
        outcomponents[0] = wptype(0.0)
        m2result = mat22(wptype(2.0) * input[0])
        m3result = mat33(wptype(3.0) * input[0])
        m4result = mat44(wptype(4.0) * input[0])
        m5result = mat55(wptype(5.0) * input[0])

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5result[i,j]
                idx = idx + 1

    input = wp.array(randvals([1],dtype), requires_grad=True, device=device)
    val = input.numpy()[0]
    kernel = getkernel(check_scalar_mat_constructor,suffix=dtype.__name__)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ input ], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], 2.0 * val * np.ones(2*2), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 3.0 * val * np.ones(3*3), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], 4.0 * val * np.ones(4*4), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], 5.0 * val * np.ones(5*5), tol=tol)

    if dtype in np_float_types:
        expectedgrads = [2.0]*(2*2) + [3.0]*(3*3) + [4.0]*(4*4) + [5.0]*(5*5)
        for idx in range(len(outcomponents)):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ input ], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
            tape.backward(loss=out)
            test.assertEqual(tape.gradients[input].numpy()[0],expectedgrads[idx])
            tape.zero()
    
    def check_component_mat_constructor(
        input: wp.array(dtype=wptype),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2result = mat22(
            wptype(2.0) * input[0], wptype(3.0) * input[1],
            wptype(4.0) * input[2], wptype(5.0) * input[3]
        )
        m3result = mat33(
            wptype(6.0) * input[4], wptype(7.0) * input[5], wptype(8.0) * input[6], 
            wptype(9.0) * input[7], wptype(10.0) * input[8], wptype(11.0) * input[9], 
            wptype(12.0) * input[10], wptype(13.0) * input[11], wptype(14.0) * input[12], 
        )
        m4result = mat44(
            wptype(15.0) * input[13], wptype(16.0) * input[14], wptype(17.0) * input[15], wptype(18.0) * input[16], 
            wptype(19.0) * input[17], wptype(20.0) * input[18], wptype(21.0) * input[19], wptype(22.0) * input[20], 
            wptype(23.0) * input[21], wptype(24.0) * input[22], wptype(25.0) * input[23], wptype(26.0) * input[24], 
            wptype(27.0) * input[25], wptype(28.0) * input[26], wptype(29.0) * input[27], wptype(30.0) * input[28], 
        )
        m5result = mat55(
            wptype(31.0) * input[29], wptype(32.0) * input[30], wptype(33.0) * input[31], wptype(34.0) * input[32], wptype(35.0) * input[33], 
            wptype(36.0) * input[34], wptype(37.0) * input[35], wptype(38.0) * input[36], wptype(39.0) * input[37], wptype(40.0) * input[38], 
            wptype(41.0) * input[39], wptype(42.0) * input[40], wptype(43.0) * input[41], wptype(44.0) * input[42], wptype(45.0) * input[43], 
            wptype(46.0) * input[44], wptype(47.0) * input[45], wptype(48.0) * input[46], wptype(49.0) * input[47], wptype(50.0) * input[48], 
            wptype(51.0) * input[49], wptype(52.0) * input[50], wptype(53.0) * input[51], wptype(54.0) * input[52], wptype(55.0) * input[53], 
        )

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5result[i,j]
                idx = idx + 1

    input = wp.array(randvals([2*2 + 3*3 + 4*4 + 5*5],dtype), requires_grad=True, device=device)
    kernel = getkernel(check_component_mat_constructor,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[ input ], outputs=[outcomponents], device=device)
    assert_np_equal( np.array( range(2,56), dtype=dtype ) * input.numpy(), outcomponents.numpy(), tol=10*tol )

    if dtype in np_float_types:
        for idx in range(len(outcomponents)):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ input ], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedgrads = np.zeros(len(input))
            expectedgrads[idx] = idx + 2
            assert_np_equal( tape.gradients[input].numpy(),expectedgrads)
            tape.zero()

    def check_vector_mat_constructor(
        input: wp.array(dtype=wptype),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2result = mat22(
            vec2(wptype(2.0) * input[0], wptype(4.0) * input[2]),
            vec2(wptype(3.0) * input[1], wptype(5.0) * input[3])
        )
        m3result = mat33(
            vec3(wptype(6.0) * input[4], wptype(9.0) * input[7], wptype(12.0) * input[10]), 
            vec3(wptype(7.0) * input[5], wptype(10.0) * input[8], wptype(13.0) * input[11]), 
            vec3(wptype(8.0) * input[6], wptype(11.0) * input[9], wptype(14.0) * input[12]),
        )
        m4result = mat44(
            vec4(wptype(15.0) * input[13], wptype(19.0) * input[17], wptype(23.0) * input[21], wptype(27.0) * input[25]), 
            vec4(wptype(16.0) * input[14], wptype(20.0) * input[18], wptype(24.0) * input[22], wptype(28.0) * input[26]), 
            vec4(wptype(17.0) * input[15], wptype(21.0) * input[19], wptype(25.0) * input[23], wptype(29.0) * input[27]), 
            vec4(wptype(18.0) * input[16], wptype(22.0) * input[20], wptype(26.0) * input[24], wptype(30.0) * input[28]), 
        )
        m5result = mat55(
            vec5(wptype(31.0) * input[29], wptype(36.0) * input[34], wptype(41.0) * input[39], wptype(46.0) * input[44], wptype(51.0) * input[49]), 
            vec5(wptype(32.0) * input[30], wptype(37.0) * input[35], wptype(42.0) * input[40], wptype(47.0) * input[45], wptype(52.0) * input[50]), 
            vec5(wptype(33.0) * input[31], wptype(38.0) * input[36], wptype(43.0) * input[41], wptype(48.0) * input[46], wptype(53.0) * input[51]), 
            vec5(wptype(34.0) * input[32], wptype(39.0) * input[37], wptype(44.0) * input[42], wptype(49.0) * input[47], wptype(54.0) * input[52]), 
            vec5(wptype(35.0) * input[33], wptype(40.0) * input[38], wptype(45.0) * input[43], wptype(50.0) * input[48], wptype(55.0) * input[53]), 
        )

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5result[i,j]
                idx = idx + 1
    
    kernel = getkernel(check_vector_mat_constructor,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[ input ], outputs=[outcomponents], device=device)
    assert_np_equal( np.array( range(2,56), dtype=dtype ) * input.numpy(), outcomponents.numpy(), tol=10*tol )

    if dtype in np_float_types:
        for idx in range(len(outcomponents)):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ input ], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedgrads = np.zeros(len(input))
            expectedgrads[idx] = idx + 2
            assert_np_equal( tape.gradients[input].numpy(),expectedgrads)
            tape.zero()
    

def test_quat_constructor(test,device,dtype):

    np.random.seed(123)
    
    tol = {
        np.float16: 1.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat44 = wp.mat(shape=(4,4),type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    quat = wp.quaternion(type=wptype)
    
    output_select_kernel = get_select_kernel(wptype)

    def check_mat_quat_constructor(
        p: wp.array(dtype=vec3),
        r: wp.array(dtype=quat),
        s: wp.array(dtype=vec3),
        
        outcomponents: wp.array(dtype=wptype),
        outcomponents_alt: wp.array(dtype=wptype),
    ):
        m = mat44(p[0],r[0],s[0])

        R = wp.transpose(wp.quat_to_matrix(r[0]))
        c0 = s[0][0] * R[0]
        c1 = s[0][1] * R[1]
        c2 = s[0][2] * R[2]
        m_alt = mat44(
            vec4(c0[0],c0[1],c0[2],wptype(0.0)),
            vec4(c1[0],c1[1],c1[2],wptype(0.0)),
            vec4(c2[0],c2[1],c2[2],wptype(0.0)),
            vec4( p[0][0], p[0][1], p[0][2],wptype(1.0)),
        )

        idx = 0
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m[i,j]
                outcomponents_alt[idx] = m_alt[i,j]
                idx = idx + 1

    kernel = getkernel(check_mat_quat_constructor)

    # translation:
    p = wp.array(np.random.randn(1,3).astype(dtype), dtype=vec3, requires_grad=True, device=device)
    
    # generate a normalized quaternion for the rotation:
    r = np.random.randn(1,4)
    r /= np.linalg.norm(r)
    r = wp.array(r.astype(dtype), dtype=quat, requires_grad=True, device=device)

    # scale:
    s = wp.array(np.random.randn(1,3).astype(dtype), dtype=vec3, requires_grad=True, device=device)
    return
    # just going to generate the matrix using the constructor, then
    # more manually, and make sure the values/gradients are the same:
    outcomponents = wp.zeros(4*4, dtype=wptype, requires_grad=True, device=device)
    outcomponents_alt = wp.zeros(4*4, dtype=wptype, requires_grad=True, device=device)
    wp.launch(kernel, dim=1, inputs=[ p,r,s ], outputs=[outcomponents,outcomponents_alt], device=device)
    assert_np_equal(outcomponents.numpy(),outcomponents_alt.numpy(),tol=1.e-6)

    idx = 0
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    out_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(4):
        for j in range(4):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ p,r,s ], outputs=[outcomponents,outcomponents_alt], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents_alt,idx ], outputs=[out_alt], device=device)
            
            tape.backward(loss=out)
            p_grad = 1.0 * tape.gradients[p].numpy()[0]
            r_grad = 1.0 * tape.gradients[r].numpy()[0]
            s_grad = 1.0 * tape.gradients[s].numpy()[0]
            tape.zero()
            
            tape.backward(loss=out_alt)
            p_grad_alt = 1.0 * tape.gradients[p].numpy()[0]
            r_grad_alt = 1.0 * tape.gradients[r].numpy()[0]
            s_grad_alt = 1.0 * tape.gradients[s].numpy()[0]
            tape.zero()

            assert_np_equal(p_grad,p_grad_alt,tol=tol)
            assert_np_equal(r_grad,r_grad_alt,tol=tol)
            assert_np_equal(s_grad,s_grad_alt,tol=tol)

            idx = idx + 1

def test_indexing(test,device,dtype):
    
    np.random.seed(123)
    
    tol = {
        np.float16: 1.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_indexing(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        
        outcomponents: wp.array(dtype=wptype),
    ):
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2[0][i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3[0][i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4[0][i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5[0][i,j]
                idx = idx + 1


    kernel = getkernel(check_mat_indexing,suffix=dtype.__name__)
    m2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)
    
    wp.launch(kernel, dim=1, inputs=[ m2,m3,m4,m5 ], outputs=[outcomponents], device=device)
    
    assert_np_equal(outcomponents.numpy()[:4], m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], m3.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], m4.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], m5.numpy().reshape(-1), tol=tol)
        
    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim,input in [(2,m2),(3,m3),(4,m4),(5,m5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[ m2,m3,m4,m5 ], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim,dim),dtype=dtype)
                    expectedresult[i,j] = 1.0
                    assert_np_equal(tape.gradients[input].numpy()[0],expectedresult)
                    tape.zero()
                    idx = idx + 1

    

def test_equality(test,device,dtype):
    

    np.random.seed(123)
    
    tol = {
        np.float16: 1.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    def check_mat_equality():

        wp.expect_eq( mat22(wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0)),mat22(wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0)) )
        wp.expect_neq( mat22(wptype(1.0),wptype(2.0),wptype(3.0),-wptype(4.0)),mat22(wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0)) )

        wp.expect_eq(
            mat33(wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),wptype(5.0),wptype(6.0),wptype(7.0),wptype(8.0),wptype(9.0)),
            mat33(wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),wptype(5.0),wptype(6.0),wptype(7.0),wptype(8.0),wptype(9.0))
        )
        wp.expect_neq(
            mat33(wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),wptype(5.0),wptype(6.0),wptype(7.0),wptype(8.0),wptype(9.0)),
            mat33(wptype(1.0),wptype(2.0),wptype(3.0),-wptype(4.0),wptype(5.0),wptype(6.0),wptype(7.0),wptype(8.0),wptype(9.0))
        )
        
        wp.expect_eq(
            mat44(
                wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),
                wptype(5.0),wptype(6.0),wptype(7.0),wptype(8.0),
                wptype(9.0),wptype(10.0),wptype(11.0),wptype(12.0),
                wptype(13.0),wptype(14.0),wptype(15.0),wptype(16.0),
            ),
            mat44(
                wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),
                wptype(5.0),wptype(6.0),wptype(7.0),wptype(8.0),
                wptype(9.0),wptype(10.0),wptype(11.0),wptype(12.0),
                wptype(13.0),wptype(14.0),wptype(15.0),wptype(16.0),
            ),
        )
        
        wp.expect_neq(
            mat44(
                wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),
                wptype(5.0),wptype(6.0),wptype(7.0),wptype(8.0),
                wptype(9.0),wptype(10.0),wptype(11.0),wptype(12.0),
                wptype(13.0),wptype(14.0),wptype(15.0),wptype(16.0),
            ),
            mat44(
                -wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),
                wptype(5.0),wptype(6.0),wptype(7.0),wptype(8.0),
                wptype(9.0),wptype(10.0),wptype(11.0),wptype(12.0),
                wptype(13.0),wptype(14.0),wptype(15.0),wptype(16.0),
            ),
        )
        
        wp.expect_eq(
            mat55(
                wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),wptype(5.0),
                wptype(6.0),wptype(7.0),wptype(8.0),wptype(9.0),wptype(10.0),
                wptype(11.0),wptype(12.0),wptype(13.0),wptype(14.0),wptype(15.0),
                wptype(16.0),wptype(17.0),wptype(18.0),wptype(19.0),wptype(20.0),
                wptype(21.0),wptype(22.0),wptype(23.0),wptype(24.0),wptype(25.0),
            ),
            mat55(
                wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),wptype(5.0),
                wptype(6.0),wptype(7.0),wptype(8.0),wptype(9.0),wptype(10.0),
                wptype(11.0),wptype(12.0),wptype(13.0),wptype(14.0),wptype(15.0),
                wptype(16.0),wptype(17.0),wptype(18.0),wptype(19.0),wptype(20.0),
                wptype(21.0),wptype(22.0),wptype(23.0),wptype(24.0),wptype(25.0),
            ),
        )
        
        wp.expect_neq(
            mat55(
                wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),wptype(5.0),
                wptype(6.0),wptype(7.0),wptype(8.0),wptype(9.0),wptype(10.0),
                wptype(11.0),wptype(12.0),wptype(13.0),wptype(14.0),wptype(15.0),
                wptype(16.0),wptype(17.0),wptype(18.0),wptype(19.0),wptype(20.0),
                wptype(21.0),wptype(22.0),wptype(23.0),wptype(24.0),wptype(25.0),
            ),
            mat55(
                wptype(1.0),wptype(2.0),wptype(3.0),wptype(4.0),wptype(5.0),
                wptype(6.0),wptype(7.0),wptype(8.0),wptype(9.0),wptype(10.0),
                wptype(11.0),wptype(12.0),wptype(13.0),wptype(14.0),wptype(15.0),
                wptype(16.0),-wptype(17.0),wptype(18.0),wptype(19.0),wptype(20.0),
                wptype(21.0),wptype(22.0),wptype(23.0),wptype(24.0),wptype(25.0),
            ),
        )

    kernel = getkernel(check_mat_equality,suffix=dtype.__name__)
    wp.launch(kernel, dim=1, inputs=[], outputs=[], device=device)


def test_negation(test,device,dtype):
    

    np.random.seed(123)
    
    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_negation(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        
        outcomponents: wp.array(dtype=wptype),
    ):
        m2read = wptype(2.0) * m2[0]
        m3read = wptype(3.0) * m3[0]
        m4read = wptype(4.0) * m4[0]
        m5read = wptype(5.0) * m5[0]

        mat2 = -m2read
        mat3 = -m3read
        mat4 = -m4read
        mat5 = -m5read

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = mat2[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = mat3[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = mat4[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = mat5[i,j]
                idx = idx + 1

    kernel = getkernel(check_mat_negation,suffix=dtype.__name__)
    m2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)
    
    wp.launch(kernel, dim=1, inputs=[ m2,m3,m4,m5 ], outputs=[outcomponents], device=device)
    
    assert_np_equal(outcomponents.numpy()[:4], -2.0 * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], -3.0 * m3.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], -4.0 * m4.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], -5.0 * m5.numpy().reshape(-1), tol=tol)
    
    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2
        for dim,input in [(2,m2),(3,m3),(4,m4),(5,m5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[ m2,m3,m4,m5 ], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim,dim),dtype=dtype)
                    expectedresult[i,j] = -m
                    assert_np_equal(tape.gradients[input].numpy()[0],expectedresult)
                    tape.zero()
                    idx = idx + 1
            m = m + 1


def test_transpose(test,device,dtype):
    

    np.random.seed(123)

    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat32 = wp.mat(shape=(3,2),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_transpose(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        m32: wp.array(dtype=mat32),
        
        outcomponents: wp.array(dtype=wptype),
    ):
        m2read = wptype(2.0) * m2[0]
        m3read = wptype(3.0) * m3[0]
        m4read = wptype(4.0) * m4[0]
        m5read = wptype(5.0) * m5[0]
        m32read = wptype(5.0) * m32[0]

        mat2 = wp.transpose(m2read)
        mat3 = wp.transpose(m3read)
        mat4 = wp.transpose(m4read)
        mat5 = wp.transpose(m5read)
        mat32 = wp.transpose(m32read)

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = mat2[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = mat3[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = mat4[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = mat5[i,j]
                idx = idx + 1
        
        for i in range(2):
            for j in range(3):
                outcomponents[idx] = mat32[i,j]
                idx = idx + 1

    kernel = getkernel(check_mat_transpose,suffix=dtype.__name__)
    m2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    m32 = wp.array(randvals([1,3,2],dtype), dtype=mat32, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5 + 2*3, dtype=wptype, requires_grad=True, device=device)
    
    wp.launch(kernel, dim=1, inputs=[ m2,m3,m4,m5,m32 ], outputs=[outcomponents], device=device)
    
    assert_np_equal(outcomponents.numpy()[:4], 2.0 * m2.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 3.0 * m3.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], 4.0 * m4.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], 5.0 * m5.numpy()[0].T.reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[54:], 5.0 * m32.numpy()[0].T.reshape(-1), tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2
        for dim,input in [(2,m2),(3,m3),(4,m4),(5,m5)]:
            for i in range(input.dtype._shape_[0]):
                for j in range(input.dtype._shape_[1]):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[ m2,m3,m4,m5,m32 ], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    expectedresult = np.zeros((input.dtype._shape_[1],input.dtype._shape_[0]),dtype=dtype)
                    expectedresult[j,i] = m
                    assert_np_equal(tape.gradients[input].numpy()[0],expectedresult)
                    tape.zero()
                    idx = idx + 1
            m = m + 1

def test_scalar_multiplication(test,device,dtype):
    

    np.random.seed(123)

    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_scalar_mul(
        s: wp.array(dtype=wptype),
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
        outcomponents_rightmul: wp.array(dtype=wptype),
    ):
        m2load = wptype(2.0) * m2[0]
        m3load = wptype(3.0) * m3[0]
        m4load = wptype(4.0) * m4[0]
        m5load = wptype(5.0) * m5[0]

        sload = wptype(6.0) * s[0]

        m2result = sload * m2load
        m3result = sload * m3load
        m4result = sload * m4load
        m5result = sload * m5load

        m2resultright = m2load * sload
        m3resultright = m3load * sload
        m4resultright = m4load * sload
        m5resultright = m5load * sload

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i,j]
                outcomponents_rightmul[idx] = m2resultright[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i,j]
                outcomponents_rightmul[idx] = m3resultright[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i,j]
                outcomponents_rightmul[idx] = m4resultright[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5result[i,j]
                outcomponents_rightmul[idx] = m5resultright[i,j]
                idx = idx + 1
    
    kernel = getkernel(check_mat_scalar_mul,suffix=dtype.__name__)
    s = wp.array(randvals([1],dtype), requires_grad=True, device=device)
    m2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)
    outcomponents_rightmul = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ s,m2,m3,m4,m5 ], outputs=[outcomponents,outcomponents_rightmul], device=device)
    
    sval = s.numpy()[0]
    assert_np_equal(outcomponents.numpy()[:4], 2.0 * 6.0 * sval * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 3.0 * 6.0 * sval * m3.numpy().reshape(-1), tol=10*tol)
    assert_np_equal(outcomponents.numpy()[13:29], 4.0 * 6.0 * sval * m4.numpy().reshape(-1), tol=10*tol)
    assert_np_equal(outcomponents.numpy()[29:54], 5.0 * 6.0 * sval * m5.numpy().reshape(-1), tol=10*tol)

    assert_np_equal(outcomponents_rightmul.numpy()[:4], 2.0 * 6.0 * sval * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents_rightmul.numpy()[4:13], 3.0 * 6.0 * sval * m3.numpy().reshape(-1), tol=10*tol)
    assert_np_equal(outcomponents_rightmul.numpy()[13:29], 4.0 * 6.0 * sval * m4.numpy().reshape(-1), tol=10*tol)
    assert_np_equal(outcomponents_rightmul.numpy()[29:54], 5.0 * 6.0 * sval * m5.numpy().reshape(-1), tol=10*tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2
        for dim,input in [(2,m2),(3,m3),(4,m4),(5,m5)]:
            for i in range(dim):
                for j in range(dim):

                    # test left mul gradient:
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[ s,m2,m3,m4,m5 ], outputs=[outcomponents,outcomponents_rightmul], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim,dim),dtype=dtype)
                    expectedresult[i,j] = m * 6 * sval
                    assert_np_equal(tape.gradients[input].numpy()[0],expectedresult,tol=10*tol)
                    assert_np_equal(tape.gradients[s].numpy()[0],m * 6 * input.numpy()[0,i,j],tol=10*tol)
                    tape.zero()

                    # test right mul gradient:
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[ s,m2,m3,m4,m5 ], outputs=[outcomponents,outcomponents_rightmul], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents_rightmul,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim,dim),dtype=dtype)
                    expectedresult[i,j] = m * 6 * sval
                    assert_np_equal(tape.gradients[input].numpy()[0],expectedresult,tol=10*tol)
                    assert_np_equal(tape.gradients[s].numpy()[0],m * 6 * input.numpy()[0,i,j],tol=10*tol)
                    tape.zero()

                    idx = idx + 1
            m = m + 1


def test_matvec_multiplication(test,device,dtype):
    

    np.random.seed(123)

    tol = {
        np.float16: 2.e-2,
        np.float32: 5.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat32 = wp.mat(shape=(3,2),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)
    
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_vec_mul(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v32: wp.array(dtype=vec2),
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        m32: wp.array(dtype=mat32),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2load = wptype(2.0) * m2[0]
        m3load = wptype(3.0) * m3[0]
        m4load = wptype(2.0) * m4[0]
        m5load = wptype(3.0) * m5[0]
        m32load = wptype(2.0) * m32[0]

        v2load = v2[0] * wptype(3.0)
        v3load = v3[0] * wptype(2.0)
        v4load = v4[0] * wptype(3.0)
        v5load = v5[0] * wptype(2.0)
        v32load = v32[0] * wptype(3.0)

        v2result = m2load * v2load
        v3result = m3load * v3load
        v4result = m4load * v4load
        v5result = m5load * v5load
        v32result = m32load * v32load

        idx = 0
        
        for i in range(2):
            outcomponents[idx] = v2result[i]
            idx = idx + 1
        
        for i in range(3):
            outcomponents[idx] = v3result[i]
            idx = idx + 1
        
        for i in range(4):
            outcomponents[idx] = v4result[i]
            idx = idx + 1
        
        for i in range(5):
            outcomponents[idx] = v5result[i]
            idx = idx + 1

        for i in range(3):
            outcomponents[idx] = v32result[i]
            idx = idx + 1
        
    kernel = getkernel(check_mat_vec_mul,suffix=dtype.__name__)
    v2 = wp.array(randvals([1,2],dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3],dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4],dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals([1,5],dtype), dtype=vec5, requires_grad=True, device=device)
    v32 = wp.array(randvals([1,2],dtype), dtype=vec2, requires_grad=True, device=device)
    m2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    m32 = wp.array(randvals([1,3,2],dtype), dtype=mat32, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 + 3 + 4 + 5 + 3, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ v2,v3,v4,v5,v32,m2,m3,m4,m5,m32 ], outputs=[outcomponents], device=device)
    
    assert_np_equal(outcomponents.numpy()[:2], (2.0 * 3.0 ) * np.matmul(m2.numpy()[0],v2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[2:5], (3.0 * 2.0 ) * np.matmul(m3.numpy()[0],v3.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[5:9], (2.0 * 3.0 ) * np.matmul(m4.numpy()[0],v4.numpy()[0]), tol=5*tol)
    assert_np_equal(outcomponents.numpy()[9:14], (3.0 * 2.0 ) * np.matmul(m5.numpy()[0],v5.numpy()[0]), tol=5*tol)
    assert_np_equal(outcomponents.numpy()[14:17], (2.0 * 3.0 ) * np.matmul(m32.numpy()[0],v32.numpy()[0]), tol=5*tol)
    
    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2
        for dim,invec,inmat in [(2,v2,m2),(3,v3,m3),(4,v4,m4),(5,v5,m5),(3,v32,m32)]:
            for i in range(dim):

                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[ v2,v3,v4,v5,v32,m2,m3,m4,m5,m32 ], outputs=[outcomponents], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                tape.backward(loss=out)
                
                assert_np_equal(tape.gradients[invec].numpy()[0],6*inmat.numpy()[0,i,:], tol=2*tol)
                expectedresult = np.zeros(inmat.dtype._shape_,dtype=dtype)
                expectedresult[i,:] = 6*invec.numpy()[0]
                assert_np_equal(tape.gradients[inmat].numpy()[0], expectedresult, tol=2*tol)
                
                tape.zero()

                idx = idx + 1
            m = m + 1


def test_matmat_multiplication(test,device,dtype):
    

    np.random.seed(123)

    tol = {
        np.float16: 2.e-2,
        np.float32: 5.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat32 = wp.mat(shape=(3,2),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_mat_mul(
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        v32: wp.array(dtype=mat32),
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        m32: wp.array(dtype=mat32),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2load = wptype(2.0) * m2[0]
        m3load = wptype(3.0) * m3[0]
        m4load = wptype(2.0) * m4[0]
        m5load = wptype(3.0) * m5[0]
        m32load = wptype(2.0) * m32[0]

        v2load = v2[0] * wptype(3.0)
        v3load = v3[0] * wptype(2.0)
        v4load = v4[0] * wptype(3.0)
        v5load = v5[0] * wptype(2.0)
        v32load = v32[0] * wptype(3.0)

        m2result = m2load * v2load
        m3result = m3load * v3load
        m4result = m4load * v4load
        m5result = m5load * v5load
        m32result = m32load * v2load
        m32result2 = m3load * v32load

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(2):
                outcomponents[idx] = m32result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(2):
                outcomponents[idx] = m32result2[i,j]
                idx = idx + 1
        
    kernel = getkernel(check_mat_mat_mul,suffix=dtype.__name__)
    v2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    v32 = wp.array(randvals([1,3,2],dtype), dtype=mat32, requires_grad=True, device=device)
    m2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    m32 = wp.array(randvals([1,3,2],dtype), dtype=mat32, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5 + 3*2 + 3*2, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ v2,v3,v4,v5,v32,m2,m3,m4,m5,m32 ], outputs=[outcomponents], device=device)
    
    assert_np_equal(outcomponents.numpy()[:4], np.matmul(2.0 * m2.numpy()[0],3.0 * v2.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], np.matmul(3.0 * m3.numpy()[0],2.0 * v3.numpy()[0]), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], np.matmul(2.0 * m4.numpy()[0],3.0 * v4.numpy()[0]), tol=2*tol)
    assert_np_equal(outcomponents.numpy()[29:54], np.matmul(3.0 * m5.numpy()[0],2.0 * v5.numpy()[0]), tol=10*tol)
    assert_np_equal(outcomponents.numpy()[54:60], np.matmul(2.0 * m32.numpy()[0],3.0 * v2.numpy()[0]), tol=5*tol)
    assert_np_equal(outcomponents.numpy()[60:], np.matmul(3.0 * m3.numpy()[0],3.0 * v32.numpy()[0]), tol=5*tol)
    
    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for v,m,fct in [(v2,m2,2.0 * 3.0),(v3,m3,3.0 * 2.0),(v4,m4,2.0 * 3.0),(v5,m5,3.0 * 2.0),(v2,m32,2.0 * 3.0),(v32,m3,3.0 * 3.0)]:

            rows,cols = m.dtype._shape_[0],v.dtype._shape_[1]
            for i in range(rows):
                for j in range(cols):

                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[ v2,v3,v4,v5,v32,m2,m3,m4,m5,m32 ], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    
                    expected = np.zeros(v.dtype._shape_,dtype=dtype)
                    expected[:,j] = fct * m.numpy()[0,i,:]
                    assert_np_equal(tape.gradients[v].numpy()[0],expected, tol=10*tol)

                    expected = np.zeros(m.dtype._shape_,dtype=dtype)
                    expected[i,:] = fct * v.numpy()[0,:,j]
                    assert_np_equal(tape.gradients[m].numpy()[0],expected, tol=10*tol)

                    tape.zero()
                    idx = idx + 1


def test_cw_multiplication(test,device,dtype):
    

    np.random.seed(123)

    tol = {
        np.float16: 5.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat32 = wp.mat(shape=(3,2),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)
    
    output_select_kernel = get_select_kernel(wptype)

    def check_mat_cw_mul(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        s2load = wptype(1.0) * s2[0]
        s3load = wptype(2.0) * s3[0]
        s4load = wptype(3.0) * s4[0]
        s5load = wptype(4.0) * s5[0]

        v2result = np.cw_mul( v2load, s2load )
        v3result = np.cw_mul( v3load, s3load )
        v4result = np.cw_mul( v4load, s4load )
        v5result = np.cw_mul( v5load, s5load )

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = v2result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = v3result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = v4result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = v5result[i,j]
                idx = idx + 1
    
    s2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)
    
    kernel = getkernel(check_mat_cw_mul,suffix=dtype.__name__)
    wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], ( 2.0 * v2.numpy() * 1.0 * s2.numpy() ).reshape(-1), tol=50*tol)
    assert_np_equal(outcomponents.numpy()[4:13], ( 3.0 * v3.numpy() * 2.0 * s3.numpy() ).reshape(-1), tol=50*tol)
    assert_np_equal(outcomponents.numpy()[13:29], ( 4.0 * v4.numpy() * 3.0 * s4.numpy() ).reshape(-1), tol=50*tol)
    assert_np_equal(outcomponents.numpy()[29:54], ( 5.0 * v5.numpy() * 4.0 * s5.numpy() ).reshape(-1), tol=50*tol)
    
    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2.0
        for dim,in1,in2 in [(2,s2,v2),(3,s3,v3),(4,s4,v4),(5,s5,v5)]:
            for i in range(dim):
                for j in range(dim):

                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim,dim),dtype=dtype)
                    expectedresult[i,j] = m * (m-1.0) * in1.numpy()[0][i,j]
                    assert_np_equal(tape.gradients[in2].numpy()[0],expectedresult,tol=5*tol)
                    expectedresult[i,j] = m * (m-1.0) * in2.numpy()[0][i,j]
                    assert_np_equal(tape.gradients[in1].numpy()[0],expectedresult,tol=5*tol)
                    tape.zero()

                    idx = idx + 1
            m = m + 1

def test_cw_division(test,device,dtype):
    

    np.random.seed(123)


    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat32 = wp.mat(shape=(3,2),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)
    
    output_select_kernel = get_select_kernel(wptype)

    def check_mat_cw_div(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        s2load = wptype(1.0) * s2[0]
        s3load = wptype(2.0) * s3[0]
        s4load = wptype(3.0) * s4[0]
        s5load = wptype(4.0) * s5[0]

        v2result = np.cw_div( v2load, s2load )
        v3result = np.cw_div( v3load, s3load )
        v4result = np.cw_div( v4load, s4load )
        v5result = np.cw_div( v5load, s5load )

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = v2result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = v3result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = v4result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = v5result[i,j]
                idx = idx + 1
    
    s2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)
    
    kernel = getkernel(check_mat_cw_div,suffix=dtype.__name__)
    wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[outcomponents], device=device)

    if dtype in np_float_types:
        assert_np_equal(outcomponents.numpy()[:4], ( 2.0 * v2.numpy() / (1.0 * s2.numpy()) ).reshape(-1), tol=50*tol)
        assert_np_equal(outcomponents.numpy()[4:13], ( 3.0 * v3.numpy() / (2.0 * s3.numpy()) ).reshape(-1), tol=50*tol)
        assert_np_equal(outcomponents.numpy()[13:29], ( 4.0 * v4.numpy() / (3.0 * s4.numpy()) ).reshape(-1), tol=50*tol)
        assert_np_equal(outcomponents.numpy()[29:54], ( 5.0 * v5.numpy() / (4.0 * s5.numpy()) ).reshape(-1), tol=50*tol)
    else:
        assert_np_equal(outcomponents.numpy()[:4], ( 2.0 * v2.numpy() // (1.0 * s2.numpy()) ).reshape(-1), tol=50*tol)
        assert_np_equal(outcomponents.numpy()[4:13], ( 3.0 * v3.numpy() // (2.0 * s3.numpy()) ).reshape(-1), tol=50*tol)
        assert_np_equal(outcomponents.numpy()[13:29], ( 4.0 * v4.numpy() // (3.0 * s4.numpy()) ).reshape(-1), tol=50*tol)
        assert_np_equal(outcomponents.numpy()[29:54], ( 5.0 * v5.numpy() // (4.0 * s5.numpy()) ).reshape(-1), tol=50*tol)
    
    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2.0
        for dim,s,v in [(2,s2,v2),(3,s3,v3),(4,s4,v4),(5,s5,v5)]:
            for i in range(dim):
                for j in range(dim):

                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)

                    # y = v/s
                    # dy/dv = 1.0/s
                    # dy/ds = -v/s^2

                    expectedresult = np.zeros((dim,dim),dtype=dtype)
                    expectedresult[i,j] = (m / (m-1.0)) / (s.numpy()[0,i,j])
                    assert_np_equal(tape.gradients[v].numpy()[0],expectedresult,tol=50*tol)
                    expectedresult[i,j] = -(m / (m-1.0)) * v.numpy()[0,i,j]/ (s.numpy()[0,i,j]**2)
                    assert_np_equal(tape.gradients[s].numpy()[0],expectedresult,tol=abs(outcomponents.numpy()[idx]) * 50*tol)
                    tape.zero()

                    idx = idx + 1
            m = m + 1


def test_outer_product(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_outer_product(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        outcomponents: wp.array(dtype=wptype),
    ):
        s2load = wptype(2.0) * s2[0]
        s3load = wptype(3.0) * s3[0]
        s4load = wptype(4.0) * s4[0]
        s5load = wptype(5.0) * s5[0]
        
        v2load = wptype(1.0) * v2[0]
        v3load = wptype(2.0) * v3[0]
        v4load = wptype(3.0) * v4[0]
        v5load = wptype(4.0) * v5[0]

        m22result = wp.outer(s2load,v2load)
        m33result = wp.outer(s3load,v3load)
        m44result = wp.outer(s4load,v4load)
        m55result = wp.outer(s5load,v5load)
        m25result = wp.outer(s2load,v5load)

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m22result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m33result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m44result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m55result[i,j]
                idx = idx + 1
        
        for i in range(2):
            for j in range(5):
                outcomponents[idx] = m25result[i,j]
                idx = idx + 1
    
    kernel = getkernel(check_mat_outer_product,suffix=dtype.__name__)
    s2 = wp.array(randvals([1,2],dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals([1,3],dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals([1,4],dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals([1,5],dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals([1,2],dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3],dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4],dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals([1,5],dtype), dtype=vec5, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5 + 2*5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ s2,s3,s4,s5, v2,v3,v4,v5 ], outputs=[outcomponents], device=device)
    
    assert_np_equal(outcomponents.numpy()[:4], 2.0 * 1.0 * s2.numpy()[0,:,None] * v2.numpy()[0,None,:], tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 3.0 * 2.0 * s3.numpy()[0,:,None] * v3.numpy()[0,None,:], tol=10*tol)
    assert_np_equal(outcomponents.numpy()[13:29], 4.0 * 3.0 * s4.numpy()[0,:,None] * v4.numpy()[0,None,:], tol=10*tol)
    assert_np_equal(outcomponents.numpy()[29:54], 5.0 * 4.0 * s5.numpy()[0,:,None] * v5.numpy()[0,None,:], tol=10*tol)
    assert_np_equal(outcomponents.numpy()[54:], 2.0 * 4.0 * s2.numpy()[0,:,None] * v5.numpy()[0,None,:], tol=10*tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for s,v,m in [(s2,v2,2.0 * 1.0),(s3,v3,3.0 * 2.0),(s4,v4,4.0 * 3.0),(s5,v5,5.0 * 4.0),(s2,v5,2.0 * 4.0)]:
            rows = s.dtype._length_
            cols = v.dtype._length_
            for i in range(rows):
                for j in range(cols):

                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)

                    # this component's gonna be s_i * v_j, so its s gradient is gonna be nozero
                    # at the ith component and its v gradient will be nonzero at the jth component:

                    expectedresult = np.zeros((rows),dtype=dtype)
                    expectedresult[i] = m * v.numpy()[0,j]
                    assert_np_equal(tape.gradients[s].numpy()[0],expectedresult,tol=10*tol)

                    expectedresult = np.zeros((cols),dtype=dtype)
                    expectedresult[j] = m * s.numpy()[0,i]
                    assert_np_equal(tape.gradients[v].numpy()[0],expectedresult,tol=10*tol)
                    tape.zero()

                    idx = idx + 1
            m = m + 1


def test_scalar_division(test,device,dtype):
    

    np.random.seed(123)

    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat32 = wp.mat(shape=(3,2),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)
    
    output_select_kernel = get_select_kernel(wptype)

    def check_mat_scalar_div(
        s: wp.array(dtype=wptype),
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2load = wptype(2.0) * m2[0]
        m3load = wptype(3.0) * m3[0]
        m4load = wptype(4.0) * m4[0]
        m5load = wptype(5.0) * m5[0]

        sload = wptype(6.0) * s[0]

        m2result = m2load / sload
        m3result = m3load / sload
        m4result = m4load / sload
        m5result = m5load / sload

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m5result[i,j]
                idx = idx + 1
    
    kernel = getkernel(check_mat_scalar_div,suffix=dtype.__name__)
    s = wp.array(randvals([1],dtype), requires_grad=True, device=device)
    m2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ s,m2,m3,m4,m5 ], outputs=[outcomponents], device=device)
    
    sval = s.numpy()[0]
    if dtype in np_float_types:
        assert_np_equal(outcomponents.numpy()[:4], 2.0 * m2.numpy().reshape(-1) / (6.0 * sval), tol=tol)
        assert_np_equal(outcomponents.numpy()[4:13], 3.0 * m3.numpy().reshape(-1) / (6.0 * sval), tol=10*tol)
        assert_np_equal(outcomponents.numpy()[13:29], 4.0 * m4.numpy().reshape(-1) / (6.0 * sval), tol=10*tol)
        assert_np_equal(outcomponents.numpy()[29:54], 5.0 * m5.numpy().reshape(-1) / (6.0 * sval), tol=10*tol)
    else:
        assert_np_equal(outcomponents.numpy()[:4], 2.0 * m2.numpy().reshape(-1) // (6.0 * sval), tol=tol)
        assert_np_equal(outcomponents.numpy()[4:13], 3.0 * m3.numpy().reshape(-1) // (6.0 * sval), tol=10*tol)
        assert_np_equal(outcomponents.numpy()[13:29], 4.0 * m4.numpy().reshape(-1) // (6.0 * sval), tol=10*tol)
        assert_np_equal(outcomponents.numpy()[29:54], 5.0 * m5.numpy().reshape(-1) // (6.0 * sval), tol=10*tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2
        for dim,input in [(2,m2),(3,m3),(4,m4),(5,m5)]:
            for i in range(dim):
                for j in range(dim):

                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[ s,m2,m3,m4,m5 ], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim,dim),dtype=dtype)
                    expectedresult[i,j] = m / (6 * sval)
                    assert_np_equal(tape.gradients[input].numpy()[0],expectedresult,tol=10*tol)
                    assert_np_equal(tape.gradients[s].numpy()[0],-m*input.numpy()[0,i,j] / (6 * sval * sval),tol=10*tol)
                    tape.zero()

                    idx = idx + 1
            m = m + 1
    
def test_addition(test,device,dtype):
    
    np.random.seed(123)

    tol = {
        np.float16: 2.e-2,
        np.float32: 5.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.vec(length=2,type=wptype)
    vec3 = wp.vec(length=3,type=wptype)
    vec4 = wp.vec(length=4,type=wptype)
    vec5 = wp.vec(length=5,type=wptype)
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)
    def check_mat_add(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
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

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = v2result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = v3result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = v4result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = v5result[i,j]
                idx = idx + 1
    
    s2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)
    
    kernel = getkernel(check_mat_add,suffix=dtype.__name__)
    wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], ( 2.0 * v2.numpy() + 1.0 * s2.numpy() ).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], ( 3.0 * v3.numpy() + 2.0 * s3.numpy() ).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], ( 4.0 * v4.numpy() + 3.0 * s4.numpy() ).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], ( 5.0 * v5.numpy() + 4.0 * s5.numpy() ).reshape(-1), tol=tol)
    
    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2.0
        for dim,in1,in2 in [(2,s2,v2),(3,s3,v3),(4,s4,v4),(5,s5,v5)]:
            for i in range(dim):
                for j in range(dim):

                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim,dim),dtype=dtype)
                    expectedresult[i,j] = m
                    assert_np_equal(tape.gradients[in2].numpy()[0],expectedresult,tol=10*tol)
                    expectedresult[i,j] = m - 1.0
                    assert_np_equal(tape.gradients[in1].numpy()[0],expectedresult,tol=10*tol)
                    tape.zero()

                    idx = idx + 1
            m = m + 1

def test_subtraction(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)
    def check_mat_sub(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
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

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = v2result[i,j]
                idx = idx + 1
        
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = v3result[i,j]
                idx = idx + 1
        
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = v4result[i,j]
                idx = idx + 1
        
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = v5result[i,j]
                idx = idx + 1
    
    s2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2*2 + 3*3 + 4*4 + 5*5, dtype=wptype, requires_grad=True, device=device)
    
    kernel = getkernel(check_mat_sub,suffix=dtype.__name__)
    wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], ( 2.0 * v2.numpy() - 1.0 * s2.numpy() ).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], ( 3.0 * v3.numpy() - 2.0 * s3.numpy() ).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], ( 4.0 * v4.numpy() - 3.0 * s4.numpy() ).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], ( 5.0 * v5.numpy() - 4.0 * s5.numpy() ).reshape(-1), tol=10*tol)
    
    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2.0
        for dim,in1,in2 in [(2,s2,v2),(3,s3,v3),(4,s4,v4),(5,s5,v5)]:
            for i in range(dim):
                for j in range(dim):

                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim,dim),dtype=dtype)
                    expectedresult[i,j] = m
                    assert_np_equal(tape.gradients[in2].numpy()[0],expectedresult,tol=10*tol)
                    expectedresult[i,j] = 1.0 - m
                    assert_np_equal(tape.gradients[in1].numpy()[0],expectedresult,tol=10*tol)
                    tape.zero()

                    idx = idx + 1
            m = m + 1

def test_ddot(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)
    def check_mat_dot(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        dot2: wp.array(dtype=wptype),
        dot3: wp.array(dtype=wptype),
        dot4: wp.array(dtype=wptype),
        dot5: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(1.0) * v3[0]
        v4load = wptype(2.0) * v4[0]
        v5load = wptype(1.0) * v5[0]

        s2load = wptype(1.0) * s2[0]
        s3load = wptype(2.0) * s3[0]
        s4load = wptype(1.0) * s4[0]
        s5load = wptype(2.0) * s5[0]

        dot2[0] = wp.ddot(v2load,s2load)
        dot3[0] = wp.ddot(v3load,s3load)
        dot4[0] = wp.ddot(v4load,s4load)
        dot5[0] = wp.ddot(v5load,s5load)

    s2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    dot2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot5 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    kernel = getkernel(check_mat_dot,suffix=dtype.__name__)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[s2,s3,s4,s5,v2,v3,v4,v5,], outputs=[dot2,dot3,dot4,dot5], device=device)

    assert_np_equal(dot2.numpy()[0], 2 * 1 * (v2.numpy() * s2.numpy()).sum(), tol=10*tol)
    assert_np_equal(dot3.numpy()[0], 1 * 2 * (v3.numpy() * s3.numpy()).sum(), tol=10*tol)
    assert_np_equal(dot4.numpy()[0], 2 * 1 * (v4.numpy() * s4.numpy()).sum(), tol=50*tol)
    assert_np_equal(dot5.numpy()[0], 1 * 2 * (v5.numpy() * s5.numpy()).sum(), tol=200*tol)
    
    if dtype in np_float_types:
        tape.backward(loss=dot2)
        sgrads = tape.gradients[s2].numpy()[0]
        expected_grads = 2.0 * 1.0 * v2.numpy()[0]
        assert_np_equal(sgrads,expected_grads, tol=10*tol)

        vgrads = tape.gradients[v2].numpy()[0]
        expected_grads = 2.0 * 1.0 * s2.numpy()[0]
        assert_np_equal(vgrads,expected_grads, tol=10*tol)
        
        tape.zero()
        
        tape.backward(loss=dot3)
        sgrads = tape.gradients[s3].numpy()[0]
        expected_grads = 1.0 * 2.0 * v3.numpy()[0]
        assert_np_equal(sgrads,expected_grads, tol=10*tol)

        vgrads = tape.gradients[v3].numpy()[0]
        expected_grads = 1.0 * 2.0 * s3.numpy()[0]
        assert_np_equal(vgrads,expected_grads, tol=10*tol)
        
        tape.zero()
        
        tape.backward(loss=dot4)
        sgrads = tape.gradients[s4].numpy()[0]
        expected_grads = 2.0 * 1.0 * v4.numpy()[0]
        assert_np_equal(sgrads,expected_grads, tol=10*tol)

        vgrads = tape.gradients[v4].numpy()[0]
        expected_grads = 2.0 * 1.0 * s4.numpy()[0]
        assert_np_equal(vgrads,expected_grads, tol=10*tol)
        
        tape.zero()
        
        tape.backward(loss=dot5)
        sgrads = tape.gradients[s5].numpy()[0]
        expected_grads = 1.0 * 2.0 * v5.numpy()[0]
        assert_np_equal(sgrads,expected_grads, tol=10*tol)

        vgrads = tape.gradients[v5].numpy()[0]
        expected_grads = 1.0 * 2.0 * s5.numpy()[0]
        assert_np_equal(vgrads,expected_grads, tol=10*tol)
        
        tape.zero()

def test_determinant(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)
    def check_mat_det(
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        det2: wp.array(dtype=wptype),
        det3: wp.array(dtype=wptype),
        det4: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(2.0) * v4[0]

        det2[0] = wp.determinant(v2load)
        det3[0] = wp.determinant(v3load)
        det4[0] = wp.determinant(v4load)
    
    v2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    det2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    det3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    det4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    kernel = getkernel(check_mat_det,suffix=dtype.__name__)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[v2,v3,v4,], outputs=[det2,det3,det4,], device=device)

    if dtype in np_float_types:
        assert_np_equal(det2.numpy()[0], np.linalg.det( 2 * v2.numpy()[0].astype(np.float64) ), tol=100*tol)
        assert_np_equal(det3.numpy()[0], np.linalg.det( 3 * v3.numpy()[0].astype(np.float64) ), tol=100*tol)
        assert_np_equal(det4.numpy()[0], np.linalg.det( 2 * v4.numpy()[0].astype(np.float64) ), tol=420*tol)
    else:
        assert_np_equal(det2.numpy()[0], np.around(np.linalg.det( 2 * v2.numpy()[0] )).astype(int) )
        assert_np_equal(det3.numpy()[0], np.around(np.linalg.det( 3 * v3.numpy()[0] )).astype(int) )
        assert_np_equal(det4.numpy()[0], np.around(np.linalg.det( 2 * v4.numpy()[0] )).astype(int) )

    if dtype in np_float_types:
        # determinant derivative formula is annoying so finite differences?
        tape.backward(loss=det2)
        v2grads = 1.0*tape.gradients[v2].numpy()[0]
        tape.zero()

        tape.backward(loss=det3)
        v3grads = 1.0*tape.gradients[v3].numpy()[0]
        tape.zero()

        tape.backward(loss=det4)
        v4grads = 1.0*tape.gradients[v4].numpy()[0]
        tape.zero()

        # finite differences are also annoying hence the large tolerance...
        # absolute nightmare in float16 too innit...
        dx = 0.01 if dtype == np.float16 else 0.0001
        fdtol = 2.e-1 if dtype == np.float16 else 2.e-3
        for i in range(2):
            for j in range(2):
                v2test = v2.numpy()
                v2test[0,i,j] += dx
                wp.launch(kernel, dim=1, inputs=[wp.array(v2test, dtype=v2.dtype, requires_grad=True, device=device),v3,v4,], outputs=[det2,det3,det4,], device=device)
                dplus = det2.numpy()[0]
                v2test[0,i,j] -= 2.0*dx
                wp.launch(kernel, dim=1, inputs=[wp.array(v2test, dtype=v2.dtype, requires_grad=True, device=device),v3,v4,], outputs=[det2,det3,det4,], device=device)
                dminus = det2.numpy()[0]
                assert_np_equal((dplus-dminus)/(2.0*dx*dplus),v2grads[i,j]/dplus,tol=fdtol)

        for i in range(3):
            for j in range(3):
                v3test = v3.numpy()
                v3test[0,i,j] += dx
                wp.launch(kernel, dim=1, inputs=[v2,wp.array(v3test, dtype=v3.dtype, requires_grad=True, device=device),v4,], outputs=[det2,det3,det4,], device=device)
                dplus = det3.numpy()[0]
                v3test[0,i,j] -= 2.0*dx
                wp.launch(kernel, dim=1, inputs=[v2,wp.array(v3test, dtype=v3.dtype, requires_grad=True, device=device),v4,], outputs=[det2,det3,det4,], device=device)
                dminus = det3.numpy()[0]
                assert_np_equal((dplus-dminus)/(2.0*dx*dplus),v3grads[i,j]/dplus,tol=fdtol)

        for i in range(4):
            for j in range(4):
                v4test = v4.numpy()
                v4test[0,i,j] += dx
                wp.launch(kernel, dim=1, inputs=[v2,v3,wp.array(v4test, dtype=v4.dtype, requires_grad=True, device=device),], outputs=[det2,det3,det4,], device=device)
                dplus = det4.numpy()[0]
                v4test[0,i,j] -= 2.0*dx
                wp.launch(kernel, dim=1, inputs=[v2,v3,wp.array(v4test, dtype=v4.dtype, requires_grad=True, device=device),], outputs=[det2,det3,det4,], device=device)
                dminus = det4.numpy()[0]
                assert_np_equal((dplus-dminus)/(2.0*dx*dplus),v4grads[i,j]/dplus,tol=fdtol)


def test_trace(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_trace(
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        tr2: wp.array(dtype=wptype),
        tr3: wp.array(dtype=wptype),
        tr4: wp.array(dtype=wptype),
        tr5: wp.array(dtype=wptype),
    ):
        v2load = wptype(2.0) * v2[0]
        v3load = wptype(3.0) * v3[0]
        v4load = wptype(4.0) * v4[0]
        v5load = wptype(5.0) * v5[0]

        tr2[0] = wp.trace(v2load)
        tr3[0] = wp.trace(v3load)
        tr4[0] = wp.trace(v4load)
        tr5[0] = wp.trace(v5load)
    
    v2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals([1,5,5],dtype), dtype=mat55, requires_grad=True, device=device)
    tr2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tr3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tr4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tr5 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    kernel = getkernel(check_mat_trace,suffix=dtype.__name__)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[v2,v3,v4,v5,], outputs=[tr2,tr3,tr4,tr5,], device=device)

    assert_np_equal(tr2.numpy()[0], np.trace( 2.0 * v2.numpy()[0] ), tol=10*tol)
    assert_np_equal(tr3.numpy()[0], np.trace( 3.0 * v3.numpy()[0] ), tol=10*tol)
    assert_np_equal(tr4.numpy()[0], np.trace( 4.0 * v4.numpy()[0] ), tol=200*tol)
    assert_np_equal(tr4.numpy()[0], np.trace( 4.0 * v4.numpy()[0] ), tol=200*tol)
    
    if dtype in np_float_types:
        tape.backward(loss=tr2)
        vgrads = tape.gradients[v2].numpy()[0]
        assert_np_equal(vgrads,2.0 * np.eye(2), tol=10*tol)
        tape.zero()
        
        tape.backward(loss=tr3)
        vgrads = tape.gradients[v3].numpy()[0]
        assert_np_equal(vgrads,3.0 * np.eye(3), tol=10*tol)
        tape.zero()
        
        tape.backward(loss=tr4)
        vgrads = tape.gradients[v4].numpy()[0]
        assert_np_equal(vgrads,4.0 * np.eye(4), tol=10*tol)
        tape.zero()
        
        tape.backward(loss=tr5)
        vgrads = tape.gradients[v5].numpy()[0]
        assert_np_equal(vgrads,5.0 * np.eye(5), tol=10*tol)
        tape.zero()
    

def test_diag(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_diag(
        s5: wp.array(dtype=vec5),
        outcomponents: wp.array(dtype=wptype),
    ):
        s5load = wptype(5.0) * s5[0]
        m55result = wp.diag(s5load)

        idx = 0
        for i in range(5):
            for j in range(5):
                outcomponents[idx] = m55result[i,j]
                idx = idx + 1
    
    kernel = getkernel(check_mat_diag,suffix=dtype.__name__)
    s5 = wp.array(randvals([1,5],dtype), dtype=vec5, requires_grad=True, device=device)
    outcomponents = wp.zeros(5*5, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ s5 ], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy(), 5.0 * np.diag(s5.numpy()[0]), tol=tol)

    if dtype in np_float_types:
        idx = 0
        for i in range(5):
            for j in range(5):

                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[s5], outputs=[outcomponents], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                tape.backward(loss=out)
                expectedresult = np.zeros(5,dtype=dtype)
                if i == j:
                    expectedresult[i] = 5.0
                assert_np_equal(tape.gradients[s5].numpy()[0],expectedresult,tol=10*tol)
                tape.zero()

                idx = idx + 1


def test_inverse(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_inverse(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2load = wptype(2.0) * m2[0]
        m3load = wptype(3.0) * m3[0]
        m4load = wptype(4.0) * m4[0]

        m2result = wp.inverse(m2load)
        m3result = wp.inverse(m3load)
        m4result = wp.inverse(m4load)

        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = m2result[i,j]
                idx = idx + 1
                
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i,j]
                idx = idx + 1
                
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m4result[i,j]
                idx = idx + 1
    
    kernel = getkernel(check_mat_inverse,suffix=dtype.__name__)
    m2 = wp.array(randvals([1,2,2],dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)

    outcomponents = wp.zeros(2*2 + 3*3 + 4*4, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ m2,m3,m4 ], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], np.linalg.inv(2.0 * m2.numpy()[0].astype(np.float64)), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], np.linalg.inv(3.0 * m3.numpy()[0].astype(np.float64)), tol=5*tol)
    assert_np_equal(outcomponents.numpy()[13:], np.linalg.inv(4.0 * m4.numpy()[0].astype(np.float64)), tol=5*tol)
    
    if dtype in np_float_types:
        # check gradients:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        m = 2.0
        for dim,input in [(2,m2),(3,m3),(4,m4)]:
            minv = np.linalg.inv(input.numpy()[0].astype(np.float64))
            for i in range(dim):
                for j in range(dim):

                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[m2,m3,m4], outputs=[outcomponents], device=device)
                        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                    tape.backward(loss=out)
                    d = np.zeros((dim,dim))
                    d[j,i] = 1.0/m
                    assert_np_equal(tape.gradients[input].numpy()[0],-np.matmul(minv,np.matmul(d,minv)).T,tol=10*tol)
                    tape.zero()

                    idx = idx + 1
            m = m + 1

    # let's check 2x2 using different formulae just for (in)sanity's sake:
    m = m2.numpy()[0]
    
    det = (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    expected = 0.5 * np.array([
        [m[1,1],-m[0,1]],
        [-m[1,0], m[0,0]]
    ], dtype=dtype) / det
    assert_np_equal(expected, outcomponents.numpy()[:4],tol=tol)

    # 0,0 component is this:
    # 0.5 * m[1,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(0.5 * m[1,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1]), outcomponents.numpy()[0],tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2,m3,m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,0 ], outputs=[out], device=device)
        
    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(-0.5 * m[1,1] * m[1,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[0,0],tol=tol)
        assert_np_equal(0.5 * m[1,1] * m[0,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[1,0],tol=tol)
        assert_np_equal(-0.5 * m[0,1] * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[1,1],tol=tol)
        assert_np_equal(0.5 * m[1,1] * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[0,1],tol=tol)
        tape.zero()

    # 0,1 component is this:
    # -0.5 * m[0,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(-0.5 * m[0,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1]), outcomponents.numpy()[1],tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2,m3,m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,1 ], outputs=[out], device=device)
    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(0.5 * m[0,1] * m[1,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[0,0],tol=tol)
        assert_np_equal(-0.5 * m[0,1] * m[0,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[1,0],tol=tol)
        assert_np_equal(0.5 * m[0,0] * m[0,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[1,1],tol=tol)
        assert_np_equal(-0.5 * m[1,1] * m[0,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[0,1],tol=tol)
        tape.zero()

    # 1,0 component is this:
    # -0.5 * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(-0.5 * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1]), outcomponents.numpy()[2],tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2,m3,m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,2 ], outputs=[out], device=device)

    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(0.5 * m[1,1] * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[0,0],tol=tol)
        assert_np_equal(-0.5 * m[0,0] * m[1,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[1,0],tol=tol)
        assert_np_equal(0.5 * m[0,0] * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[1,1],tol=tol)
        assert_np_equal(-0.5 * m[1,0] * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[0,1],tol=tol)
        tape.zero()

    # 1,1 component is this:
    # 0.5 * m[0,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(0.5 * m[0,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1]), outcomponents.numpy()[3],tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2,m3,m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,3 ], outputs=[out], device=device)

    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(-0.5 * m[0,1] * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[0,0],tol=tol)
        assert_np_equal(0.5 * m[0,0] * m[0,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[1,0],tol=tol)
        assert_np_equal(0.5 * m[0,0] * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[0,1],tol=tol)
        assert_np_equal(-0.5 * m[0,0] * m[0,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])**2, g[1,1],tol=tol)
        tape.zero()


def test_svd(test,device,dtype):
    
    np.random.seed(123)

    tol = {
        np.float16: 1.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-6,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.vec(length=3,type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)

    def check_mat_svd(
        m3: wp.array(dtype=mat33),
        Uout: wp.array(dtype=mat33),
        sigmaout: wp.array(dtype=vec3),
        Vout: wp.array(dtype=mat33),
        outcomponents: wp.array(dtype=wptype),
    ):
        m3load = wptype(2)*m3[0]
        U = mat33()
        sigma = vec3()
        V = mat33()

        wp.svd3(m3load, U, sigma, V)

        Uout[0] = U
        sigmaout[0] = sigma
        Vout[0] = V

        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = U[i,j]
                idx = idx + 1
        
        for i in range(3):
            outcomponents[idx] = sigma[i]
            idx = idx + 1
                
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = V[i,j]
                idx = idx + 1
    
    kernel = getkernel(check_mat_svd,suffix=dtype.__name__)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)

    outcomponents = wp.zeros(2*3*3 + 3, dtype=wptype, requires_grad=True, device=device)
    Uout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)
    sigmaout = wp.zeros(1, dtype=vec3, requires_grad=True, device=device)
    Vout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ m3 ], outputs=[Uout,sigmaout,Vout,outcomponents], device=device)

    Uout_np = Uout.numpy()[0].astype(np.float64)
    sigmaout_np = np.diag(sigmaout.numpy()[0].astype(np.float64))
    Vout_np = Vout.numpy()[0].astype(np.float64)

    assert_np_equal(np.matmul(Uout_np,np.matmul(sigmaout_np,Vout_np.T)),2*m3.numpy()[0].astype(np.float64),tol=30*tol)

    if dtype == np.float16:
        # I'm not even going to bother testing the gradients for float16
        # because the rounding errors are terrible...
        return

    # check gradients:
    output_select_kernel = get_select_kernel(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(3*3+3+3*3):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[ m3 ], outputs=[Uout,sigmaout,Vout,outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
        tape.backward(out)
        m3grads = 1.0 * tape.gradients[m3].numpy()[0]

        tape.zero()

        dx = 0.0001
        fdtol = 5.e-4 if dtype == np.float64 else 2.e-2
        for ii in range(3):
            for jj in range(3):
                m3test = 1.0 * m3.numpy()
                m3test[0,ii,jj] += dx
                wp.launch(kernel, dim=1, inputs=[ wp.array(m3test,dtype=mat33,device=device) ], outputs=[Uout,sigmaout,Vout,outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m3test = 1.0 * m3.numpy()
                m3test[0,ii,jj] -= dx
                wp.launch(kernel, dim=1, inputs=[ wp.array(m3test,dtype=mat33,device=device) ], outputs=[Uout,sigmaout,Vout,outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2*dx),m3grads[ii,jj],tol=fdtol)

def test_qr(test,device,dtype):
    
    np.random.seed(123)

    tol = {
        np.float16: 2.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-6,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.vec(length=3,type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)

    def check_mat_qr(
        m3: wp.array(dtype=mat33),
        Qout: wp.array(dtype=mat33),
        Rout: wp.array(dtype=mat33),
        outcomponents: wp.array(dtype=wptype),
    ):
        m3load = wptype(2)*m3[0]
        Q = mat33()
        R = mat33()

        wp.qr3(m3load, Q,R)

        Qout[0] = Q
        Rout[0] = R

        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = Q[i,j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = R[i,j]
                idx = idx + 1
    
    kernel = getkernel(check_mat_qr,suffix=dtype.__name__)
    m3 = wp.array(randvals([1,3,3],dtype), dtype=mat33, requires_grad=True, device=device)

    outcomponents = wp.zeros(2*3*3, dtype=wptype, requires_grad=True, device=device)
    Qout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)
    Rout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ m3 ], outputs=[Qout,Rout,outcomponents], device=device)

    Qout_np = Qout.numpy()[0].astype(np.float64)
    Rout_np = Rout.numpy()[0].astype(np.float64)

    # check it's actually a q and an r:
    assert_np_equal(np.matmul(Qout_np.T,Qout_np), np.eye(3,dtype=np.float64),tol=tol)
    assert_np_equal(Rout_np[1,[0]],np.zeros(1,dtype=np.float64),tol=tol)
    assert_np_equal(Rout_np[2,[0,1]],np.zeros(2,dtype=np.float64),tol=tol)

    # check it's a factorization:
    assert_np_equal(np.matmul(Qout_np,Rout_np),2*m3.numpy()[0].astype(np.float64),tol=30*tol)
    
    if dtype == np.float16:
        # I'm not even going to bother testing the gradients for float16
        # because the rounding errors are terrible...
        return

    # check gradients:
    output_select_kernel = get_select_kernel(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(len(outcomponents)):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[ m3 ], outputs=[Qout,Rout,outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
        tape.backward(out)
        m3grads = 1.0 * tape.gradients[m3].numpy()[0]

        tape.zero()

        dx = 0.0001
        fdtol = 5.e-4 if dtype == np.float64 else 2.e-2
        for ii in range(3):
            for jj in range(3):
                m3test = 1.0 * m3.numpy()
                m3test[0,ii,jj] += dx
                wp.launch(kernel, dim=1, inputs=[ wp.array(m3test,dtype=mat33,device=device) ], outputs=[Qout,Rout,outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m3test = 1.0 * m3.numpy()
                m3test[0,ii,jj] -= dx
                wp.launch(kernel, dim=1, inputs=[ wp.array(m3test,dtype=mat33,device=device) ], outputs=[Qout,Rout,outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2*dx),m3grads[ii,jj],tol=fdtol)

def test_eig(test,device,dtype):
    
    np.random.seed(123)

    tol = {
        np.float16: 4.e-2,
        np.float32: 1.e-5,
        np.float64: 1.e-5,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.vec(length=3,type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)

    def check_mat_eig(
        m3: wp.array(dtype=mat33),
        Qout: wp.array(dtype=mat33),
        dout: wp.array(dtype=vec3),
        outcomponents: wp.array(dtype=wptype),
    ):
        m3load = wptype(2)*m3[0]
        Q = mat33()
        d = vec3()

        wp.eig3(m3load, Q,d)

        Qout[0] = Q
        dout[0] = d

        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = Q[i,j]
                idx = idx + 1

        for i in range(3):
            outcomponents[idx] = d[i]
            idx = idx + 1
    
    kernel = getkernel(check_mat_eig,suffix=dtype.__name__)
    m3_np = randvals([1,3,3],dtype)
    m3_np[0] += m3_np[0].T
    m3 = wp.array(m3_np, dtype=mat33, requires_grad=True, device=device)

    outcomponents = wp.zeros(2*3*3, dtype=wptype, requires_grad=True, device=device)
    Qout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)
    dout = wp.zeros(1, dtype=vec3, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ m3 ], outputs=[Qout,dout,outcomponents], device=device)

    Qout_np = Qout.numpy()[0].astype(np.float64)
    dout_np = dout.numpy()[0].astype(np.float64)
    Dout_np = np.diag(dout_np)
    
    # check Q is orthogonal:
    assert_np_equal(np.matmul(Qout_np.T,Qout_np), np.eye(3),tol=tol)

    # check Q contains eigenvectors:
    assert_np_equal(np.matmul(Qout_np,np.matmul(Dout_np,Qout_np.T)), 2*m3_np,tol=tol)

    if dtype == np.float16:
        # I'm not even going to bother testing the gradients for float16
        # because the rounding errors are terrible...
        return

    # check gradients:
    output_select_kernel = get_select_kernel(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(len(outcomponents)):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[ m3 ], outputs=[Qout,dout,outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
        tape.backward(out)
        m3grads = 1.0 * tape.gradients[m3].numpy()[0]

        tape.zero()

        dx = 0.0001
        fdtol = 5.e-4 if dtype == np.float64 else 2.e-2
        for ii in range(3):
            for jj in range(3):
                m3test = 1.0 * m3.numpy()
                m3test[0,ii,jj] += dx
                wp.launch(kernel, dim=1, inputs=[ wp.array(m3test,dtype=mat33,device=device) ], outputs=[Qout,dout,outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m3test = 1.0 * m3.numpy()
                m3test[0,ii,jj] -= dx
                wp.launch(kernel, dim=1, inputs=[ wp.array(m3test,dtype=mat33,device=device) ], outputs=[Qout,dout,outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2*dx),m3grads[ii,jj],tol=fdtol)

def test_skew(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_skew(
        v3: wp.array(dtype=vec3),
        outcomponents: wp.array(dtype=wptype),
    ):
        v3load = wptype(3.0) * v3[0]

        m3result = wp.skew(v3load)

        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = m3result[i,j]
                idx = idx + 1
    
    kernel = getkernel(check_mat_skew,suffix=dtype.__name__)
    v3 = wp.array(randvals([1,3],dtype), dtype=vec3, requires_grad=True, device=device)

    outcomponents = wp.zeros(3*3, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ v3 ], outputs=[outcomponents], device=device)

    # make sure it gives you a cross product matrix:
    crossprodmat = outcomponents.numpy().reshape(3,3)
    v = np.array([1,0,0])
    assert_np_equal(np.matmul(crossprodmat,np.array([1,0,0])).reshape(-1),3 * np.cross(v3.numpy()[0],np.array([1,0,0])),tol=tol)
    assert_np_equal(np.matmul(crossprodmat,np.array([0,1,0])).reshape(-1),3 * np.cross(v3.numpy()[0],np.array([0,1,0])),tol=tol)
    assert_np_equal(np.matmul(crossprodmat,np.array([0,0,1])).reshape(-1),3 * np.cross(v3.numpy()[0],np.array([0,0,1])),tol=tol)
    
    # check it another way:
    x0 = v3.numpy()[0,0]
    x1 = v3.numpy()[0,1]
    x2 = v3.numpy()[0,2]
    crossprodmat_expected = np.array([
        [0,-x2, x1],
        [x2, 0,-x0],
        [-x1,x0, 0],
    ],dtype=dtype)
    assert_np_equal(crossprodmat,3*crossprodmat_expected, tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        
        for i in range(3):
            for j in range(3):

                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[v3], outputs=[outcomponents], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,idx ], outputs=[out], device=device)
                tape.backward(loss=out)
                if i == j:
                    assert_np_equal(tape.gradients[v3].numpy()[0],np.zeros(3))
                elif [i,j] == [0,1]:
                    assert_np_equal(tape.gradients[v3].numpy()[0],np.array([0,0,-3]))
                elif [i,j] == [1,0]:
                    assert_np_equal(tape.gradients[v3].numpy()[0],np.array([0,0,3]))
                elif [i,j] == [0,2]:
                    assert_np_equal(tape.gradients[v3].numpy()[0],np.array([0,3,0]))
                elif [i,j] == [2,0]:
                    assert_np_equal(tape.gradients[v3].numpy()[0],np.array([0,-3,0]))
                elif [i,j] == [1,2]:
                    assert_np_equal(tape.gradients[v3].numpy()[0],np.array([-3,0,0]))
                elif [i,j] == [2,1]:
                    assert_np_equal(tape.gradients[v3].numpy()[0],np.array([3,0,0]))
                tape.zero()

                idx = idx + 1

def test_transform_point(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_transform_point(
        v3: wp.array(dtype=vec3),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        v3load = wptype(3.0) * v3[0]
        m4load = wptype(1.0) * m4[0]

        presult = wp.transform_point(m4load,v3load)

        outcomponents[0] = presult[0]
        outcomponents[1] = presult[1]
        outcomponents[2] = presult[2]
    
    kernel = getkernel(check_mat_transform_point,suffix=dtype.__name__)
    v3 = wp.array(randvals([1,3],dtype), dtype=vec3, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)

    outcomponents = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ v3,m4 ], outputs=[outcomponents], device=device)

    v3homog = np.ones(4,dtype=dtype)
    v3homog[:3] = 3*v3.numpy()[0]
    assert_np_equal(outcomponents.numpy(),np.matmul(1.0*m4.numpy()[0], v3homog )[:3],tol=10*tol)

    if dtype in np_float_types:
        for j in range(3):

            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[v3,m4], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,j ], outputs=[out], device=device)
            tape.backward(loss=out)

            assert_np_equal(3*1.0 * m4.numpy()[0,j,:3], tape.gradients[v3].numpy(), tol=tol)
            expected = np.zeros((4,4),dtype=dtype)
            expected[j,:3] = 3*1.0*v3.numpy()
            expected[j,3] = 1.0
            assert_np_equal(tape.gradients[m4].numpy(),expected, tol=tol)

            tape.zero()

def test_transform_vector(test,device,dtype):
    

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
    mat22 = wp.mat(shape=(2,2),type=wptype)
    mat33 = wp.mat(shape=(3,3),type=wptype)
    mat44 = wp.mat(shape=(4,4),type=wptype)
    mat55 = wp.mat(shape=(5,5),type=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_transform_vector(
        v3: wp.array(dtype=vec3),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        v3load = wptype(3.0) * v3[0]
        m4load = wptype(1.0) * m4[0]

        presult = wp.transform_vector(m4load,v3load)

        outcomponents[0] = presult[0]
        outcomponents[1] = presult[1]
        outcomponents[2] = presult[2]
    
    kernel = getkernel(check_mat_transform_vector,suffix=dtype.__name__)
    v3 = wp.array(randvals([1,3],dtype), dtype=vec3, requires_grad=True, device=device)
    m4 = wp.array(randvals([1,4,4],dtype), dtype=mat44, requires_grad=True, device=device)

    outcomponents = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[ v3,m4 ], outputs=[outcomponents], device=device)

    v3homog = np.zeros(4,dtype=dtype)
    v3homog[:3] = 3*v3.numpy()[0]
    assert_np_equal(outcomponents.numpy(),np.matmul(1.0*m4.numpy()[0], v3homog )[:3],tol=10*tol)

    if dtype in np_float_types:
        for j in range(3):

            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[v3,m4], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outcomponents,j ], outputs=[out], device=device)
            tape.backward(loss=out)

            assert_np_equal(3*1.0 * m4.numpy()[0,j,:3], tape.gradients[v3].numpy(), tol=tol)
            expected = np.zeros((4,4),dtype=dtype)
            expected[j,:3] = 3*1.0*v3.numpy()
            assert_np_equal(tape.gradients[m4].numpy(),expected, tol=tol)

            tape.zero()


def register(parent):

    devices = wp.get_devices()

    class TestMat(parent):
        pass
    
    for dtype in np_signed_int_types + np_float_types:
        add_function_test(TestMat, f"test_negation_{dtype.__name__}", test_negation, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_subtraction_{dtype.__name__}", test_subtraction, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_determinant_{dtype.__name__}", test_determinant, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_skew_{dtype.__name__}", test_skew, devices=devices, dtype=dtype)

    for dtype in np_scalar_types:
        add_function_test(TestMat, f"test_arrays_{dtype.__name__}", test_arrays, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_constructors_{dtype.__name__}", test_constructors, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_indexing_{dtype.__name__}", test_indexing, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_equality_{dtype.__name__}", test_equality, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_scalar_multiplication_{dtype.__name__}", test_scalar_multiplication, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_matvec_multiplication_{dtype.__name__}", test_matvec_multiplication, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_matmat_multiplication_{dtype.__name__}", test_matmat_multiplication, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_cw_multiplication_{dtype.__name__}", test_cw_multiplication, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_cw_division_{dtype.__name__}", test_cw_division, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_outer_product_{dtype.__name__}", test_outer_product, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_transpose_{dtype.__name__}", test_transpose, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_scalar_division_{dtype.__name__}", test_scalar_division, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_addition_{dtype.__name__}", test_addition, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_ddot_{dtype.__name__}", test_ddot, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_trace_{dtype.__name__}", test_trace, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_diag_{dtype.__name__}", test_diag, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_transform_point_{dtype.__name__}", test_transform_point, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_transform_vector_{dtype.__name__}", test_transform_vector, devices=devices, dtype=dtype)
    
    for dtype in np_float_types:
        add_function_test(TestMat, f"test_quat_constructor_{dtype.__name__}", test_quat_constructor, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_inverse_{dtype.__name__}", test_inverse, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_svd_{dtype.__name__}", test_svd, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_qr_{dtype.__name__}", test_qr, devices=devices, dtype=dtype)
        add_function_test(TestMat, f"test_eig_{dtype.__name__}", test_eig, devices=devices, dtype=dtype)

    return TestMat

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
