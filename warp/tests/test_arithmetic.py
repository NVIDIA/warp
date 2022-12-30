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

def get_select_kernel2(dtype):
    
    def output_select_kernel2_fn(
        input: wp.array(dtype=dtype,ndim=2),
        index0: int,
        index1: int,
        out: wp.array(dtype=dtype),
    ):
        out[0] = input[index0,index1]
    
    return getkernel(output_select_kernel2_fn,suffix=dtype.__name__)

def test_arrays(test, device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 1.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    arr_np = randvals((10,5),dtype)
    arr = wp.array(arr_np, dtype=wptype, requires_grad=True, device=device)

    assert_np_equal(arr.numpy(), arr_np, tol=tol)

def test_unary_ops(test, device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_unary(
        inputs: wp.array(dtype=wptype,ndim=2),
        outputs: wp.array(dtype=wptype,ndim=2),
    ):
        for i in range(10):
            i0 = wptype(2.0) * inputs[0,i]
            i1 = wptype(3.0) * inputs[1,i]
            i2 = wptype(4.0) * inputs[2,i]
            i3 = wptype(5.0) * inputs[3,i]
            
            outputs[0,i] = -i0
            outputs[1,i] = wp.sign(i1)
            outputs[2,i] = wp.abs(i2)
            outputs[3,i] = wp.step(i3)

    if dtype in np_float_types:
        inputs = wp.array(np.random.randn(4,10).astype(dtype), dtype=wptype, requires_grad=True, device=device)
    else:
        inputs = wp.array(np.random.randint(-2,3,size=(4,10),dtype=dtype), dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(inputs)
    kernel = getkernel(check_unary,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[inputs ], outputs=[outputs], device=device)
    assert_np_equal(outputs.numpy()[0], -2 * inputs.numpy()[0], tol=tol)
    expected = np.sign(3 * inputs.numpy()[1])
    expected[expected == 0] = 1
    assert_np_equal(outputs.numpy()[1], expected, tol=tol)
    assert_np_equal(outputs.numpy()[2], np.abs(4 * inputs.numpy()[2]), tol=tol)
    assert_np_equal(outputs.numpy()[3], 1-np.heaviside(5 * inputs.numpy()[3],1), tol=tol)

    output_select_kernel = get_select_kernel2(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):

            # grad of -2x:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,0,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            expected_grads[0,i] = -2
            assert_np_equal(tape.gradients[inputs].numpy(),expected_grads, tol=tol)
            tape.zero()

            # grad of sign(3*x):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,1,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            assert_np_equal(tape.gradients[inputs].numpy(),expected_grads, tol=tol)
            tape.zero()
            
            # grad of abs(4*x):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,2,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            expected_grads[2,i] = 4*np.sign(inputs.numpy()[2,i])
            assert_np_equal(tape.gradients[inputs].numpy(),expected_grads, tol=tol)
            tape.zero()
            
            # grad of step(5*x):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,3,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            assert_np_equal(tape.gradients[inputs].numpy(),expected_grads, tol=tol)
            tape.zero()


def test_nonzero(test, device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_nonzero(
        inputs: wp.array(dtype=wptype),
        outputs: wp.array(dtype=wptype),
    ):
        for i in range(10):
            i0 = wptype(2.0) * inputs[i]
            outputs[i] = wp.nonzero(i0)

    inputs = wp.array(np.random.randint(-2,3,size=10).astype(dtype), dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(inputs)
    kernel = getkernel(check_nonzero,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[inputs ], outputs=[outputs], device=device)
    assert_np_equal(outputs.numpy(), (inputs.numpy() != 0))

    output_select_kernel = get_select_kernel(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):

            # grad should just be zero:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            assert_np_equal(tape.gradients[inputs].numpy(),expected_grads, tol=tol)
            tape.zero()

def test_binary_ops(test, device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_binary_ops(
        in1: wp.array(dtype=wptype,ndim=2),
        in2: wp.array(dtype=wptype,ndim=2),
        outputs: wp.array(dtype=wptype,ndim=2),
    ):
        for i in range(10):
            i0 = wptype(2.0) * in1[0,i]
            i1 = wptype(3.0) * in1[1,i]
            i2 = wptype(4.0) * in1[2,i]
            i3 = wptype(5.0) * in1[3,i]
            i4 = wptype(6.0) * in1[4,i]
            i5 = wptype(7.0) * in1[5,i]
            i6 = wptype(8.0) * in1[6,i]
            i7 = wptype(9.0) * in1[7,i]
            
            j0 = wptype(1.0) * in2[0,i]
            j1 = wptype(2.0) * in2[1,i]
            j2 = wptype(3.0) * in2[2,i]
            j3 = wptype(4.0) * in2[3,i]
            j4 = wptype(5.0) * in2[4,i]
            j5 = wptype(6.0) * in2[5,i]
            j6 = wptype(7.0) * in2[6,i]
            j7 = wptype(8.0) * in2[7,i]
            
            outputs[0,i] = wp.mul(i0,j0)
            outputs[1,i] = wp.div(i1,j1)
            outputs[2,i] = wp.add(i2,j2)
            outputs[3,i] = wp.sub(i3,j3)
            outputs[4,i] = wp.mod(i4,j4)
            outputs[5,i] = wp.min(i5,j5)
            outputs[6,i] = wp.max(i6,j6)
            outputs[7,i] = wp.floordiv(i7,j7)

    vals1 = randvals([8,10],dtype)
    if dtype in [np_unsigned_int_types]:
        vals2 = vals1 + randvals([8,10],dtype)
    else:
        vals2 = np.abs(randvals([8,10],dtype))
    
    in1 = wp.array(vals1, dtype=wptype, requires_grad=True, device=device)
    in2 = wp.array(vals2, dtype=wptype, requires_grad=True, device=device)

    outputs = wp.zeros_like(in1)
    kernel = getkernel(check_binary_ops,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[in1,in2], outputs=[outputs], device=device)
    
    assert_np_equal(outputs.numpy()[0],2 * in1.numpy()[0] * 1 * in2.numpy()[0], tol=tol)
    if dtype in np_float_types:
        assert_np_equal(outputs.numpy()[1],3 * in1.numpy()[1] / ( 2 * in2.numpy()[1] ), tol=tol)
    else:
        assert_np_equal(outputs.numpy()[1],3 * in1.numpy()[1] // ( 2 * in2.numpy()[1] ), tol=tol)
    assert_np_equal(outputs.numpy()[2],4 * in1.numpy()[2] + ( 3 * in2.numpy()[2] ), tol=tol)
    assert_np_equal(outputs.numpy()[3],5 * in1.numpy()[3] - ( 4 * in2.numpy()[3] ), tol=tol)
    
    # ...so this is actually the desired behaviour right? Looks like wp.mod doesn't behave like
    # python's % operator or np.mod()...
    assert_np_equal(outputs.numpy()[4], (6*in1.numpy()[4]) -(5*in2.numpy()[4]) * np.sign(6*in1.numpy()[4]) * np.floor(np.abs(6*in1.numpy()[4]) / (5*in2.numpy()[4])), tol=tol)

    assert_np_equal(outputs.numpy()[5],np.minimum(7 * in1.numpy()[5], 6 * in2.numpy()[5] ), tol=tol)
    assert_np_equal(outputs.numpy()[6],np.maximum(8 * in1.numpy()[6], 7 * in2.numpy()[6] ), tol=tol)
    assert_np_equal(outputs.numpy()[7],np.floor_divide(9 * in1.numpy()[7], 8 * in2.numpy()[7] ), tol=tol)

    output_select_kernel = get_select_kernel2(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:

        for i in range(10):

            # multiplication:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,0,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[0,i] = 2.0 * in2.numpy()[0,i]
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
            expected[0,i] = 2.0 * in1.numpy()[0,i]
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()

            # division:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,1,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[1,i] = 3.0 / ( 2.0 * in2.numpy()[1,i] )
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
            # y = x1/x2
            # dy/dx2 = -x1/x2^2
            expected[1,i] = (-3.0 / 2.0) * (in1.numpy()[1,i]/(in2.numpy()[1,i]**2))
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()

            # addition:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,2,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[2,i] = 4.0
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
            expected[2,i] = 3.0
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()

            # subtraction:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,3,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[3,i] = 5.0
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
            expected[3,i] = -4.0
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()

            # modulus. unless at discontinuities,
            # d/dx1( x1 % x2 ) == 1
            # d/dx2( x1 % x2 ) == 0
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,4,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[4,i] = 6.0
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
            expected[4,i] = 0.0
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()

            # min
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,5,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[5,i] = 7.0 if (7 * in1.numpy()[5,i] < 6 * in2.numpy()[5,i]) else 0.0
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
            expected[5,i] = 6.0 if (6 * in2.numpy()[5,i] < 7 * in1.numpy()[5,i]) else 0.0
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()

            # max
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,6,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[6,i] = 8.0 if (7 * in1.numpy()[6,i] > 7 * in2.numpy()[6,i]) else 0.0
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
            expected[6,i] = 7.0 if (6 * in2.numpy()[6,i] > 8 * in1.numpy()[6,i]) else 0.0
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()

            # floor_divide. Returns integers so gradient is zero
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,7,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[7,i] = 0.0
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
            expected[7,i] = 0.0
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()


def test_special_funcs(test, device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_special_funcs(
        inputs: wp.array(dtype=wptype,ndim=2),
        outputs: wp.array(dtype=wptype,ndim=2),
    ):
        for i in range(10):
            outputs[0,i] = wp.log(wptype(1.0/2.0) * inputs[0,i])
            outputs[1,i] = wp.log2(wptype(2.0/2.0) * inputs[1,i])
            outputs[2,i] = wp.log10(wptype(3.0/2.0) * inputs[2,i])
            outputs[3,i] = wp.exp(wptype(4.0/2.0) * inputs[3,i])
            outputs[4,i] = wp.atan(wptype(5.0/2.0) * inputs[4,i])
            outputs[5,i] = wp.sin(wptype(6.0/2.0) * inputs[5,i])
            outputs[6,i] = wp.cos(wptype(7.0/2.0) * inputs[6,i])
            outputs[7,i] = wp.sqrt(wptype(8.0/2.0) * inputs[7,i])
            outputs[8,i] = wp.tan(wptype(9.0/2.0) * inputs[8,i])
            outputs[9,i] = wp.sinh(wptype(2.0/10.0) * inputs[9,i])
            outputs[10,i] = wp.cosh(wptype(2.0/11.0) * inputs[10,i])
            outputs[11,i] = wp.tanh(wptype(12.0/2.0) * inputs[11,i])
            outputs[12,i] = wp.acos(wptype(13.0/2.0) * inputs[12,i])
            outputs[13,i] = wp.asin(wptype(14.0/2.0) * inputs[13,i])

    invals = np.random.randn(14,10).astype(dtype)
    invals[[0,1,2,7]] = np.abs(invals[[0,1,2,7]])
    invals[12] = np.clip(2.0/14.0 * invals[12],-2.0/14.0,2.0/14.0)
    invals[13] = np.clip(2.0/15.0 * invals[13],-2.0/15.0,2.0/15.0)
    inputs = wp.array(invals, dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(inputs)
    kernel = getkernel(check_special_funcs,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)

    assert_np_equal(outputs.numpy()[0],np.log(1.0/2.0 * inputs.numpy()[0]), tol=tol)
    assert_np_equal(outputs.numpy()[1],np.log2(inputs.numpy()[1]), tol=tol)
    assert_np_equal(outputs.numpy()[2],np.log10(3.0/2.0 * inputs.numpy()[2]), tol=tol)
    assert_np_equal(outputs.numpy()[3],np.exp(4.0/2.0 * inputs.numpy()[3]), tol=tol)
    assert_np_equal(outputs.numpy()[4],np.arctan(5.0/2.0 * inputs.numpy()[4]), tol=tol)
    assert_np_equal(outputs.numpy()[5],np.sin(6.0/2.0 * inputs.numpy()[5]), tol=tol)
    assert_np_equal(outputs.numpy()[6],np.cos(7.0/2.0 * inputs.numpy()[6]), tol=tol)
    assert_np_equal(outputs.numpy()[7],np.sqrt(8.0/2.0 * inputs.numpy()[7]), tol=tol)
    assert_np_equal(outputs.numpy()[8],np.tan(9.0/2.0 * inputs.numpy()[8]), tol=tol)
    assert_np_equal(outputs.numpy()[9],np.sinh(2.0/10.0 * inputs.numpy()[9]), tol=tol)
    assert_np_equal(outputs.numpy()[10],np.cosh(2.0/11.0 * inputs.numpy()[10]), tol=tol)
    assert_np_equal(outputs.numpy()[11],np.tanh(12.0/2.0 * inputs.numpy()[11]), tol=tol)
    assert_np_equal(outputs.numpy()[12],np.arccos(13.0/2.0 * inputs.numpy()[12]), tol=tol)
    assert_np_equal(outputs.numpy()[13],np.arcsin(14.0/2.0 * inputs.numpy()[13]), tol=tol)

    output_select_kernel = get_select_kernel2(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:

        for i in range(10):

            # log:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,0,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[0,i] = 1.0 / inputs.numpy()[0,i]
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # log2:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,1,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[1,i] = 1.0 / ( inputs.numpy()[1,i] * np.log(2.0) )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # log10:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,2,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[2,i] = 1.0 / ( inputs.numpy()[2,i] * np.log(10.0) )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # exp:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,3,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[3,i] = 4.0/2.0 * outputs.numpy()[3,i]
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # arctan:
            # looks like the autodiff formula in warp was wrong? Was (1 + x^2) rather than
            # 1/(1 + x^2)
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,4,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[4,i] = 2.5 / ( 6.25 * inputs.numpy()[4,i] ** 2 + 1 )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # sin:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,5,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[5,i] = np.cos( inputs.numpy()[5,i] * 3 ) * 3
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # cos:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,6,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[6,i] = -np.sin( inputs.numpy()[6,i] * 7.0/2.0 ) * 7.0/2.0
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # sqrt:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,7,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[7,i] = 1.0 / ( np.sqrt( inputs.numpy()[7,i] ) )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # tan:
            # looks like there was a bug in autodiff formula here too - gradient was zero if cos(x) > 0
            # (should have been "if(cosx != 0)")
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,8,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[8,i] = 4.5 / ( np.cos( 4.5 * inputs.numpy()[8,i] ) ** 2 )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=200*tol)
            tape.zero()

            # sinh:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,9,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[9,i] = (2.0/10.0) * np.cosh( (2.0/10.0) * inputs.numpy()[9,i] )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # cosh:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,10,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[10,i] = 2.0/11.0 * np.sinh( (2.0/11.0) * inputs.numpy()[10,i] )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # tanh:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,11,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[11,i] = 6.0 / ( np.cosh( 6 * inputs.numpy()[11,i] ) ** 2 )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # arccos:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,12,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[12,i] = -6.5 / np.sqrt( 1 - 42.25 *  inputs.numpy()[12,i] ** 2 )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=tol)
            tape.zero()

            # arcsin:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,13,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[13,i] = 7.0 / np.sqrt( 1 - 49*inputs.numpy()[13,i] ** 2 )
            assert_np_equal(tape.gradients[inputs].numpy(),expected, tol=6*tol)
            tape.zero()


def test_special_funcs_2arg(test, device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 1.e-2,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_special_funcs_2arg(
        in1: wp.array(dtype=wptype,ndim=2),
        in2: wp.array(dtype=wptype,ndim=2),
        outputs: wp.array(dtype=wptype,ndim=2),
    ):
        for i in range(10):
            outputs[0,i] = wp.pow(wptype(1.0/2.0) * in1[0,i],wptype(2.0/2.0) * in2[0,i])
            outputs[1,i] = wp.atan2(wptype(3.0/2.0) * in1[1,i],wptype(4.0/2.0) * in2[1,i])

    in1 = wp.array(np.abs(randvals([2,10],dtype)), dtype=wptype, requires_grad=True, device=device)
    in2 = wp.array(randvals([2,10],dtype), dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(in1)
    kernel = getkernel(check_special_funcs_2arg,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[in1,in2], outputs=[outputs], device=device)

    assert_np_equal(outputs.numpy()[0], np.power(1.0/2.0 * in1.numpy()[0],in2.numpy()[0]),tol=tol)
    assert_np_equal(outputs.numpy()[1], np.arctan2(3.0/2.0 * in1.numpy()[1],4.0/2.0 * in2.numpy()[1]),tol=tol)

    output_select_kernel = get_select_kernel2(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:

        for i in range(10):

            # pow:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,0,i ], outputs=[out], device=device)
            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[0,i] = np.power(2.0,-in2.numpy()[0,i]) * in2.numpy()[0,i] * np.power( in1.numpy()[0,i], in2.numpy()[0,i]-1 )
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=5*tol)
            expected[0,i] = np.power(2.0,-in2.numpy()[0,i]) * np.power(in1.numpy()[0,i],in2.numpy()[0,i]) * np.log(in1.numpy()[0,i]/2.0)
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()

            # atan2:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,1,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[1,i] = 12 * in2.numpy()[1,i] / ( 9 * in1.numpy()[1,i]**2 + 16 * in2.numpy()[1,i]**2)
            assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
            expected[1,i] = -12 * in1.numpy()[1,i] / ( 9 * in1.numpy()[1,i]**2 + 16 * in2.numpy()[1,i]**2)
            assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            tape.zero()
    

def test_float_to_int(test, device, dtype):
    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_float_to_int(
        inputs: wp.array(dtype=wptype,ndim=2),
        outputs: wp.array(dtype=wptype,ndim=2),
    ):
        for i in range(10):
            outputs[0,i] = wp.round(wptype(2.0) * inputs[0,i])
            outputs[1,i] = wp.rint(wptype(3.0) * inputs[1,i])
            outputs[2,i] = wp.trunc(wptype(4.0) * inputs[2,i])
            outputs[3,i] = wp.floor(wptype(5.0) * inputs[3,i])
            outputs[4,i] = wp.ceil(wptype(6.0) * inputs[4,i])
    
    inputs = wp.array(np.random.randn(5,10).astype(dtype), dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(inputs)
    kernel = getkernel(check_float_to_int,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)

    assert_np_equal(outputs.numpy()[0],np.round(2.0*inputs.numpy()[0]))
    assert_np_equal(outputs.numpy()[1],np.rint(3.0*inputs.numpy()[1]))
    assert_np_equal(outputs.numpy()[2],np.trunc(4.0*inputs.numpy()[2]))
    assert_np_equal(outputs.numpy()[3],np.floor(5.0*inputs.numpy()[3]))
    assert_np_equal(outputs.numpy()[4],np.ceil(6.0*inputs.numpy()[4]))

    # all the gradients should be zero as these functions are piecewise constant:
    output_select_kernel = get_select_kernel2(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(10):
        for j in range(5):

            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ inputs ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,j,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            assert_np_equal(tape.gradients[inputs].numpy(),np.zeros_like(inputs.numpy()), tol=tol)
            tape.zero()


def test_interp(test, device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 1.e-2,
        np.float32: 5.e-6,
        np.float64: 1.e-8,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_interp(
        in1: wp.array(dtype=wptype,ndim=2),
        in2: wp.array(dtype=wptype,ndim=2),
        in3: wp.array(dtype=wptype,ndim=2),
        outputs: wp.array(dtype=wptype,ndim=2),
    ):
        for i in range(10):
            outputs[0,i] = wp.smoothstep(wptype(1.5) * in1[0,i],wptype(2.0) * in2[0,i],wptype(4.0) * in3[0,i])
            outputs[1,i] = wp.lerp(wptype(1.5) * in1[1,i],wptype(2.0) * in2[1,i],wptype(4.0) * in3[1,i])

    in1 = wp.array(randvals([2,10],dtype), dtype=wptype, requires_grad=True, device=device)
    in2 = wp.array(randvals([2,10],dtype), dtype=wptype, requires_grad=True, device=device)
    in3 = wp.array(randvals([2,10],dtype), dtype=wptype, requires_grad=True, device=device)
    
    outputs = wp.zeros_like(in1)
    kernel = getkernel(check_interp,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[in1,in2,in3], outputs=[outputs], device=device)
    
    edge0 = 1.5*in1.numpy()[0]
    edge1 = 2.0*in2.numpy()[0]
    t_smoothstep = 4.0*in3.numpy()[0]
    x = np.clip((t_smoothstep - edge0) / (edge1 - edge0), 0,1)
    smoothstep_expected = x * x * (3 - 2 * x)

    assert_np_equal( outputs.numpy()[0],smoothstep_expected,tol=tol )

    a = 1.5 * in1.numpy()[1]
    b = 2.0 * in2.numpy()[1]
    t = 4.0 * in3.numpy()[1]
    assert_np_equal( outputs.numpy()[1],a*(1-t) + b*t,tol=tol )

    output_select_kernel = get_select_kernel2(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):

            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2,in3 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,0,i ], outputs=[out], device=device)
            tape.backward(loss=out)

            # e0 = 1.5*in1
            # e1 = 2.0*in2
            # t = 4.0*in3
            
            # x = clamp((t - e0) / (e1 - e0), 0,1)
            # dx/dt = 1 / (e1 - e0) if e0 < t < e1 else 0

            # y = x * x * (3 - 2 * x)

            # y = 3 * x * x - 2 * x * x * x
            # dy/dx = 6 * ( x - x^2 )
            dydx = 6 * x * ( 1 - x )
            
            # dy/in1 = dy/dx dx/de0 de0/din1
            dxde0 = (t_smoothstep - edge1) / ((edge1 - edge0)**2)
            dxde0[x==0] = 0
            dxde0[x==1] = 0

            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[0,i] = dydx[i] * 1.5 * dxde0[i]
            assert_np_equal(tape.gradients[in1].numpy(),expected_grads, tol=tol)

            # dy/in2 = dy/dx dx/de1 de1/din2
            dxde1 = (edge0 - t_smoothstep) / ((edge1 - edge0)**2)
            dxde1[x==0] = 0
            dxde1[x==1] = 0

            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[0,i] = 2 * dydx[i] * dxde1[i]
            assert_np_equal(tape.gradients[in2].numpy(),expected_grads, tol=tol)

            # dy/in3 = dy/dx dx/dt dt/din3
            dxdt = 1.0/(edge1-edge0)
            dxdt[x==0] = 0
            dxdt[x==1] = 0
            
            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[0,i] = 4 * dydx[i] * dxdt[i]
            assert_np_equal(tape.gradients[in3].numpy(),expected_grads, tol=tol)
            tape.zero()

            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2,in3 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,1,i ], outputs=[out], device=device)
            tape.backward(loss=out)

            # y = a*(1-t) + b*t
            # a = 1.5 * in1
            # b = 2.0 * in2
            # t = 4.0 * in3

            # y = 1.5 * in1*( 1 - 4.0 * in3 ) + 8 * in2*in3

            # dy/din1 = 1.5*(1-4*in3)
            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[1,i] = 1.5*(1-4*in3.numpy()[1,i])
            assert_np_equal(tape.gradients[in1].numpy(),expected_grads, tol=tol)
            
            # dy/din2 = 8*in3
            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[1,i] = 8*in3.numpy()[1,i]
            assert_np_equal(tape.gradients[in2].numpy(),expected_grads, tol=tol)
            
            # dy/din3 = 8*in2 - 1.5*4*in1
            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[1,i] = 8*in2.numpy()[1,i] - 1.5 * 4 * in1.numpy()[1,i]
            assert_np_equal(tape.gradients[in3].numpy(),expected_grads, tol=tol)
            tape.zero()

def test_clamp(test, device, dtype):

    np.random.seed(123)

    tol = {
        np.float16: 5.e-3,
        np.float32: 1.e-6,
        np.float64: 1.e-6,
    }.get(dtype,0)
    
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_clamp(
        in1: wp.array(dtype=wptype),
        in2: wp.array(dtype=wptype),
        in3: wp.array(dtype=wptype),
        outputs: wp.array(dtype=wptype),
    ):
        for i in range(100):
            outputs[i] = wp.clamp(wptype(4.0) * in1[i],wptype(2.0) * in2[i],wptype(2.0) * in3[i])

    in1 = wp.array(randvals([100],dtype), dtype=wptype, requires_grad=True, device=device)
    starts = randvals([100],dtype)
    diffs = np.abs(randvals([100],dtype))
    in2 = wp.array(starts, dtype=wptype, requires_grad=True, device=device)
    in3 = wp.array(starts + diffs, dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(in1)
    kernel = getkernel(check_clamp,suffix=dtype.__name__)

    wp.launch(kernel, dim=1, inputs=[in1,in2,in3], outputs=[outputs], device=device)

    assert_np_equal(np.clip(4*in1.numpy(),2*in2.numpy(),2*in3.numpy()),outputs.numpy(),tol=tol)

    output_select_kernel = get_select_kernel(wptype)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(100):

            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[ in1,in2,in3 ], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[ outputs,i ], outputs=[out], device=device)

            tape.backward(loss=out)
            t = 4*in1.numpy()[i]
            lower = 2*in2.numpy()[i]
            upper = 2*in3.numpy()[i]
            expected = np.zeros_like(in1.numpy())
            if t < lower:
                expected[i] = 2.0
                assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
                expected[i] = 0.0
                assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
                assert_np_equal(tape.gradients[in3].numpy(),expected, tol=tol)
            elif t > upper:
                expected[i] = 2.0
                assert_np_equal(tape.gradients[in3].numpy(),expected, tol=tol)
                expected[i] = 0.0
                assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
                assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
            else:
                expected[i] = 4.0
                assert_np_equal(tape.gradients[in1].numpy(),expected, tol=tol)
                expected[i] = 0.0
                assert_np_equal(tape.gradients[in2].numpy(),expected, tol=tol)
                assert_np_equal(tape.gradients[in3].numpy(),expected, tol=tol)

            tape.zero()


def register(parent):

    devices = wp.get_devices()

    class TestArithmetic(parent):
        pass

    # these unary ops only make sense for signed values:
    for dtype in np_signed_int_types + np_float_types:
        add_function_test(TestArithmetic, f"test_unary_ops_{dtype.__name__}", test_unary_ops, devices=devices, dtype=dtype)
    
    for dtype in np_float_types:
        add_function_test(TestArithmetic, f"test_special_funcs_{dtype.__name__}", test_special_funcs, devices=devices, dtype=dtype)
        add_function_test(TestArithmetic, f"test_special_funcs_2arg_{dtype.__name__}", test_special_funcs_2arg, devices=devices, dtype=dtype)
        add_function_test(TestArithmetic, f"test_interp_{dtype.__name__}", test_interp, devices=devices, dtype=dtype)
        add_function_test(TestArithmetic, f"test_float_to_int_{dtype.__name__}", test_float_to_int, devices=devices, dtype=dtype)

    for dtype in np_scalar_types:
        add_function_test(TestArithmetic, f"test_clamp_{dtype.__name__}", test_clamp, devices=devices, dtype=dtype)
        add_function_test(TestArithmetic, f"test_nonzero_{dtype.__name__}", test_nonzero, devices=devices, dtype=dtype)
        add_function_test(TestArithmetic, f"test_arrays_{dtype.__name__}", test_arrays, devices=devices, dtype=dtype)
        add_function_test(TestArithmetic, f"test_binary_ops_{dtype.__name__}", test_binary_ops, devices=devices, dtype=dtype)

    return TestArithmetic

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
