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

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy

import warp
import warp.context

if TYPE_CHECKING:
    import paddle
    from paddle.base.libpaddle import CPUPlace, CUDAPinnedPlace, CUDAPlace, Place


# return the warp device corresponding to a paddle device
def device_from_paddle(paddle_device: Place | CPUPlace | CUDAPinnedPlace | CUDAPlace | str) -> warp.context.Device:
    """Return the Warp device corresponding to a Paddle device.

    Args:
        paddle_device (`Place`, `CPUPlace`, `CUDAPinnedPlace`, `CUDAPlace`, or `str`): Paddle device identifier

    Raises:
        RuntimeError: Paddle device does not have a corresponding Warp device
    """
    if type(paddle_device) is str:
        if paddle_device.startswith("gpu:"):
            paddle_device = paddle_device.replace("gpu:", "cuda:")
        warp_device = warp.context.runtime.device_map.get(paddle_device)
        if warp_device is not None:
            return warp_device
        elif paddle_device == "gpu":
            return warp.context.runtime.get_current_cuda_device()
        else:
            raise RuntimeError(f"Unsupported Paddle device {paddle_device}")
    else:
        try:
            from paddle.base.libpaddle import CPUPlace, CUDAPinnedPlace, CUDAPlace, Place

            if isinstance(paddle_device, Place):
                if paddle_device.is_gpu_place():
                    return warp.context.runtime.cuda_devices[paddle_device.gpu_device_id()]
                elif paddle_device.is_cpu_place():
                    return warp.context.runtime.cpu_device
                else:
                    raise RuntimeError(f"Unsupported Paddle device type {paddle_device}")
            elif isinstance(paddle_device, (CPUPlace, CUDAPinnedPlace)):
                return warp.context.runtime.cpu_device
            elif isinstance(paddle_device, CUDAPlace):
                return warp.context.runtime.cuda_devices[paddle_device.get_device_id()]
            else:
                raise RuntimeError(f"Unsupported Paddle device type {paddle_device}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Please install paddlepaddle first.") from e
        except Exception as e:
            if not isinstance(paddle_device, (Place, CPUPlace, CUDAPinnedPlace, CUDAPlace)):
                raise TypeError(
                    "device_from_paddle() received an invalid argument - "
                    f"got {paddle_device}({type(paddle_device)}), but expected one of:\n"
                    "* paddle.base.libpaddle.Place\n"
                    "* paddle.CPUPlace\n"
                    "* paddle.CUDAPinnedPlace\n"
                    "* paddle.CUDAPlace or 'gpu' or 'gpu:x'(x means device id)"
                ) from e
            raise


def device_to_paddle(warp_device: warp.context.Devicelike) -> str:
    """Return the Paddle device string corresponding to a Warp device.

    Args:
        warp_device: An identifier that can be resolved to a :class:`warp.context.Device`.

    Raises:
        RuntimeError: The Warp device is not compatible with PyPaddle.
    """
    device = warp.get_device(warp_device)
    if device.is_cpu or device.is_primary:
        return str(device).replace("cuda", "gpu")
    elif device.is_cuda and device.is_uva:
        # it's not a primary context, but paddle can access the data ptr directly thanks to UVA
        return f"gpu:{device.ordinal}"
    raise RuntimeError(f"Warp device {device} is not compatible with paddle")


def dtype_to_paddle(warp_dtype):
    """Return the Paddle dtype corresponding to a Warp dtype.

    Args:
        warp_dtype: A Warp data type that has a corresponding ``paddle.dtype``.
            ``warp.uint16``, ``warp.uint32``, and ``warp.uint64`` are mapped
            to the signed integer ``paddle.dtype`` of the same width.
    Raises:
        TypeError: Unable to find a corresponding PyPaddle data type.
    """
    # initialize lookup table on first call to defer paddle import
    if dtype_to_paddle.type_map is None:
        import paddle

        dtype_to_paddle.type_map = {
            warp.float16: paddle.float16,
            warp.float32: paddle.float32,
            warp.float64: paddle.float64,
            warp.int8: paddle.int8,
            warp.int16: paddle.int16,
            warp.int32: paddle.int32,
            warp.int64: paddle.int64,
            warp.uint8: paddle.uint8,
            warp.bool: paddle.bool,
            # paddle doesn't support unsigned ints bigger than 8 bits
            warp.uint16: paddle.int16,
            warp.uint32: paddle.int32,
            warp.uint64: paddle.int64,
        }

    paddle_dtype = dtype_to_paddle.type_map.get(warp_dtype)
    if paddle_dtype is not None:
        return paddle_dtype
    else:
        raise TypeError(f"Cannot convert {warp_dtype} to a Paddle type")


def dtype_from_paddle(paddle_dtype):
    """Return the Warp dtype corresponding to a Paddle dtype.

    Args:
        paddle_dtype: A ``paddle.dtype`` that has a corresponding Warp data type.
            Currently ``paddle.bfloat16``, ``paddle.complex64``, and
            ``paddle.complex128`` are not supported.

    Raises:
        TypeError: Unable to find a corresponding Warp data type.
    """
    # initialize lookup table on first call to defer paddle import
    if dtype_from_paddle.type_map is None:
        import paddle

        dtype_from_paddle.type_map = {
            paddle.float16: warp.float16,
            paddle.float32: warp.float32,
            paddle.float64: warp.float64,
            paddle.int8: warp.int8,
            paddle.int16: warp.int16,
            paddle.int32: warp.int32,
            paddle.int64: warp.int64,
            paddle.uint8: warp.uint8,
            paddle.bool: warp.bool,
            # currently unsupported by Warp
            # paddle.bfloat16:
            # paddle.complex64:
            # paddle.complex128:
        }

    warp_dtype = dtype_from_paddle.type_map.get(paddle_dtype)

    if warp_dtype is not None:
        return warp_dtype
    else:
        raise TypeError(f"Cannot convert {paddle_dtype} to a Warp type")


def dtype_is_compatible(paddle_dtype: paddle.dtype, warp_dtype) -> bool:
    """Evaluates whether the given paddle dtype is compatible with the given Warp dtype."""
    # initialize lookup table on first call to defer paddle import
    if dtype_is_compatible.compatible_sets is None:
        import paddle

        dtype_is_compatible.compatible_sets = {
            paddle.float64: {warp.float64},
            paddle.float32: {warp.float32},
            paddle.float16: {warp.float16},
            # allow aliasing integer tensors as signed or unsigned integer arrays
            paddle.int64: {warp.int64, warp.uint64},
            paddle.int32: {warp.int32, warp.uint32},
            paddle.int16: {warp.int16, warp.uint16},
            paddle.int8: {warp.int8, warp.uint8},
            paddle.uint8: {warp.uint8, warp.int8},
            paddle.bool: {warp.bool, warp.uint8, warp.int8},
            # currently unsupported by Warp
            # paddle.bfloat16:
            # paddle.complex64:
            # paddle.complex128:
        }

    compatible_set = dtype_is_compatible.compatible_sets.get(paddle_dtype)

    if compatible_set is not None:
        if warp_dtype in compatible_set:
            return True
        # check if it's a vector or matrix type
        if hasattr(warp_dtype, "_wp_scalar_type_"):
            return warp_dtype._wp_scalar_type_ in compatible_set

    return False


# lookup tables initialized when needed
dtype_from_paddle.type_map = None
dtype_to_paddle.type_map = None
dtype_is_compatible.compatible_sets = None


# wrap a paddle tensor to a wp array, data is not copied
def from_paddle(
    t: paddle.Tensor,
    dtype: paddle.dtype | None = None,
    requires_grad: bool | None = None,
    grad: paddle.Tensor | None = None,
    return_ctype: bool = False,
) -> warp.array:
    """Convert a Paddle tensor to a Warp array without copying the data.

    Args:
        t (paddle.Tensor): The paddle tensor to wrap.
        dtype (warp.dtype, optional): The target data type of the resulting Warp array. Defaults to the tensor value type mapped to a Warp array value type.
        requires_grad (bool, optional): Whether the resulting array should wrap the tensor's gradient, if it exists (the grad tensor will be allocated otherwise). Defaults to the tensor's `requires_grad` value.
        grad (paddle.Tensor, optional): The grad attached to given tensor. Defaults to None.
        return_ctype (bool, optional): Whether to return a low-level array descriptor instead of a ``wp.array`` object (faster).  The descriptor can be passed to Warp kernels.

    Returns:
        warp.array: The wrapped array or array descriptor.
    """
    if dtype is None:
        dtype = dtype_from_paddle(t.dtype)
    elif not dtype_is_compatible(t.dtype, dtype):
        raise RuntimeError(f"Cannot convert Paddle type {t.dtype} to Warp type {dtype}")

    # get size of underlying data type to compute strides
    ctype_size = ctypes.sizeof(dtype._type_)

    shape = tuple(t.shape)
    strides = tuple(s * ctype_size for s in t.strides)

    # if target is a vector or matrix type
    # then check if trailing dimensions match
    # the target type and update the shape
    if hasattr(dtype, "_shape_"):
        dtype_shape = dtype._shape_
        dtype_dims = len(dtype._shape_)
        # ensure inner shape matches
        if dtype_dims > len(shape) or dtype_shape != shape[-dtype_dims:]:
            raise RuntimeError(
                f"Could not convert Paddle tensor with shape {shape} to Warp array with dtype={dtype}, ensure that source inner shape is {dtype_shape}"
            )
        # ensure inner strides are contiguous
        if strides[-1] != ctype_size or (dtype_dims > 1 and strides[-2] != ctype_size * dtype_shape[-1]):
            raise RuntimeError(
                f"Could not convert Paddle tensor with shape {shape} to Warp array with dtype={dtype}, because the source inner strides are not contiguous"
            )
        # trim shape and strides
        shape = tuple(shape[:-dtype_dims]) or (1,)
        strides = tuple(strides[:-dtype_dims]) or (ctype_size,)

    # gradient
    # - if return_ctype is False, we set `grad` to a wp.array or None
    # - if return_ctype is True, we set `grad_ptr` and set `grad` as the owner (wp.array or paddle.Tensor)
    requires_grad = (not t.stop_gradient) if requires_grad is None else requires_grad
    grad_ptr = 0
    if grad is not None:
        if isinstance(grad, warp.array):
            if return_ctype:
                if grad.strides != strides:
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {strides} but got {grad.strides}"
                    )
                grad_ptr = grad.ptr
        else:
            # assume grad is a paddle.Tensor
            if return_ctype:
                if t.strides != grad.strides:
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {t.strides} but got {grad.strides}"
                    )
                grad_ptr = grad.data_ptr()
            else:
                grad = from_paddle(grad, dtype=dtype, requires_grad=False)
    elif requires_grad:
        # wrap the tensor gradient, allocate if necessary
        if t.grad is not None:
            if return_ctype:
                grad = t.grad
                if t.strides != grad.strides:
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {t.strides} but got {grad.strides}"
                    )
                grad_ptr = grad.data_ptr()
            else:
                grad = from_paddle(t.grad, dtype=dtype, requires_grad=False)
        else:
            # allocate a zero-filled gradient if it doesn't exist
            # Note: we use Warp to allocate the shared gradient with compatible strides
            grad = warp.zeros(dtype=dtype, shape=shape, strides=strides, device=device_from_paddle(t.place))
            # use .grad_ for zero-copy
            t.grad_ = to_paddle(grad, requires_grad=False)
            grad_ptr = grad.ptr

    if return_ctype:
        ptr = t.data_ptr()

        # create array descriptor
        array_ctype = warp.types.array_t(ptr, grad_ptr, len(shape), shape, strides)

        # keep data and gradient alive
        array_ctype._ref = t
        array_ctype._gradref = grad

        return array_ctype

    else:
        a = warp.array(
            ptr=t.data_ptr(),
            dtype=dtype,
            shape=shape,
            strides=strides,
            device=device_from_paddle(t.place),
            copy=False,
            grad=grad,
            requires_grad=requires_grad,
        )

        # save a reference to the source tensor, otherwise it may get deallocated
        a._tensor = t

        return a


def to_paddle(a: warp.array, requires_grad: bool = None) -> paddle.Tensor:
    """
    Convert a Warp array to a Paddle tensor without copying the data.

    Args:
        a (warp.array): The Warp array to convert.
        requires_grad (bool, optional): Whether the resulting tensor should convert the array's gradient, if it exists, to a grad tensor. Defaults to the array's `requires_grad` value.

    Returns:
        paddle.Tensor: The converted tensor.
    """
    import paddle
    import paddle.utils.dlpack

    if requires_grad is None:
        requires_grad = a.requires_grad

    # Paddle does not support structured arrays
    if isinstance(a.dtype, warp.codegen.Struct):
        raise RuntimeError("Cannot convert structured Warp arrays to Paddle.")

    if a.device.is_cpu:
        # Paddle has an issue wrapping CPU objects
        # that support the __array_interface__ protocol
        # in this case we need to workaround by going
        # to an ndarray first, see https://pearu.github.io/array_interface_pypaddle.html
        t = paddle.to_tensor(numpy.asarray(a), place="cpu")
        t.stop_gradient = not requires_grad
        if requires_grad and a.requires_grad:
            # use .grad_ for zero-copy
            t.grad_ = paddle.to_tensor(numpy.asarray(a.grad), place="cpu")
        return t

    elif a.device.is_cuda:
        # Paddle does support the __cuda_array_interface__
        # correctly, but we must be sure to maintain a reference
        # to the owning object to prevent memory allocs going out of scope
        t = paddle.utils.dlpack.from_dlpack(warp.to_dlpack(a)).to(device=device_to_paddle(a.device))
        t.stop_gradient = not requires_grad
        if requires_grad and a.requires_grad:
            # use .grad_ for zero-copy
            t.grad_ = paddle.utils.dlpack.from_dlpack(warp.to_dlpack(a.grad)).to(device=device_to_paddle(a.device))
        return t

    else:
        raise RuntimeError("Unsupported device")


def stream_from_paddle(stream_or_device=None):
    """Convert from a Paddle CUDA stream to a Warp CUDA stream."""
    import paddle

    if isinstance(stream_or_device, paddle.device.Stream):
        stream = stream_or_device
    else:
        # assume arg is a paddle device
        stream = paddle.device.current_stream(stream_or_device)

    device = device_from_paddle(stream.device)

    warp_stream = warp.Stream(device, cuda_stream=stream.stream_base.cuda_stream)

    # save a reference to the source stream, otherwise it may be destroyed
    warp_stream._paddle_stream = stream

    return warp_stream
