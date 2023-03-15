# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp
import numpy
import ctypes
from typing import Union


# return the warp device corresponding to a torch device
def device_from_torch(torch_device):
    return warp.get_device(str(torch_device))


# return the torch device corresponding to a warp device
def device_to_torch(wp_device):
    device = warp.get_device(wp_device)
    if device.is_cpu or device.is_primary:
        return str(device)
    elif device.is_cuda and device.is_uva:
        # it's not a primary context, but torch can access the data ptr directly thanks to UVA
        return f"cuda:{device.ordinal}"
    raise RuntimeError(f"Warp device {device} is not compatible with torch")


def dtype_from_torch(torch_dtype):

    # initialize lookup table on first call to defer torch import
    if dtype_from_torch.type_map is None:
        import torch
        dtype_from_torch.type_map = {
            torch.float64: warp.float64,
            torch.float32: warp.float32,
            torch.float16: warp.float16,
            torch.int64: warp.int64,
            torch.int32: warp.int32,
            torch.int16: warp.int16,
            torch.int8: warp.int8,
            torch.uint8: warp.uint8,
            torch.bool: warp.uint8,
            # currently unsupported by Warp
            # torch.bfloat16:
            # torch.complex64:
            # torch.complex128:
        }

    warp_dtype = dtype_from_torch.type_map.get(torch_dtype)

    if warp_dtype is not None:
        return warp_dtype
    else:
        raise TypeError(f"Invalid or unsupported data type: {torch_dtype}")

dtype_from_torch.type_map = None


def dtype_is_compatible(torch_dtype, warp_dtype):

    # initialize lookup table on first call to defer torch import
    if dtype_is_compatible.compatible_sets is None:
        import torch
        dtype_is_compatible.compatible_sets = {
            torch.float64: {warp.float64},
            # float32 can be used for scalars and vector/matrix types
            torch.float32: {warp.float32, *warp.types.warp.types.vector_types},
            torch.float16: {warp.float16},
            # allow aliasing integer tensors as signed or unsigned integer arrays
            torch.int64: {warp.int64, warp.uint64},
            torch.int32: {warp.int32, warp.uint32},
            torch.int16: {warp.int16, warp.uint16},
            torch.int8: {warp.int8, warp.uint8},
            torch.uint8: {warp.uint8, warp.int8},
            torch.bool: {warp.uint8, warp.int8},
            # currently unsupported by Warp
            # torch.bfloat16:
            # torch.complex64:
            # torch.complex128:
        }

    compatible_set = dtype_is_compatible.compatible_sets.get(torch_dtype)

    if compatible_set is not None:
        return warp_dtype in compatible_set
    else:
        raise TypeError(f"Invalid or unsupported data type: {torch_dtype}")

dtype_is_compatible.compatible_sets = None


# wrap a torch tensor to a wp array, data is not copied
def from_torch(t, dtype=None, requires_grad=None):

    if dtype is None:
        dtype = dtype_from_torch(t.dtype)
    elif not dtype_is_compatible(t.dtype, dtype):
        raise RuntimeError(f"Incompatible data types: {t.dtype} and {dtype}")

    if requires_grad is None:
        requires_grad = t.requires_grad

    # get size of underlying data type to compute strides
    ctype_size = ctypes.sizeof(dtype._type_)

    shape = tuple(t.shape)
    strides = tuple(s * ctype_size for s in t.stride())

    # if target is a vector or matrix type
    # then check if trailing dimensions match
    # the target type and update the shape
    if hasattr(dtype, "_shape_"):

        dtype_shape = dtype._shape_
        dtype_dims = len(dtype._shape_)
        if dtype_dims > len(shape) or dtype_shape != shape[-dtype_dims:]:
            raise RuntimeError(f"Could not convert Torch tensor with shape {shape} to Warp array with dtype={dtype}, ensure that source inner shape is {dtype_shape}")

        # ensure the inner strides are contiguous
        stride = 4
        for i in range(dtype_dims):
            if strides[-i - 1] != stride:
                raise RuntimeError(f"Could not convert Torch tensor with shape {shape} to Warp array with dtype={dtype}, because the source inner strides are not contiguous")
            stride *= dtype_shape[-i - 1]

        shape = tuple(shape[:-dtype_dims])
        strides = tuple(strides[:-dtype_dims])
    
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=dtype,
        shape=shape,
        strides=strides,
        copy=False,
        owner=False,
        requires_grad=requires_grad,
        device=device_from_torch(t.device))

    # save a reference to the source tensor, otherwise it will be deallocated
    a.tensor = t
    return a


def to_torch(a):

    import torch

    if a.device.is_cpu:
        # Torch has an issue wrapping CPU objects 
        # that support the __array_interface__ protocol
        # in this case we need to workaround by going
        # to an ndarray first, see https://pearu.github.io/array_interface_pytorch.html
        return torch.as_tensor(numpy.asarray(a))

    elif a.device.is_cuda:
        # Torch does support the __cuda_array_interface__
        # correctly, but we must be sure to maintain a reference
        # to the owning object to prevent memory allocs going out of scope
        return torch.as_tensor(a, device=device_to_torch(a.device))
    
    else:
        raise RuntimeError("Unsupported device")


def stream_from_torch(stream_or_device=None):

    import torch

    if isinstance(stream_or_device, torch.cuda.Stream):
        stream = stream_or_device
    else:
        # assume arg is a torch device
        stream = torch.cuda.current_stream(stream_or_device)

    device = device_from_torch(stream.device)

    warp_stream = warp.Stream(device, cuda_stream=stream.cuda_stream)

    # save a reference to the source stream, otherwise it may be destroyed
    warp_stream._torch_stream = stream

    return warp_stream


def stream_to_torch(stream_or_device=None):

    import torch

    if isinstance(stream_or_device, warp.Stream):
        stream = stream_or_device
    else:
        # assume arg is a warp device
        stream = warp.get_device(stream_or_device).stream

    device = device_to_torch(stream.device)

    torch_stream = torch.cuda.ExternalStream(stream.cuda_stream, device=device)

    # save a reference to the source stream, otherwise it may be destroyed
    torch_stream._warp_stream = stream

    return torch_stream
