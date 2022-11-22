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


# wrap a torch tensor to a wp array, data is not copied
def from_torch(t, dtype=warp.types.float32):

    import torch

    # ensure tensors are contiguous
    assert(t.is_contiguous())

    if (t.dtype != torch.float32 and t.dtype != torch.int32):
        raise RuntimeError("Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type")

    # if target is a vector or matrix type
    # then check if trailing dimensions match
    # the target type and update the shape
    if hasattr(dtype, "_shape_"):
        
        try:
            num_dims = len(dtype._shape_)
            type_dims = dtype._shape_
            source_dims = t.shape[-num_dims:]

            for i in range(len(type_dims)):
                if source_dims[i] != type_dims[i]:
                    raise RuntimeError()

            shape = t.shape[:-num_dims]

        except:
            raise RuntimeError(f"Could not convert source Torch tensor with shape {t.shape}, to Warp array with dtype={dtype}, ensure that trailing dimensions match ({source_dims} != {type_dims}")
    
    else:
        shape = t.shape

    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=dtype,
        shape=shape,
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
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
