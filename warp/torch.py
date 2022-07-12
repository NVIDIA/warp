# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp
import torch
import numpy


# wrap a torch tensor to a wp array, data is not copied
def from_torch(t, dtype=warp.types.float32):
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
        device=t.device.type)

    # save a reference to the source tensor, otherwise it will be deallocated
    a.tensor = t
    return a


def to_torch(a):
    if a.device == "cpu":
        # Torch has an issue wrapping CPU objects 
        # that support the __array_interface__ protocol
        # in this case we need to workaround by going
        # to an ndarray first, see https://pearu.github.io/array_interface_pytorch.html
        return torch.as_tensor(numpy.asarray(a))

    elif a.device == "cuda":
        # Torch does support the __cuda_array_interface__
        # correctly, but we must be sure to maintain a reference
        # to the owning object to prevent memory allocs going out of scope
        return torch.as_tensor(a, device="cuda")
    
    else:
        raise RuntimeError("Unsupported device")



