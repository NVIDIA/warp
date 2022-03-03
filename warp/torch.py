# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp
import torch

# wrap a torch tensor to a wp array, data is not copied
def from_torch(t, dtype=warp.types.float32):
    
    # ensure tensors are contiguous
    assert(t.is_contiguous())

    rows = 0

    if len(t.shape) > 1 and warp.type_length(dtype) == 1:
        rows = t.numel()
    elif len(t.shape) == 1:
        rows = t.shape[0]

    if (t.dtype != torch.float32 and t.dtype != torch.int32):
        raise RuntimeError("Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type")

    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=dtype,
        length=rows,
        copy=False,
        owner=False,
        requires_grad=True,
        device=t.device.type)

    # save a reference to the source tensor, otherwise it will be deallocated
    a.tensor = t
    
    return a

# wrap a wp array to a tensor throug CUDA array interface protocol
# note that users must maintain reference to original Warp array
# to ensure that memory is not deallocated underneath the tensor
def to_torch(a):
    
    return torch.as_tensor(a, device=a.device)



