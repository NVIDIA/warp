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
        data=t.data_ptr(),
        dtype=dtype,
        length=rows,
        copy=False,
        owner=False,
        requires_grad=True,
        device=t.device.type)

    # save a reference to the source tensor, otherwise it will be deallocated
    a.tensor = t
    
    return a

# wrap a wp array to a tensor, data is copied
def to_torch(a):
    
    t = torch.empty(a.shape, dtype=torch.float32, device=a.device)

    # alias as a wp array to use built-in copy
    dest = from_torch(t)
    warp.copy(dest, a)
        
    return t


