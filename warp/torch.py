import types 
import torch

# wrap a torch tensor to a wp array, data is not copied
def torch_to_wp(t, dtype=types.float32):
    
    # ensure tensors are contiguous
    assert(t.is_contiguous())

    a = types.array(
        data=t.storage().data_ptr(),
        dtype=dtype,
        length=t.shape[0],
        copy=False,
        owner=False,
        device=t.device.type)
    
    return a
