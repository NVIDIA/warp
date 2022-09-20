import ctypes

from .scripts.extension import *
from .ogn import *

# convert an OmniGraph database attribute to a wp.array
def from_omni_graph(attr, dtype=float):

    ptr_type = ctypes.POINTER(ctypes.c_size_t)
    ptr = ctypes.cast(attr.memory, ptr_type).contents.value

    return wp.types.array(
        dtype=dtype,
        length=attr.shape[0],
        capacity=attr.shape[0] * wp.types.type_size_in_bytes(dtype),
        ptr=ptr,
        device="cuda",
        owner=False,
        requires_grad=False
    )


