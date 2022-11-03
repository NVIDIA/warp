"""Public API for the omni.warp extension"""

__all__ = [
    "from_omni_graph",
]

import ctypes

import warp as wp

from .scripts.extension import OmniWarpExtension

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


