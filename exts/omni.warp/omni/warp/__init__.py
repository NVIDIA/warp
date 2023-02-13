"""Public API for the omni.warp extension"""

__all__ = [
    "from_omni_graph",
]

import warp as wp

from .scripts.extension import OmniWarpExtension

# convert an OmniGraph database attribute to a wp.array
def from_omni_graph(attr, dtype=float, device="cuda"):
    return wp.types.from_ptr(attr.memory, attr.shape[0], dtype=dtype, device=device)
