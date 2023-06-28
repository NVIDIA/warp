"""Public API for the omni.warp.core extension"""

# Register the extension by importing its entry point class.
from ._impl.extension import OmniWarpCoreExtension
_ = OmniWarpCoreExtension
