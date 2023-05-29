"""Public API for the omni.warp.core extension"""

# Register the extension by importing its entry point class.
from .scripts.extension import OmniWarpCoreExtension
_ = OmniWarpCoreExtension
