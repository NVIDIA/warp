
import omni.ext
from .ogn import *

import warp as wp

import sys
import os

# Any class derived from `omni.ext.IExt` in a top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when the extension is enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() will be called.
class OmniWarpExtension(omni.ext.IExt):
    # ext_id is the current extension id. It can be used with the extension manager to query additional information,
    # such as where this extension is located in the filesystem.
    def on_startup(self, ext_id):
        print("[omni.warp] OmniWarpExtension startup", flush=True)

        # try and find CUDA toolchain in target-deps
        # note this is only valid for developer machines
        # end users need the CUDA_SDK installed to create nodes
        cuda_path = os.path.dirname(sys.argv[0]) + "/../../target-deps/cuda"

        if (os.path.exists(cuda_path)):
            wp.config.cuda_path = cuda_path

        wp.init()


    def on_shutdown(self):
        print("[omni.warp] OmniWarpExtension shutdown", flush=True)
