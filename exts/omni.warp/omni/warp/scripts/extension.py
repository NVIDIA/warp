# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import warp as wp
import omni.ext
from .menu import WarpMenu
from .common import log_info
sample_path = os.path.join(os.path.dirname(__file__), "../../../data/scenes")

class OmniWarpExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        log_info("OmniWarpExtension startup")
        
        wp.init()
        self._menu = WarpMenu()

        omni.kit.browser.sample.register_sample_folder(
            sample_path,
            "Warp"
        )


    def on_shutdown(self):
        log_info("OmniWarpExtension shutdown")

        self._menu.shutdown()
        self._menu = None

        omni.kit.browser.sample.unregister_sample_folder(
            sample_path
        )