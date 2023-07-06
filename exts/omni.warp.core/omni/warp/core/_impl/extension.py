# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
import omni.ext
import warp as wp


def log_info(msg):
    carb.log_info("[omni.warp.core] {}".format(msg))


class OmniWarpCoreExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        log_info("OmniWarpCoreExtension startup")

        wp.init()

    def on_shutdown(self):
        log_info("OmniWarpCoreExtension shutdown")
