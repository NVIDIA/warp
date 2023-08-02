# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import omni.ext
import warp as wp


class OmniWarpCoreExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        wp.init()
