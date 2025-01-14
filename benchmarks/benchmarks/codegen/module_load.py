# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp


class SimModules:
    timeout = 120
    warmup_time = 0
    rounds = 4
    number = 1

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        import warp.sim.collide  # noqa: F401

    def time_warp_sim_collide_cuda(self):
        wp.load_module(module="warp.sim.collide", device="cuda:0")

    def teardown(self):
        wp.get_module("warp.sim.collide").unload()
