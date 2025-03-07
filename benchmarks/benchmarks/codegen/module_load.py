# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
