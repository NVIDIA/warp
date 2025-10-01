# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import numpy as np

import warp as wp


@wp.kernel
def compute_tri_aabbs(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=1),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    v1 = pos[tri_indices[t_id * 3]]
    v2 = pos[tri_indices[t_id * 3 + 1]]
    v3 = pos[tri_indices[t_id * 3 + 2]]

    lower_bounds[t_id] = wp.min(wp.min(v1, v2), v3)
    upper_bounds[t_id] = wp.max(wp.max(v1, v2), v3)


class BvhBuild:
    params = (["sah", "median", "lbvh"], ["bunny", "bear", "rocks"])
    param_names = ["method", "asset"]

    repeat = 20
    number = 31

    def setup(self, method, asset):
        from pxr import Usd, UsdGeom

        if asset == "bear":
            self.repeat = 40

        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")

        asset_stage = Usd.Stage.Open(os.path.join(wp.examples.get_asset_directory(), f"{asset}.usd"))
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(f"/root/{asset}"))

        points_np = np.array(mesh_geom.GetPointsAttr().Get())
        indices_np = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

        points = wp.array(points_np, dtype=wp.vec3, device=self.device)
        indices = wp.array(indices_np, dtype=wp.int32, device=self.device)

        num_faces = int(indices.shape[0] / 3)

        self.lowers = wp.zeros(num_faces, dtype=wp.vec3, device=self.device)
        self.uppers = wp.zeros(num_faces, dtype=wp.vec3, device=self.device)

        wp.launch(
            dim=num_faces,
            kernel=compute_tri_aabbs,
            inputs=[points, indices],
            outputs=[self.lowers, self.uppers],
            device=self.device,
        )

        wp.synchronize_device(self.device)

    def time_build(self, method, asset):
        _bvh = wp.Bvh(self.lowers, self.uppers, constructor=method)
        wp.synchronize_device(self.device)
