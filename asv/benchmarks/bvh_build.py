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

# ruff: noqa: PLC0415

import importlib.util
import os

import numpy as np
from asv_runner.benchmarks.mark import skip_benchmark_if

import warp as wp
import warp.examples

pxr = importlib.util.find_spec("pxr")
USD_AVAILABLE = pxr is not None


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
    params = (["median", "lbvh"], ["bunny", "bear", "rocks"])
    param_names = ["method", "asset"]

    repeat = 100
    number = 5

    assets = ["bunny", "bear", "rocks"]

    def setup_cache(self):
        from pxr import Usd, UsdGeom

        wp.init()

        # Load and parse USD assets once, compute AABBs, cache as numpy arrays
        asset_data = {}
        for asset_name in self.assets:
            asset_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), f"{asset_name}.usd"))
            mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(f"/root/{asset_name}"))

            points_np = np.array(mesh_geom.GetPointsAttr().Get())
            indices_np = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

            # Compute AABBs on GPU
            with wp.ScopedDevice("cuda:0"):
                points = wp.array(points_np, dtype=wp.vec3)
                indices = wp.array(indices_np, dtype=wp.int32)
                num_faces = int(indices.shape[0] / 3)

                lowers = wp.zeros(num_faces, dtype=wp.vec3)
                uppers = wp.zeros(num_faces, dtype=wp.vec3)

                wp.launch(
                    dim=num_faces,
                    kernel=compute_tri_aabbs,
                    inputs=[points, indices],
                    outputs=[lowers, uppers],
                )

                # Convert to numpy for serialization
                asset_data[asset_name] = {
                    "lowers_np": lowers.numpy(),
                    "uppers_np": uppers.numpy(),
                }

        return asset_data

    def setup(self, asset_data, method, asset):
        self.device = wp.get_device("cuda:0")

        # Get pre-computed AABB numpy arrays and transfer to GPU
        lowers_np = asset_data[asset]["lowers_np"]
        uppers_np = asset_data[asset]["uppers_np"]

        self.lowers = wp.array(lowers_np, dtype=wp.vec3, device=self.device)
        self.uppers = wp.array(uppers_np, dtype=wp.vec3, device=self.device)
        wp.synchronize_device(self.device)

    @skip_benchmark_if(USD_AVAILABLE is False)
    def time_build(self, asset_data, method, asset):
        _bvh = wp.Bvh(self.lowers, self.uppers, constructor=method)
        wp.synchronize_device(self.device)
