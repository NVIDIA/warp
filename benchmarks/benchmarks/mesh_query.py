# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import numpy as np

import warp as wp

wp.set_module_options({"enable_backward": False})

NUM_QUERY_POINTS = 1000000
seed = 42


def get_asset_directory():
    return os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "..", "warp", "examples", "assets")


@wp.kernel
def sample_mesh_query_no_sign(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    query_d_max: float,
    query_closest_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p = query_points[tid]
    query = wp.mesh_query_point_no_sign(mesh, p, query_d_max)

    if query.result:
        face = query.face
        cp = wp.vec3(float(face), query.u, query.v)
        query_closest_points[tid] = cp


class MeshQuery:
    params = ["bunny", "bear", "cube", "rocks", "sphere"]
    rounds = 3
    number = 10
    timeout = 120
    warmup_time = 1.0

    def setup(self, asset):
        from pxr import Usd, UsdGeom

        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module("cuda:0")

        asset_stage = Usd.Stage.Open(os.path.join(get_asset_directory(), f"{asset}.usd"))
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(f"/root/{asset}"))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())
        bounding_box = np.array([points.min(axis=0), points.max(axis=0)])

        global seed
        rng = np.random.default_rng(seed)
        seed = seed + 1

        query_points_np = rng.uniform(bounding_box[0, :], bounding_box[1, :], size=(NUM_QUERY_POINTS, 3)).astype(
            np.float32
        )
        self.query_points = wp.array(query_points_np, dtype=wp.vec3, device="cuda:0")

        # create wp mesh
        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device="cuda:0"),
            velocities=None,
            indices=wp.array(indices, dtype=int, device="cuda:0"),
        )

        self.query_closest_points = wp.empty_like(self.query_points, device="cuda:0")

        with wp.ScopedCapture() as capture:
            wp.launch(
                sample_mesh_query_no_sign,
                dim=(NUM_QUERY_POINTS,),
                inputs=[self.mesh.id, self.query_points, 1.0e7, self.query_closest_points],
                device="cuda:0",
            )
        self.graph = capture.graph

        for _warmup in range(5):
            wp.capture_launch(self.graph)

        wp.synchronize_device("cuda:0")

    def time_mesh_query(self, asset):
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")

    time_mesh_query.param_names = ["asset"]
