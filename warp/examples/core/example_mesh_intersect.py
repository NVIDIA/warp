# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#############################################################################
# Example Mesh Intersection
#
# Show how to use built-in BVH query to test if two triangle meshes intersect.
#
##############################################################################

import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.render


@wp.func
def cw_min(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def cw_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.kernel
def intersect(
    mesh_0: wp.uint64,
    mesh_1: wp.uint64,
    num_faces: int,
    xforms: wp.array(dtype=wp.transform),
    result: wp.array(dtype=int),
):
    tid = wp.tid()

    # mesh_0 is assumed to be the query mesh, we launch one thread
    # for each face in mesh_0 and test it against the opposing mesh's BVH
    face = tid % num_faces
    batch = tid // num_faces

    # transforms from mesh_0 -> mesh_1 space
    xform = xforms[batch]

    # load query triangles points and transform to mesh_1's space
    v0 = wp.transform_point(xform, wp.mesh_eval_position(mesh_0, face, 1.0, 0.0))
    v1 = wp.transform_point(xform, wp.mesh_eval_position(mesh_0, face, 0.0, 1.0))
    v2 = wp.transform_point(xform, wp.mesh_eval_position(mesh_0, face, 0.0, 0.0))

    # compute bounds of the query triangle
    lower = cw_min(cw_min(v0, v1), v2)
    upper = cw_max(cw_max(v0, v1), v2)

    query = wp.mesh_query_aabb(mesh_1, lower, upper)

    for f in query:
        u0 = wp.mesh_eval_position(mesh_1, f, 1.0, 0.0)
        u1 = wp.mesh_eval_position(mesh_1, f, 0.0, 1.0)
        u2 = wp.mesh_eval_position(mesh_1, f, 0.0, 0.0)

        # test for triangle intersection
        i = wp.intersect_tri_tri(v0, v1, v2, u0, u1, u2)

        if i > 0:
            result[batch] = 1
            return

        # use if you want to count all intersections
        # wp.atomic_add(result, batch, i)


class Example:
    def __init__(self, stage_path="example_mesh_intersect.usd"):
        rng = np.random.default_rng(42)

        self.query_count = 1024
        self.has_queried = False

        self.path_0 = os.path.join(warp.examples.get_asset_directory(), "cube.usd")
        self.path_1 = os.path.join(warp.examples.get_asset_directory(), "sphere.usd")

        self.mesh_0 = self.load_mesh(self.path_0, "/root/cube")
        self.mesh_1 = self.load_mesh(self.path_1, "/root/sphere")

        self.query_num_faces = int(len(self.mesh_0.indices) / 3)
        self.query_num_points = len(self.mesh_0.points)

        # generate random relative transforms
        self.xforms = []

        for _ in range(self.query_count):
            # random offset
            p = wp.vec3(rng.random(size=3) * 0.5 - 0.5) * 5.0

            # random orientation
            axis = wp.normalize(wp.vec3(rng.random(size=3) * 0.5 - 0.5))
            angle = rng.random()

            q = wp.quat_from_axis_angle(wp.normalize(axis), angle)

            self.xforms.append(wp.transform(p, q))

        self.array_result = wp.zeros(self.query_count, dtype=int)
        self.array_xforms = wp.array(self.xforms, dtype=wp.transform)

        # renderer
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = None

    def step(self):
        with wp.ScopedTimer("step"):
            wp.launch(
                kernel=intersect,
                dim=self.query_num_faces * self.query_count,
                inputs=[self.mesh_0.id, self.mesh_1.id, self.query_num_faces, self.array_xforms, self.array_result],
            )

    def render(self):
        if self.renderer is None:
            return

        # bring results back to host
        result = self.array_result.numpy()

        with wp.ScopedTimer("render", active=True):
            self.renderer.begin_frame(0.0)

            for i in range(self.query_count):
                spacing = 8.0
                offset = i * spacing

                xform = self.xforms[i]
                self.renderer.render_ref(
                    f"mesh_{i}_0",
                    self.path_0,
                    pos=wp.vec3(xform.p[0] + offset, xform.p[1], xform.p[2]),
                    rot=xform.q,
                    scale=wp.vec3(1.0, 1.0, 1.0),
                )
                self.renderer.render_ref(
                    f"mesh_{i}_1",
                    self.path_1,
                    pos=wp.vec3(offset, 0.0, 0.0),
                    rot=wp.quat_identity(),
                    scale=wp.vec3(1.0, 1.0, 1.0),
                )

                # if pair intersects then draw a small box above the pair
                if result[i] > 0:
                    self.renderer.render_box(
                        f"result_{i}",
                        pos=wp.vec3(xform.p[0] + offset, xform.p[1] + 5.0, xform.p[2]),
                        rot=wp.quat_identity(),
                        extents=(0.1, 0.1, 0.1),
                    )

            self.renderer.end_frame()

    # create collision meshes
    def load_mesh(self, path, prim):
        usd_stage = Usd.Stage.Open(path)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(prim))

        mesh = wp.Mesh(
            points=wp.array(usd_geom.GetPointsAttr().Get(), dtype=wp.vec3),
            indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
        )

        return mesh


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_mesh_intersect.usd",
        help="Path to the output USD file.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        example.step()
        example.render()

        if example.renderer:
            example.renderer.save()
