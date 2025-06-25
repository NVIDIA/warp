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

import importlib.util
import os

from asv_runner.benchmarks.mark import skip_benchmark_if

import warp as wp

pxr = importlib.util.find_spec("pxr")
USD_AVAILABLE = pxr is not None


def get_asset_directory():
    return os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "..", "..", "warp", "examples", "assets")


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


@wp.kernel
def init_xforms(kernel_seed: int, xforms: wp.array(dtype=wp.transform)):
    i = wp.tid()

    state = wp.rand_init(kernel_seed, i)

    # random offset
    p = (wp.vec3(wp.randf(state), wp.randf(state), wp.randf(state)) * 0.5 - wp.vec3(0.5, 0.5, 0.5)) * 5.0

    # random orientation
    axis = wp.normalize(wp.vec3(wp.randf(state), wp.randf(state), wp.randf(state)) * 0.5 - wp.vec3(0.5, 0.5, 0.5))
    angle = wp.randf(state)

    q = wp.quat_from_axis_angle(wp.normalize(axis), angle)

    xforms[i] = wp.transform(p, q)


class MeshIntersect:
    number = 250

    def setup(self):
        wp.init()
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)
        wp.build.clear_kernel_cache()

        self.query_count = 1024
        self.has_queried = False

        self.path_0 = os.path.join(get_asset_directory(), "cube.usd")
        self.path_1 = os.path.join(get_asset_directory(), "sphere.usd")

        self.mesh_0 = self.load_mesh(self.path_0, "/root/cube")
        self.mesh_1 = self.load_mesh(self.path_1, "/root/sphere")

        self.query_num_faces = int(len(self.mesh_0.indices) / 3)
        self.query_num_points = len(self.mesh_0.points)

        self.array_result = wp.zeros(self.query_count, dtype=int, device=self.device)
        self.array_xforms = wp.empty(self.query_count, dtype=wp.transform, device=self.device)

        # generate random relative transforms
        wp.launch(init_xforms, (self.query_count,), inputs=[42, self.array_xforms], device=self.device)

        self.cmd = wp.launch(
            kernel=intersect,
            dim=self.query_num_faces * self.query_count,
            inputs=[self.mesh_0.id, self.mesh_1.id, self.query_num_faces, self.array_xforms, self.array_result],
            device=self.device,
            record_cmd=True,
        )

        wp.synchronize_device(self.device)

    # create collision meshes
    def load_mesh(self, path, prim):
        from pxr import Usd, UsdGeom

        usd_stage = Usd.Stage.Open(path)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(prim))

        mesh = wp.Mesh(
            points=wp.array(usd_geom.GetPointsAttr().Get(), dtype=wp.vec3, device=self.device),
            indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int, device=self.device),
        )

        return mesh

    @skip_benchmark_if(USD_AVAILABLE is False)
    def time_intersect(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)
