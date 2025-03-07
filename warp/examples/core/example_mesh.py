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

###########################################################################
# Example Mesh
#
# Shows how to implement a PBD particle simulation with collision against
# a deforming triangle mesh. The mesh collision uses wp.mesh_query_point_sign_normal()
# to compute the closest point, and wp.Mesh.refit() to update the mesh
# object after deformation.
#
###########################################################################

import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.render


@wp.kernel
def deform(positions: wp.array(dtype=wp.vec3), t: float):
    tid = wp.tid()

    x = positions[tid]

    offset = -wp.sin(x[0]) * 0.02
    scale = wp.sin(t)

    x = x + wp.vec3(0.0, offset * scale, 0.0)

    positions[tid] = x


@wp.kernel
def simulate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    margin: float,
    dt: float,
):
    tid = wp.tid()

    x = positions[tid]
    v = velocities[tid]

    v = v + wp.vec3(0.0, 0.0 - 9.8, 0.0) * dt - v * 0.1 * dt
    xpred = x + v * dt

    max_dist = 1.5

    query = wp.mesh_query_point_sign_normal(mesh, xpred, max_dist)
    if query.result:
        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

        delta = xpred - p

        dist = wp.length(delta) * query.sign
        err = dist - margin

        # mesh collision
        if err < 0.0:
            n = wp.normalize(delta) * query.sign
            xpred = xpred - n * err

    # pbd update
    v = (xpred - x) * (1.0 / dt)
    x = xpred

    positions[tid] = x
    velocities[tid] = v


class Example:
    def __init__(self, stage_path="example_mesh.usd"):
        rng = np.random.default_rng(42)
        self.num_particles = 1000

        self.sim_dt = 1.0 / 60.0

        self.sim_time = 0.0
        self.sim_timers = {}

        self.sim_margin = 0.1

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        usd_scale = 10.0

        # create collision mesh
        self.mesh = wp.Mesh(
            points=wp.array(usd_geom.GetPointsAttr().Get() * usd_scale, dtype=wp.vec3),
            indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
        )

        # random particles
        init_pos = (rng.random((self.num_particles, 3)) - np.array([0.5, -1.5, 0.5])) * 10.0
        init_vel = rng.random((self.num_particles, 3)) * 0.0

        self.positions = wp.from_numpy(init_pos, dtype=wp.vec3)
        self.velocities = wp.from_numpy(init_vel, dtype=wp.vec3)

        # renderer
        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)

    def step(self):
        with wp.ScopedTimer("step", dict=self.sim_timers):
            wp.launch(kernel=deform, dim=len(self.mesh.points), inputs=[self.mesh.points, self.sim_time])

            # refit the mesh BVH to account for the deformation
            self.mesh.refit()

            wp.launch(
                kernel=simulate,
                dim=self.num_particles,
                inputs=[self.positions, self.velocities, self.mesh.id, self.sim_margin, self.sim_dt],
            )

            self.sim_time += self.sim_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_mesh(
                name="mesh",
                points=self.mesh.points.numpy(),
                indices=self.mesh.indices.numpy(),
                colors=(0.35, 0.55, 0.9),
            )
            self.renderer.render_points(
                name="points", points=self.positions.numpy(), radius=self.sim_margin, colors=(0.8, 0.3, 0.2)
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_mesh.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=500, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
