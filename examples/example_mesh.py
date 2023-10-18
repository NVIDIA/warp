# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
import warp.render

wp.init()


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

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 1.5

    if wp.mesh_query_point_sign_normal(mesh, xpred, max_dist, sign, face_index, face_u, face_v):
        p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

        delta = xpred - p

        dist = wp.length(delta) * sign
        err = dist - margin

        # mesh collision
        if err < 0.0:
            n = wp.normalize(delta) * sign
            xpred = xpred - n * err

    # pbd update
    v = (xpred - x) * (1.0 / dt)
    x = xpred

    positions[tid] = x
    velocities[tid] = v


class Example:
    def __init__(self, stage):
        self.num_particles = 1000

        self.sim_steps = 500
        self.sim_dt = 1.0 / 60.0

        self.sim_time = 0.0
        self.sim_timers = {}

        self.sim_margin = 0.1

        self.renderer = wp.render.UsdRenderer(stage)

        usd_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets/bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/bunny/bunny"))
        usd_scale = 10.0

        # create collision mesh
        self.mesh = wp.Mesh(
            points=wp.array(usd_geom.GetPointsAttr().Get() * usd_scale, dtype=wp.vec3),
            indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
        )

        # random particles
        init_pos = (np.random.rand(self.num_particles, 3) - np.array([0.5, -1.5, 0.5])) * 10.0
        init_vel = np.random.rand(self.num_particles, 3) * 0.0

        self.positions = wp.from_numpy(init_pos, dtype=wp.vec3)
        self.velocities = wp.from_numpy(init_vel, dtype=wp.vec3)

    def update(self):
        with wp.ScopedTimer("simulate", detailed=False, dict=self.sim_timers):
            wp.launch(kernel=deform, dim=len(self.mesh.points), inputs=[self.mesh.points, self.sim_time])

            # refit the mesh BVH to account for the deformation
            self.mesh.refit()

            wp.launch(
                kernel=simulate,
                dim=self.num_particles,
                inputs=[self.positions, self.velocities, self.mesh.id, self.sim_margin, self.sim_dt],
            )

            self.sim_time += self.sim_dt

    def render(self, is_live=False):
        with wp.ScopedTimer("render", detailed=False):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render_mesh(name="mesh", points=self.mesh.points.numpy(), indices=self.mesh.indices.numpy())
            self.renderer.render_points(name="points", points=self.positions.numpy(), radius=self.sim_margin)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_mesh.usd")

    example = Example(stage_path)

    for i in range(example.sim_steps):
        example.update()
        example.render()

    example.renderer.save()
