# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Neo-Hookean
#
# Shows a simulation of an Neo-Hookean FEM beam being twisted through a
# 180 degree rotation.
#
###########################################################################

import math
import os

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


@wp.kernel
def twist_points(
    rest: wp.array(dtype=wp.vec3), points: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), xform: wp.transform
):
    tid = wp.tid()

    r = rest[tid]
    p = points[tid]
    m = mass[tid]

    # twist the top layer of particles in the beam
    if m == 0 and p[1] != 0.0:
        points[tid] = wp.transform_point(xform, r)


@wp.kernel
def compute_volume(points: wp.array(dtype=wp.vec3), indices: wp.array2d(dtype=int), volume: wp.array(dtype=float)):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    x0 = points[i]
    x1 = points[j]
    x2 = points[k]
    x3 = points[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v = wp.dot(x10, wp.cross(x20, x30)) / 6.0

    wp.atomic_add(volume, 0, v)


class Example:
    def __init__(self, stage):
        sim_fps = 60.0
        self.sim_substeps = 64
        sim_duration = 5.0
        self.sim_frames = int(sim_duration * sim_fps)
        self.sim_dt = (1.0 / sim_fps) / self.sim_substeps
        self.sim_time = 0.0
        self.lift_speed = 2.5 / sim_duration * 2.0  # from Smith et al.
        self.rot_speed = math.pi / sim_duration

        builder = wp.sim.ModelBuilder()

        cell_dim = 15
        cell_size = 2.0 / cell_dim

        center = cell_size * cell_dim * 0.5

        builder.add_soft_grid(
            pos=wp.vec3(-center, 0.0, -center),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cell_dim,
            dim_y=cell_dim,
            dim_z=cell_dim,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=100.0,
            fix_bottom=True,
            fix_top=True,
            k_mu=1000.0,
            k_lambda=5000.0,
            k_damp=0.0,
        )

        self.model = builder.finalize()
        self.model.ground = False
        self.model.gravity[1] = 0.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.rest = self.model.state()
        self.rest_vol = (cell_size * cell_dim) ** 3

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.volume = wp.zeros(1, dtype=wp.float32)

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=20.0)

    def update(self):
        with wp.ScopedTimer("simulate"):
            xform = wp.transform(
                (0.0, self.lift_speed * self.sim_time, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self.rot_speed * self.sim_time),
            )
            wp.launch(
                kernel=twist_points,
                dim=len(self.state_0.particle_q),
                inputs=[self.rest.particle_q, self.state_0.particle_q, self.model.particle_mass, xform],
            )

            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.sim_dt

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

            self.volume.zero_()
            wp.launch(
                kernel=compute_volume,
                dim=self.model.tet_count,
                inputs=[self.state_0.particle_q, self.model.tet_indices, self.volume],
            )

    def render(self, is_live=False):
        with wp.ScopedTimer("render"):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_neo_hookean.usd")

    example = Example(stage_path)

    for i in range(example.sim_frames):
        example.update()
        example.render()

    example.renderer.save()
