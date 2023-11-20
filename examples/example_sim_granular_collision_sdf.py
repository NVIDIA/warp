# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Granular Collision SDF
#
# Shows how to set up a particle-based granular material model using the
# wp.sim.ModelBuilder(). This version shows how to create collision geometry
# objects from SDFs.
#
###########################################################################

import math
import os

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    def __init__(self, stage):
        self.frame_dt = 1.0 / 60
        self.frame_count = 400

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = self.frame_count * self.sim_substeps
        self.sim_time = 0.0

        self.radius = 0.1

        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius

        builder.add_particle_grid(
            dim_x=16,
            dim_y=32,
            dim_z=16,
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            pos=wp.vec3(0.0, 20.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(2.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        )
        rock_file = open(os.path.join(os.path.dirname(__file__), "assets/rocks.nvdb"), "rb")
        rock_vdb = wp.Volume.load_from_nvdb(rock_file.read())
        rock_file.close()

        rock_sdf = wp.sim.SDF(rock_vdb)

        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=rock_sdf,
            body=-1,
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi),
            scale=wp.vec3(0.01, 0.01, 0.01),
        )

        mins = np.array([-3.0, -3.0, -3.0])
        voxel_size = 0.2
        maxs = np.array([3.0, 3.0, 3.0])
        nums = np.ceil((maxs - mins) / (voxel_size)).astype(dtype=int)
        center = np.array([0.0, 0.0, 0.0])
        rad = 2.5
        sphere_sdf_np = np.zeros(tuple(nums))
        for x in range(nums[0]):
            for y in range(nums[1]):
                for z in range(nums[2]):
                    pos = mins + voxel_size * np.array([x, y, z])
                    dis = np.linalg.norm(pos - center)
                    sphere_sdf_np[x, y, z] = dis - rad

        sphere_vdb = wp.Volume.load_from_numpy(sphere_sdf_np, mins, voxel_size, rad + 3.0 * voxel_size)
        sphere_sdf = wp.sim.SDF(sphere_vdb)

        self.sphere_pos = wp.vec3(3.0, 15.0, 0.0)
        self.sphere_scale = 1.0
        self.sphere_radius = rad
        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=sphere_sdf,
            body=-1,
            pos=self.sphere_pos,
            scale=wp.vec3(self.sphere_scale, self.sphere_scale, self.sphere_scale),
        )

        self.model = builder.finalize()
        self.model.particle_kf = 25.0

        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf *= 2.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=20.0)

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)

            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                wp.sim.collide(self.model, self.state_0)
                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

            self.sim_time += self.frame_dt

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)

            # Note the extra wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi) is because .usd is oriented differently from .nvdb
            self.renderer.render_ref(
                name="collision",
                path=os.path.join(os.path.dirname(__file__), "assets/rocks.usd"),
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
                * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi),
                scale=wp.vec3(0.01, 0.01, 0.01),
            )

            self.renderer.render_sphere(
                name="sphere",
                pos=self.sphere_pos,
                radius=self.sphere_scale * self.sphere_radius,
                rot=wp.quat(0.0, 0.0, 0.0, 1.0),
            )

            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_sdf_shape.usd")

    example = Example(stage_path)

    for _ in range(example.frame_count):
        example.update()
        example.render()

    example.renderer.save()
