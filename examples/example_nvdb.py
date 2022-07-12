# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example NanoVDB
#
# Shows how to implement a particle simulation with collision against
# a NanoVDB signed-distance field. In this example the NanoVDB field
# is created offline in Houdini. The particle kernel uses the Warp
# wp.volume_sample_f() method to compute the SDF and normal at a point.
#
###########################################################################

import os
import math

import numpy as np
import warp as wp

wp.init()

@wp.func
def volume_grad(volume: wp.uint64,
                p: wp.vec3):
    
    eps = 1.0
    q = wp.volume_world_to_index(volume, p)

    # compute gradient of the SDF using finite differences
    dx = wp.volume_sample_f(volume, q + wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(volume, q - wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR)
    dy = wp.volume_sample_f(volume, q + wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(volume, q - wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR)
    dz = wp.volume_sample_f(volume, q + wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR) - wp.volume_sample_f(volume, q - wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR)

    return wp.normalize(wp.vec3(dx, dy, dz))

@wp.kernel
def simulate(positions: wp.array(dtype=wp.vec3),
            velocities: wp.array(dtype=wp.vec3),
            volume: wp.uint64,
            restitution: float,
            margin: float,
            dt: float):
    
    
    tid = wp.tid()

    x = positions[tid]
    v = velocities[tid]

    v = v + wp.vec3(0.0, 0.0, -980.0)*dt - v*0.1*dt
    xpred = x + v*dt
    xpred_local = wp.volume_world_to_index(volume, xpred)
    
    d = wp.volume_sample_f(volume, xpred_local, wp.Volume.LINEAR)

    if (d < margin):
        
        n = volume_grad(volume, xpred)
        err = d - margin

        # mesh collision
        xpred = xpred - n*err


    # ground collision
    if (xpred[2] < 0.0):
        xpred = wp.vec3(xpred[0], xpred[1], 0.0)

    # pbd update
    v = (xpred - x)*(1.0/dt)
    x = xpred

    positions[tid] = x
    velocities[tid] = v

class Example:

    def __init__(self, stage):

        self.num_particles = 10000

        self.sim_steps = 1000
        self.sim_dt = 1.0/60.0
        self.sim_substeps = 3

        self.sim_time = 0.0
        self.sim_timers = {}
        self.sim_render = True

        self.sim_restitution = 0.0
        self.sim_margin = 15.0

        self.renderer = wp.render.UsdRenderer(stage, upaxis="z")
        self.renderer.render_ground(size=10000.0)

        init_pos = 1000.0*(np.random.rand(self.num_particles, 3)*2.0 - 1.0) + np.array((0.0, 0.0, 3000.0))
        init_vel = np.random.rand(self.num_particles, 3)

        self.positions = wp.from_numpy(init_pos.astype(np.float32), dtype=wp.vec3)
        self.velocities = wp.from_numpy(init_vel.astype(np.float32), dtype=wp.vec3)

        # load collision volume
        file = open(os.path.join(os.path.dirname(__file__), "assets/rocks.nvdb"), "rb")

        # create Volume object
        self.volume = wp.Volume.load_from_nvdb(file)

    def update(self):

        with wp.ScopedTimer("simulate", detailed=False, dict=self.sim_timers):

            for s in range(self.sim_substeps):
                wp.launch(
                    kernel=simulate, 
                    dim=self.num_particles, 
                    inputs=[self.positions, self.velocities, self.volume.id, self.sim_restitution, self.sim_margin, self.sim_dt/float(self.sim_substeps)])

            wp.synchronize_device()
    
    def render(self, is_live=False):

        with wp.ScopedTimer("render", detailed=False):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)

            self.renderer.render_ref(name="collision", path=os.path.join(os.path.dirname(__file__), "assets/rocks.usd"), pos=(0.0, 0.0, 0.0), rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi), scale=(1.0, 1.0, 1.0))
            self.renderer.render_points(name="points", points=self.positions.numpy(), radius=self.sim_margin)

            self.renderer.end_frame()

        self.sim_time += self.sim_dt
    



if __name__ == '__main__':
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_nvdb.usd")

    example = Example(stage_path)

    for i in range(example.sim_steps):
        example.update()
        example.render()

    example.renderer.save()