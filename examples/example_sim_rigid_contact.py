# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Contact
#
# Shows how to set up free rigid bodies with different shape types falling
# and colliding against the ground using wp.sim.ModelBuilder().
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:

    def __init__(self, stage):

        self.sim_steps = 2000
        self.sim_dt = 1.0/60.0
        self.sim_time = 0.0
        self.sim_substeps = 8

        self.num_bodies = 8
        self.scale = 0.5
        self.ke = 1.e+5
        self.kd = 250.0
        self.kf = 500.0

        builder = wp.sim.ModelBuilder()

        # boxes
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((i, 1.0, 0.0), wp.quat_identity()))

            s = builder.add_shape_box( 
                pos=(0.0, 0.0, 0.0),
                hx=0.5*self.scale,
                hy=0.2*self.scale,
                hz=0.2*self.scale,
                body=i,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf)

        # spheres
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((i, 1.0, 2.0), wp.quat_identity()))

            s = builder.add_shape_sphere(
                pos=(0.0, 0.0, 0.0),
                radius=0.25*self.scale, 
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf)

        # capsules
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((i, 1.0, 4.0), wp.quat_identity()))

            s = builder.add_shape_capsule( 
                pos=(0.0, 0.0, 0.0),
                radius=0.25*self.scale,
                half_width=self.scale*0.5,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf)

        # initial spin 
        for i in range(len(builder.body_qd)):
            builder.body_qd[i] = (0.0, 2.0, 10.0, 0.0, 0.0, 0.0)
        
        self.model = builder.finalize()
        self.model.ground = True

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.state = self.model.state()

        # one time collide for ground contact
        self.model.collide(self.state)

        self.renderer = wp.sim.render.SimRenderer(self.model, stage)

    def update(self):

        with wp.ScopedTimer("simulate", active=True):
            
            for i in range(self.sim_substeps):
                self.state.clear_forces()
                self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt/self.sim_substeps)   

    def render(self, is_live=False):

        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state)
            self.renderer.end_frame()
        
        self.sim_time += self.sim_dt


if __name__ == '__main__':
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd")

    example = Example(stage_path)

    for i in range(example.sim_steps):
        example.update()
        example.render()

    example.renderer.save()





