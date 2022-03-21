# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Force
#
# Shows how to apply an external force (torque) to a rigid body causing
# it to roll.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

import warp.render

# params
sim_width = 8
sim_height = 8

sim_fps = 60.0
sim_substeps = 64
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0
sim_render = True
sim_iterations = 1
sim_relaxation = 1.0

device = wp.get_preferred_device()

builder = wp.sim.ModelBuilder()

builder.add_body(origin=wp.transform((0.0, 2.0, 0.0), wp.quat_identity()))
builder.add_shape_box(body=0, hx=0.5, hy=0.5, hz=0.5, density=1000.0, ke=2.e+5, kd=1.e+4)

model = builder.finalize(device=device)
model.ground = True

integrator = wp.sim.SemiImplicitIntegrator()

state_0 = model.state()
state_1 = model.state()

model.collide(state_0)

stage = wp.sim.render.SimRenderer(model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_force.usd"))

for i in range(sim_frames):
    
    if (sim_render):
    
        with wp.ScopedTimer("render"):

            stage.begin_frame(sim_time)
            stage.render(state_0)
            stage.end_frame()

    with wp.ScopedTimer("simulate"):

        for s in range(sim_substeps):

            wp.sim.collide(model, state_0)

            state_0.clear_forces()
            state_1.clear_forces()

            state_0.body_f.assign([ [0.0, 0.0, -3000.0, 0.0, 0.0, 0.0], ])

            integrator.simulate(model, state_0, state_1, sim_dt)
            sim_time += sim_dt

            # swap states
            (state_0, state_1) = (state_1, state_0)


stage.save()
