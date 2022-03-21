# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid FEM
#
# Shows how to set up a rigid sphere colliding with an FEM beam
# using wp.sim.ModelBuilder().
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

# params
sim_width = 8
sim_height = 8

sim_fps = 60.0
sim_substeps = 64
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0
sim_iterations = 1
sim_relaxation = 1.0

device = wp.get_preferred_device()

builder = wp.sim.ModelBuilder()


builder.add_soft_grid(
    pos=(0.0, 0.0, 0.0), 
    rot=wp.quat_identity(), 
    vel=(0.0, 0.0, 0.0), 
    dim_x=20, 
    dim_y=10, 
    dim_z=10,
    cell_x=0.1, 
    cell_y=0.1,
    cell_z=0.1,
    density=100.0, 
    k_mu=50000.0, 
    k_lambda=20000.0,
    k_damp=0.0)

builder.add_body(origin=wp.transform((0.5, 2.5, 0.5), wp.quat_identity()))
builder.add_shape_sphere(body=0, radius=0.75, density=100.0)

model = builder.finalize(device=device)
model.ground = True
model.soft_contact_distance = 0.01
model.soft_contact_ke = 1.e+3
model.soft_contact_kd = 0.0
model.soft_contact_kf = 1.e+3

integrator = wp.sim.SemiImplicitIntegrator()

state_0 = model.state()
state_1 = model.state()

model.collide(state_0)

renderer = wp.sim.render.SimRenderer(model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_fem.usd"))

for i in range(sim_frames):
    
    with wp.ScopedTimer("render"): 

        renderer.begin_frame(sim_time)
        renderer.render(state_0)
        renderer.end_frame()

    with wp.ScopedTimer("simulate"):

        for s in range(sim_substeps):

            wp.sim.collide(model, state_0)

            state_0.clear_forces()
            state_1.clear_forces()

            integrator.simulate(model, state_0, state_1, sim_dt)
            sim_time += sim_dt

            # swap states
            (state_0, state_1) = (state_1, state_0)


renderer.save()
