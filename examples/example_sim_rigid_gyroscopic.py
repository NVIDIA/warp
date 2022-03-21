# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Gyroscopic
#
# Demonstrates the Dzhanibekov effect where rigid bodies will tumble in 
# free space due to unstable axes of rotation.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

sim_steps = 2000
sim_dt = 1.0/120.0
sim_time = 0.0

device = wp.get_preferred_device()

builder = wp.sim.ModelBuilder()

builder.add_body(
    parent=-1,
    origin=wp.transform_identity())
    
scale = 0.5

# axis shape
builder.add_shape_box( 
    pos=(0.3*scale, 0.0, 0.0),
    hx=0.25*scale,
    hy=0.1*scale,
    hz=0.1*scale,
    density=100.0,
    body=0)

# tip shape
builder.add_shape_box(
    pos=(0.0, 0.0, 0.0),
    hx=0.05*scale,
    hy=0.2*scale,
    hz=1.0*scale,
    density=100.0,
    body=0)

# initial spin 
builder.body_qd[0] = (25.0, 0.01, 0.01, 0.0, 0.0, 0.0)

model = builder.finalize(device)
model.gravity[1] = 0.0
model.ground = False

integrator = wp.sim.SemiImplicitIntegrator()
state = model.state()

renderer = wp.sim.render.SimRenderer(model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_gyroscopic.usd"))

for i in range(sim_steps):

    state.clear_forces()

    state = integrator.simulate(model, state, state, sim_dt)   
    
    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()
   
    sim_time += sim_dt

renderer.save()




