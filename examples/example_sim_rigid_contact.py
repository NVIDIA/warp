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

sim_steps = 2000
sim_dt = 1.0/240.0
sim_time = 0.0

num_bodies = 8

device = wp.get_preferred_device()

builder = wp.sim.ModelBuilder()

scale = 0.5

ke = 1.e+4
kd = 100.0
kf = 100.0

# boxes
for i in range(num_bodies):
    
    b = builder.add_body(origin=wp.transform((i, 1.0, 0.0), wp.quat_identity()))

    s = builder.add_shape_box( 
        pos=(0.0, 0.0, 0.0),
        hx=0.5*scale,
        hy=0.2*scale,
        hz=0.2*scale,
        body=i,
        ke=ke,
        kd=kd,
        kf=kf)

# spheres
for i in range(num_bodies):
    
    b = builder.add_body(origin=wp.transform((i, 1.0, 2.0), wp.quat_identity()))

    s = builder.add_shape_sphere(
        pos=(0.0, 0.0, 0.0),
        radius=0.25*scale, 
        body=b,
        ke=ke,
        kd=kd,
        kf=kf)

# capsules
for i in range(num_bodies):
    
    b = builder.add_body(origin=wp.transform((i, 1.0, 4.0), wp.quat_identity()))

    s = builder.add_shape_capsule( 
        pos=(0.0, 0.0, 0.0),
        radius=0.25*scale,
        half_width=scale*0.5,
        body=b,
        ke=ke,
        kd=kd,
        kf=kf)

# initial spin 
for i in range(len(builder.body_qd)):
    builder.body_qd[i] = (0.0, 2.0, 10.0, 0.0, 0.0, 0.0)
 
model = builder.finalize(device)
model.ground = True

integrator = wp.sim.SemiImplicitIntegrator()
state = model.state()

# one time collide for ground contact
model.collide(state)

renderer = wp.sim.render.SimRenderer(model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd"))

for i in range(sim_steps):

    state.clear_forces()

    # sim
    state = integrator.simulate(model, state, state, sim_dt)   

    # render
    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()
    
    sim_time += sim_dt


renderer.save()




