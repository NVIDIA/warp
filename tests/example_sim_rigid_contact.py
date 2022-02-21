# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxr import Usd, UsdGeom, Gf, Sdf

import warp as wp
import warp.sim
import warp.sim.render

from warp.utils import quat_identity


wp.init()

sim_steps = 2000
sim_dt = 1.0/240.0
sim_time = 0.0

num_bodies = 8

device = "cuda"

builder = wp.sim.ModelBuilder()

scale = 0.5

ke = 1.e+4
kd = 100.0
kf = 100.0

# boxes
for i in range(num_bodies):
    
    b = builder.add_body(origin=wp.transform((i, 1.0, 0.0), quat_identity()))

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
    
    b = builder.add_body(origin=wp.transform((i, 1.0, 2.0), quat_identity()))

    s = builder.add_shape_sphere(
        pos=(0.0, 0.0, 0.0),
        radius=0.25*scale, 
        body=b,
        ke=ke,
        kd=kd,
        kf=kf)

# capsules
for i in range(num_bodies):
    
    b = builder.add_body(origin=wp.transform((i, 1.0, 4.0), quat_identity()))

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

model.collide(state)

# one time collide for ground contact
renderer = wp.sim.render.SimRenderer(model, "tests/outputs/test_sim_rigid_contact.usda")

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




