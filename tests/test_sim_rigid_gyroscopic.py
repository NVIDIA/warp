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

import render

wp.init()

sim_steps = 2000
sim_dt = 1.0/120.0
sim_time = 0.0

device = "cpu"

builder = wp.sim.ModelBuilder()

builder.add_body(
    parent=-1,
    X_pj=wp.transform_identity(),
    axis=(0.0, 0.0, 0.0),
    type=wp.sim.JOINT_FREE)

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

integrator = wp.sim.SemiImplicitIntegrator()
state = model.state()

stage = render.UsdRenderer("tests/outputs/test_sim_rigid_gyroscopic.usda")

for i in range(sim_steps):

    state = integrator.simulate(model, state, state, sim_dt)   
    
    rigid_xform = state.body_q.to("cpu").numpy()

    xform = wp.transform(rigid_xform[0][0:3], rigid_xform[0][3:7])

    t0 = wp.transform_multiply(xform, wp.transform(builder.shape_transform[0][0], builder.shape_transform[0][1]))
    t1 = wp.transform_multiply(xform, wp.transform(builder.shape_transform[1][0], builder.shape_transform[1][1]))

    stage.begin_frame(sim_time)
    stage.render_box(t0[0], t0[1], extents=builder.shape_geo_scale[0], name="axis")
    stage.render_box(t1[0], t1[1], extents=builder.shape_geo_scale[1], name="tip")
    stage.end_frame()
    

    sim_time += sim_dt


stage.save()




