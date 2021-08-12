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
from warp.utils import quat_identity

import render

wp.init()

sim_steps = 2000
sim_dt = 1.0/240.0
sim_time = 0.0

num_bodies = 32

device = "cpu"

builder = wp.sim.ModelBuilder()

for i in range(num_bodies):
    
    builder.add_body(
        parent=-1,
        X_pj=wp.transform((i, 1.0, 0.0), quat_identity()))

    scale = 0.5

    # shape
    builder.add_shape_box( 
        pos=(0.0, 0.0, 0.0),
        hx=0.25*scale,
        hy=0.1*scale,
        hz=0.1*scale,
        density=1000.0,
        body=i,
        ke=1.e+3,
        kd=10.0,
        kf=10.0,
        mu=0.5)

    # builder.add_shape_sphere( 
    #     pos=(0.0, 0.0, 0.0),
    #     radius=0.25*scale,
    #     density=100.0,
    #     body=i,
    #     ke=1.e+3,
    #     kd=1.0,
    #     kf=10.0,
    #     mu=0.5)


    # initial spin 
    builder.body_qd[i] = (0.0, 2.0, 10.0, 0.0, 0.0, 0.0)

model = builder.finalize(device)
model.ground = True

integrator = wp.sim.SemiImplicitIntegrator()
state = model.state()

model.collide(state)

# one time collide for ground contact
stage = render.UsdRenderer("tests/outputs/test_sim_rigid_contact.usda")

for i in range(sim_steps):

    state = integrator.simulate(model, state, state, sim_dt)   

    X_wb = state.body_q.to("cpu").numpy()
    X_bs = model.shape_transform.to("cpu").numpy()

    stage.begin_frame(sim_time)
    
    stage.render_ground()   

    for i in range(model.body_count):
    
        # shape world transform    
        X_ws = wp.transform_multiply(wp.transform(X_wb[i, 0:3], X_wb[i,3:7]), wp.transform(X_bs[0, 0:3], X_bs[0,3:7]))

        stage.render_box(X_ws[0], X_ws[1], extents=builder.shape_geo_scale[0], name="body" + str(i))
        #stage.render_sphere(X_ws[0], X_ws[1], radius=builder.shape_geo_scale[0][0], name="body" + str(i))

    stage.end_frame()
    
    sim_time += sim_dt


stage.save()




