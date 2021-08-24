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

import render_sim as render

wp.init()

sim_steps = 2000
sim_dt = 1.0/120.0
sim_time = 0.0

device = "cpu"

builder = wp.sim.ModelBuilder()

for i in range(1):

    b = builder.add_body(
            parent=i-1,
            origin=wp.transform([i, 0.0, 0.0], wp.quat_identity()),
            joint_xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
            axis=(0.0, 0.0, 1.0),
            type=wp.sim.JOINT_REVOLUTE)

    s = builder.add_shape_box( 
            pos=(0.0, 0.0, 0.0),
            hx=0.5,
            hy=0.2,
            hz=0.2,
            body=b)


model = builder.finalize(device)
model.ground = False

integrator = wp.sim.SemiImplicitIntegrator()
state = model.state()

renderer = render.SimRenderer(model, "tests/outputs/test_sim_rigid_chain.usda")

for i in range(sim_steps):

    state.clear_forces()

    state = integrator.simulate(model, state, state, sim_dt)   
    
    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()
   

    sim_time += sim_dt


renderer.save()




