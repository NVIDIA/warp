# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import warp.sim

wp.config.mode = "debug"

import render_sim as render

wp.init()

sim_steps = 200
sim_substeps = 32
sim_dt = 1.0/60.0
sim_time = 0.0

device = "cpu"

builder = wp.sim.ModelBuilder()

for i in range(5):

    if i == 0:
        joint_xform= wp.transform_identity()
    else:
        joint_xform = wp.transform([1.0, 0.0, 0.0], wp.quat_identity())

    b = builder.add_body(
            parent=i-1,
            origin=wp.transform([i, 0.0, 0.0], wp.quat_identity()),
            joint_xform=joint_xform,
            
            # revolute
            # joint_axis=(0.0, 0.0, 1.0),
            # joint_type=wp.sim.JOINT_REVOLUTE,
            # joint_limit_lower=-np.deg2rad(60.0),
            # joint_limit_upper=np.deg2rad(60.0),
            
            # prismatic
            # joint_axis=(1.0, 0.0, 0.0),
            # joint_type=wp.sim.JOINT_PRISMATIC,
            # joint_limit_lower=0.0,
            # joint_limit_upper=0.5,

            # ball
            # joint_type=wp.sim.JOINT_BALL,

            # fixed
            # joint_type=wp.sim.JOINT_FIXED,

            joint_limit_ke=100.0,
            joint_limit_kd=10.0)

    s = builder.add_shape_box( 
            pos=(0.5, 0.0, 0.0),
            hx=0.5,
            hy=0.1,
            hz=0.1,
            density=10.0,
            body=b)

builder.body_qd[0] = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]

model = builder.finalize(device)
model.ground = False
#model.gravity = np.array([0.0, 0.0, 0.0])

integrator = wp.sim.SemiImplicitIntegrator()
state = model.state()

renderer = render.SimRenderer(model, "tests/outputs/test_sim_rigid_chain.usda")

for i in range(sim_steps):

    for s in range(sim_substeps):

        state.clear_forces()

        state = integrator.simulate(model, state, state, sim_dt/sim_substeps)   
    
    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()
   
    sim_time += sim_dt


renderer.save()
print("finished")




