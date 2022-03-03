# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import warp.sim
import warp.sim.render

wp.config.mode = "debug"

wp.init()

sim_steps = 200
sim_substeps = 32
sim_dt = 1.0/60.0
sim_time = 0.0

device = "cpu"

builder = wp.sim.ModelBuilder()

for a in range(1):
    builder.add_articulation()

    for i in range(1):

        if i == 0:
            joint_xform= wp.transform([0.0, 0.0, a*1.0], wp.quat_identity())
            parent = -1
        else:
            joint_xform = wp.transform([1.0, 0.0, 0.0], wp.quat_identity())
            parent = len(builder.joint_type)-1
        
        b = builder.add_body(
                parent=parent,
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

                # compound
                joint_type=wp.sim.JOINT_COMPOUND,
                joint_limit_lower=-np.deg2rad(60.0),
                joint_limit_upper=np.deg2rad(60.0),
                joint_target_ke=0.0,
                joint_target_kd=0.0,
                joint_limit_ke=30.0,
                joint_limit_kd=30.0,
                joint_armature=0.1)

        s = builder.add_shape_box( 
                pos=(0.5, 0.0, 0.0),
                hx=0.5,
                hy=0.1,
                hz=0.1,
                density=10.0,
                body=b)


#builder.body_qd[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# builder.joint_target = [math.radians(30), math.radians(30), math.radians(45), 
#                         math.radians(30), math.radians(30), math.radians(45)]

# builder.joint_q = [0.0, 0.0, 0.0, 
#                    math.radians(30), math.radians(30), math.radians(45)]

# builder.joint_act = [0.0, 0.0, 0.1, 
#                      0.0, 0.0, 0.1]


model = builder.finalize(device)
model.ground = False
model.gravity = np.array([0.0, -10.0, 0.0])

integrator = wp.sim.SemiImplicitIntegrator()
state = model.state()

wp.sim.eval_fk(
    model,
    model.joint_q,
    model.joint_qd,
    None,
    state)

renderer = wp.sim.render.SimRenderer(model, "tests/outputs/test_sim_rigid_chain.usda")

for i in range(sim_steps):

    for s in range(sim_substeps):

        state.clear_forces()

        state = integrator.simulate(model, state, state, sim_dt/sim_substeps)   
    
    wp.sim.eval_ik(
        model,
        state,
        model.joint_q,
        model.joint_qd)

    print(np.rad2deg(model.joint_q))


    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()
   
    sim_time += sim_dt


renderer.save()
print("finished")




