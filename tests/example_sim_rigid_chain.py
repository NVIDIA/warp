# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

sim_steps = 200
sim_substeps = 32
sim_dt = 1.0/60.0
sim_time = 0.0

chain_length = 8
chain_width = 1.0
chain_types = [wp.sim.JOINT_REVOLUTE,
               wp.sim.JOINT_FIXED, 
               wp.sim.JOINT_BALL,
               wp.sim.JOINT_UNIVERSAL,
               wp.sim.JOINT_COMPOUND]

device = wp.get_preferred_device()

builder = wp.sim.ModelBuilder()


for c, t in enumerate(chain_types):

    # start a new articulation
    builder.add_articulation()

    for i in range(chain_length):

        if i == 0:
            parent = -1
            parent_joint_xform = wp.transform([0.0, 0.0, c*1.0], wp.quat_identity())           
        else:
            parent = builder.joint_count-1
            parent_joint_xform = wp.transform([chain_width, 0.0, 0.0], wp.quat_identity())

        joint_type = t

        if joint_type == wp.sim.JOINT_REVOLUTE:

            joint_axis=(0.0, 0.0, 1.0)
            joint_limit_lower=-np.deg2rad(60.0)
            joint_limit_upper=np.deg2rad(60.0)

        elif joint_type == wp.sim.JOINT_PRISMATIC:
            joint_axis=(1.0, 0.0, 0.0),
            joint_limit_lower=0.0,
            joint_limit_upper=0.5,

        elif joint_type == wp.sim.JOINT_BALL:
            joint_axis=(0.0, 0.0, 0.0)
            joint_limit_lower = 100.0
            joint_limit_upper = -100.0

        elif joint_type == wp.sim.JOINT_FIXED:
            joint_axis=(0.0, 0.0, 0.0)
            joint_limit_lower = 0.0
            joint_limit_upper = 0.0
       
        elif joint_type == wp.sim.JOINT_COMPOUND:
            joint_limit_lower=-np.deg2rad(60.0)
            joint_limit_upper=np.deg2rad(60.0)

        # create body
        b = builder.add_body(
                parent=parent,
                origin=wp.transform([i, 0.0, c*1.0], wp.quat_identity()),
                joint_xform=parent_joint_xform,
                joint_axis=joint_axis,
                joint_type=joint_type,
                joint_limit_lower=joint_limit_lower,
                joint_limit_upper=joint_limit_upper,
                joint_target_ke=0.0,
                joint_target_kd=0.0,
                joint_limit_ke=30.0,
                joint_limit_kd=30.0,
                joint_armature=0.1)

        # create shape
        s = builder.add_shape_box( 
                pos=(chain_width*0.5, 0.0, 0.0),
                hx=chain_width*0.5,
                hy=0.1,
                hz=0.1,
                density=10.0,
                body=b)


model = builder.finalize(device)
model.ground = False

integrator = wp.sim.SemiImplicitIntegrator()
state = model.state()

renderer = wp.sim.render.SimRenderer(model, "tests/outputs/example_sim_chain.usd")

for i in range(sim_steps):

    for s in range(sim_substeps):

        state.clear_forces()

        state = integrator.simulate(model, state, state, sim_dt/sim_substeps)   
    
    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()
   
    sim_time += sim_dt

renderer.save()




