# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Chain
#
# Shows how to set up a chain of rigid bodies connected by different joint
# types using wp.sim.ModelBuilder(). There is one chain for each joint
# type, including fixed joints which act as a flexible beam.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:

    def __init__(self, stage):

        self.sim_steps = 200
        self.sim_substeps = 16
        self.sim_dt = 1.0/60.0
        self.sim_time = 0.0

        self.chain_length = 8
        self.chain_width = 1.0
        self.chain_types = [wp.sim.JOINT_REVOLUTE,
                            wp.sim.JOINT_FIXED, 
                            wp.sim.JOINT_BALL,
                            wp.sim.JOINT_UNIVERSAL,
                            wp.sim.JOINT_COMPOUND]

        builder = wp.sim.ModelBuilder()

        for c, t in enumerate(self.chain_types):

            # start a new articulation
            builder.add_articulation()

            for i in range(self.chain_length):

                if i == 0:
                    parent = -1
                    parent_joint_xform = wp.transform([0.0, 0.0, c*1.0], wp.quat_identity())           
                else:
                    parent = builder.joint_count-1
                    parent_joint_xform = wp.transform([self.chain_width, 0.0, 0.0], wp.quat_identity())

                joint_type = t

                if joint_type == wp.sim.JOINT_REVOLUTE:

                    joint_axis=(0.0, 0.0, 1.0)
                    joint_limit_lower=-np.deg2rad(60.0)
                    joint_limit_upper=np.deg2rad(60.0)

                elif joint_type == wp.sim.JOINT_UNIVERSAL:
                    joint_axis=(1.0, 0.0, 0.0)
                    joint_limit_lower=-np.deg2rad(60.0),
                    joint_limit_upper=np.deg2rad(60.0),

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
                        pos=(self.chain_width*0.5, 0.0, 0.0),
                        hx=self.chain_width*0.5,
                        hy=0.1,
                        hz=0.1,
                        density=10.0,
                        body=b)


        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.state = self.model.state()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage)

    def update(self):

        with wp.ScopedTimer("simulate", active=True):
            for s in range(self.sim_substeps):
                self.state.clear_forces()
                self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt/self.sim_substeps)   
    
    def render(self, is_live=False):

        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state)
            self.renderer.end_frame()
        
        self.sim_time += self.sim_dt


if __name__ == '__main__':
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_chain.usd")

    example = Example(stage_path)

    for i in range(example.sim_steps):
        example.update()
        example.render()

    example.renderer.save()


