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
import warp as wp
import warp.sim
import numpy as np

from sim_demo import WarpSimDemonstration, run_demo


class Demo(WarpSimDemonstration):
    sim_name = "example_sim_rigid_chain"
    env_offset=(6.0, 0.0, 6.0)
    num_envs = 1
    tiny_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    activate_ground_plane = False

    def create_articulation(self, builder):

        self.chain_length = 8
        self.chain_width = 1.0
        self.chain_types = [
            wp.sim.JOINT_REVOLUTE,
            wp.sim.JOINT_FIXED, 
            wp.sim.JOINT_BALL,
            wp.sim.JOINT_UNIVERSAL,
            wp.sim.JOINT_COMPOUND
        ]

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


                # create body
                b = builder.add_body(
                        origin=wp.transform([i, 0.0, c*1.0], wp.quat_identity()),
                        armature=0.1)

                # create shape
                s = builder.add_shape_box( 
                        pos=(self.chain_width*0.5, 0.0, 0.0),
                        hx=self.chain_width*0.5,
                        hy=0.1,
                        hz=0.1,
                        density=10.0,
                        body=b)

                joint_type = t

                if joint_type == wp.sim.JOINT_REVOLUTE:
 
                    joint_limit_lower=-np.deg2rad(60.0)
                    joint_limit_upper=np.deg2rad(60.0)
                    builder.add_joint_revolute(
                        parent=parent,
                        child=b,
                        axis=(0.0, 0.0, 1.0),
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                        limit_lower=joint_limit_lower,
                        limit_upper=joint_limit_upper,
                        target_ke=0.0,
                        target_kd=0.0,
                        limit_ke=30.0,
                        limit_kd=30.0,
                    )

                elif joint_type == wp.sim.JOINT_UNIVERSAL:
                    builder.add_joint_universal(
                        parent=parent,
                        child=b,
                        axis_0=wp.sim.JointAxis((1.0, 0.0, 0.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        axis_1=wp.sim.JointAxis((0.0, 0.0, 1.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                    )

                elif joint_type == wp.sim.JOINT_BALL:
                    builder.add_joint_ball(
                        parent=parent,
                        child=b,
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                    )

                elif joint_type == wp.sim.JOINT_FIXED:
                    builder.add_joint_fixed(
                        parent=parent,
                        child=b,
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                    )
            
                elif joint_type == wp.sim.JOINT_COMPOUND:
                    builder.add_joint_compound(
                        parent=parent,
                        child=b,
                        axis_0=wp.sim.JointAxis((1.0, 0.0, 0.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        axis_1=wp.sim.JointAxis((0.0, 1.0, 0.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        axis_2=wp.sim.JointAxis((0.0, 0.0, 1.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                    )

if __name__ == "__main__":
    run_demo(Demo)
