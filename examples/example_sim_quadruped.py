# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Quadruped
#
# Shows how to set up a simulation of a rigid-body quadruped articulation 
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import math
import os

import warp as wp
import warp.sim

from sim_demo import WarpSimDemonstration, run_demo

class Demo(WarpSimDemonstration):
    sim_name = "example_sim_quadruped"
    env_offset=(1.5, 0.0, 1.5)
    tiny_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    xpbd_settings = dict(
        iterations=3,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.5,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    def create_articulation(self, builder):
        wp.sim.parse_urdf(os.path.join(os.path.dirname(__file__), "assets/quadruped.urdf"), 
            builder,
            xform=wp.transform([0.0, 0.7, 0.0], wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
            floating=True,
            density=1000,
            armature=0.01,
            stiffness=120,
            damping=1,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1)

        builder.joint_q[-12:] = [
            0.2, 0.4, -0.6,
            -0.2, -0.4, 0.6,
            -0.2, 0.4, -0.6,
            0.2, -0.4, 0.6]

        builder.joint_target[-12:] = [
            0.2, 0.4, -0.6,
            -0.2, -0.4, 0.6,
            -0.2, 0.4, -0.6,
            0.2, -0.4, 0.6]

if __name__ == "__main__":
    run_demo(Demo)
