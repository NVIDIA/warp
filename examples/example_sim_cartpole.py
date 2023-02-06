# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation 
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os
import math
import warp as wp
import warp.sim
import numpy as np

from sim_demo import WarpSimDemonstration, run_demo, IntegratorType

class Demo(WarpSimDemonstration):

    sim_name = "example_sim_cartpole"
    env_offset=(1.0, 0.0, 4.0)
    tiny_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)
    
    sim_substeps_euler = 16
    sim_substeps_xpbd = 3

    activate_ground_plane = False
    integrator_type = IntegratorType.EULER

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"),
            builder,
            xform=wp.transform(np.zeros(3), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
            floating=False, 
            density=0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False)
        builder.joint_q[-3:] = [0.0, 0.3, 0.0]
        builder.joint_target[-3:] = [0.0, 0.0, 0.0]

if __name__ == "__main__":
    run_demo(Demo)