# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Gyroscopic
#
# Demonstrates the Dzhanibekov effect where rigid bodies will tumble in 
# free space due to unstable axes of rotation.
#
###########################################################################

import warp as wp
import warp.sim

from sim_demo import WarpSimDemonstration, run_demo

class Demo(WarpSimDemonstration):
    
    sim_name = "example_sim_rigid_gyroscopic"
    env_offset=(2.0, 0.0, 2.0)
    tiny_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    activate_ground_plane = False
    
    def create_articulation(self, builder):

        self.scale = 0.5

        b = builder.add_body()    

        # axis shape
        builder.add_shape_box( 
            pos=(0.3*self.scale, 0.0, 0.0),
            hx=0.25*self.scale,
            hy=0.1*self.scale,
            hz=0.1*self.scale,
            density=100.0,
            body=b)

        # tip shape
        builder.add_shape_box(
            pos=(0.0, 0.0, 0.0),
            hx=0.05*self.scale,
            hy=0.2*self.scale,
            hz=1.0*self.scale,
            density=100.0,
            body=b)

        # initial spin 
        builder.body_qd[0] = (25.0, 0.01, 0.01, 0.0, 0.0, 0.0)

        builder.gravity = 0.0

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)   
    
    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state)
            self.renderer.end_frame()
   
        self.sim_time += self.sim_dt


if __name__ == "__main__":
    run_demo(Demo)

