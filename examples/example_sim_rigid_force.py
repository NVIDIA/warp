# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Force
#
# Shows how to apply an external force (torque) to a rigid body causing
# it to roll.
#
###########################################################################

import numpy as np
import warp as wp
import warp.sim

from sim_demo import WarpSimDemonstration, run_demo, RenderMode

class Demo(WarpSimDemonstration):
    sim_name = "example_sim_rigid_force"
    env_offset=(2.0, 0.0, 2.0)
    tiny_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    render_mode = RenderMode.TINY

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    num_envs = 100

    
    def create_articulation(self, builder):
        builder.add_body(origin=wp.transform((0.0, 2.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=0, hx=0.5, hy=0.5, hz=0.5, density=1000.0, ke=2.e+5, kd=1.e+4)

    def before_simulate(self):
        # allocate force buffer
        self.force = wp.array(np.tile([0.0, 0.0, -3000.0, 0.0, 0.0, 0.0], (self.num_envs, 1)), dtype=wp.spatial_vector)

    def custom_update(self):
        # apply force to body at every simulation time step
        self.state.body_f.assign(self.force)

    def update(self):

        with wp.ScopedTimer("simulate"):

            for s in range(self.sim_substeps):

                wp.sim.collide(self.model, self.state_0)

                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.state_0.body_f.assign([ [0.0, 0.0, -3000.0, 0.0, 0.0, 0.0], ])

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.sim_dt

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def render(self, is_live=False):

        with wp.ScopedTimer("render"):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

if __name__ == "__main__":
    run_demo(Demo)
