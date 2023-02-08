# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Contact
#
# Shows how to set up free rigid bodies with different shape types falling
# and colliding against each other and the ground using wp.sim.ModelBuilder().
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim

from pxr import Usd, UsdGeom

from environment import Environment, run_env

class Demo(Environment):
    sim_name = "example_sim_rigid_contact"
    env_offset=(2.0, 0.0, 2.0)
    tiny_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    xpbd_settings = dict(
        iterations=2,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.5,
        rigid_contact_relaxation=1.0,
        rigid_contact_con_weighting=True,
    )

    num_envs = 1

    def create_articulation(self, builder):

        self.num_bodies = 8
        self.scale = 0.8
        self.ke = 1.e+5
        self.kd = 250.0
        self.kf = 500.0

        # boxes
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((i, 1.0, 0.0), wp.quat_identity()))

            s = builder.add_shape_box( 
                pos=(0.0, 0.0, 0.0),
                hx=0.5*self.scale,
                hy=0.2*self.scale,
                hz=0.2*self.scale,
                body=i,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf)

        # spheres
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((i, 1.0, 2.0), wp.quat_identity()))

            s = builder.add_shape_sphere(
                pos=(0.0, 0.0, 0.0),
                radius=0.25*self.scale, 
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf)

        # capsules
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform((i, 1.0, 6.0), wp.quat_identity()))

            s = builder.add_shape_capsule( 
                pos=(0.0, 0.0, 0.0),
                radius=0.25*self.scale,
                half_height=self.scale*0.5,
                up_axis=0,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf)

        # initial spin 
        for i in range(len(builder.body_qd)):
            builder.body_qd[i] = (0.0, 2.0, 10.0, 0.0, 0.0, 0.0)

        # meshes
        bunny = self.load_mesh(os.path.join(os.path.dirname(__file__), "assets/bunny.usd"), "/bunny/bunny")
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform(
                (i*0.5*self.scale, 1.0 + i*1.7*self.scale, 4.0 + i*0.5*self.scale),
                wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.1*i)))

            s = builder.add_shape_mesh(
                    body=b,
                    mesh=bunny,
                    pos=(0.0, 0.0, 0.0),
                    scale=(self.scale, self.scale, self.scale),
                    ke=self.ke,
                    kd=self.kd,
                    kf=self.kf,
                    density=1e3,
                )

    def load_mesh(self, filename, path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()
        
        return wp.sim.Mesh(points, indices)


if __name__ == "__main__":
    run_env(Demo)
