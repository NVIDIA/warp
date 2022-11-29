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
import warp.sim.render

wp.init()


class Example:

    def __init__(self, stage):

        self.sim_steps = 1000
        self.sim_dt = 1.0/60.0
        self.sim_time = 0.0
        self.sim_substeps = 8

        self.num_bodies = 8
        self.scale = 0.5
        self.ke = 1.e+5
        self.kd = 250.0
        self.kf = 500.0

        builder = wp.sim.ModelBuilder()

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
                half_width=self.scale*0.5,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf)

        # initial spin 
        for i in range(len(builder.body_qd)):
            builder.body_qd[i] = (0.0, 2.0, 10.0, 0.0, 0.0, 0.0)

        # meshes
        monkey = self.load_mesh(os.path.join(os.path.dirname(__file__), f"assets/monkey.obj"))
        for i in range(self.num_bodies):
            
            b = builder.add_body(origin=wp.transform(
                (i*0.5*self.scale, 1.0 + i*1.7*self.scale, 4.0 + i*0.5*self.scale),
                wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.1*i)))

            s = builder.add_shape_mesh(
                    body=b,
                    mesh=monkey,
                    pos=(0.0, 0.0, 0.0),
                    scale=(self.scale, self.scale, self.scale),
                    ke=self.ke,
                    kd=self.kd,
                    kf=self.kf,
                    density=1e3,
                )
        
        self.model = builder.finalize()
        self.model.ground = True

        self.integrator = wp.sim.XPBDIntegrator()
        self.state = self.model.state()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=30.0)

    def load_mesh(self, filename, use_meshio=False):
        if use_meshio:
            import meshio
            m = meshio.read(filename)
            mesh_points = np.array(m.points)
            mesh_indices = np.array(m.cells[0].data, dtype=np.int32).flatten()
        else:
            import openmesh
            m = openmesh.read_trimesh(filename)
            mesh_points = np.array(m.points())
            mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32).flatten()
        return wp.sim.Mesh(mesh_points, mesh_indices)

    def update(self):

        with wp.ScopedTimer("simulate", active=False):
            
            for i in range(self.sim_substeps):
                self.state.clear_forces()
                wp.sim.collide(self.model, self.state)
                self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt/self.sim_substeps)   

    def render(self, is_live=False):

        with wp.ScopedTimer("render", active=False):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state)
            self.renderer.end_frame()
        
        self.sim_time += self.sim_dt


if __name__ == '__main__':
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd")

    example = Example(stage_path)

    use_graph = True
    if use_graph:
        wp.capture_begin()
        example.update()
        graph = wp.capture_end()

    for i in range(example.sim_steps):
        if use_graph:
            wp.capture_launch(graph)
        else:
            example.update()
        example.render()

    example.renderer.save()





