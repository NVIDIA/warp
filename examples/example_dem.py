# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example DEM
#
# Shows how to implement a DEM particle simulation with cohesion between
# particles. Neighbors are found using the wp.HashGrid class, and
# wp.hash_grid_query(), wp.hash_grid_query_next() kernel methods.
#
###########################################################################

import numpy as np

import warp as wp
import warp.render 
from pxr import Usd

import os

wp.init()

@wp.func 
def contact_force(n: wp.vec3,
                       v: wp.vec3,
                       c: float,
                       k_n: float,
                       k_d: float,
                       k_f: float,
                       k_mu: float):
    vn = wp.dot(n, v)
    jn = c*k_n
    jd = min(vn, 0.0)*k_d

    # contact force
    fn = jn + jd

    # friction force
    vt = v - n*vn
    vs = wp.length(vt)
    
    if (vs > 0.0):
        vt = vt/vs

    # Coulomb condition
    ft = wp.min(vs*k_f, k_mu*wp.abs(fn))

    # total force
    return  -n*fn - vt*ft

@wp.kernel
def apply_forces(grid : wp.uint64,
                 particle_x: wp.array(dtype=wp.vec3),
                 particle_v: wp.array(dtype=wp.vec3),
                 particle_f: wp.array(dtype=wp.vec3),
                 radius: float,
                 k_contact: float,
                 k_damp: float,
                 k_friction: float,
                 k_mu: float):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x = particle_x[i]
    v = particle_v[i]

    f = wp.vec3()

    # ground contact
    n = wp.vec3(0.0, 1.0, 0.0)
    c = wp.dot(n, x)

    cohesion_ground = 0.02
    cohesion_particle = 0.0075

    if c < cohesion_ground:
        f = f + contact_force(n, v, c, k_contact, k_damp, 100.0, 0.5)

    # particle contact
    query = wp.hash_grid_query(grid, x, radius*5.0)
    index = int(0)

    while(wp.hash_grid_query_next(query, index)):

        if index != i:
            
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius*2.0

            if (err <= cohesion_particle):
                
                n = n/d
                vrel = v - particle_v[index]

                f = f + contact_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu)

    particle_f[i] = f

@wp.kernel
def integrate(x: wp.array(dtype=wp.vec3),
              v: wp.array(dtype=wp.vec3),
              f: wp.array(dtype=wp.vec3),
              gravity: wp.vec3,
              dt: float,
              inv_mass: float):

    tid = wp.tid()

    v_new = v[tid] + f[tid]*inv_mass*dt + gravity*dt
    x_new = x[tid] + v_new*dt

    v[tid] = v_new
    x[tid] = x_new  

# creates a grid of particles
def particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter):
    points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
    points_t = np.array((points[0], points[1], points[2])).T*radius*2.0 + np.array(lower)
    points_t = points_t + np.random.rand(*points_t.shape)*radius*jitter
    
    return points_t.reshape((-1, 3))


class Example:
    frame_dt = 1.0/60
    frame_count = 400

    sim_substeps = 64
    sim_dt = frame_dt/sim_substeps
    sim_steps = frame_count*sim_substeps
    sim_time = 0.0

    device = wp.get_preferred_device()

    point_radius = 0.1

    grid = wp.HashGrid(128, 128, 128, device)
    grid_cell_size = point_radius*5.0

    points = particle_grid(32, 128, 32, (0.0, 0.3, 0.0), point_radius, 0.1)

    k_contact = 8000.0
    k_damp = 2.0
    k_friction = 1.0
    k_mu = 100000.0 # for cohesive materials

    inv_mass = 64.0

    def init(self, stage):
        self.renderer = wp.render.UsdRenderer(stage)
        self.renderer.render_ground()

        self.x = wp.array(Example.points, dtype=wp.vec3, device=Example.device)
        self.v = wp.array(np.ones([len(self.x), 3])*np.array([0.0, 0.0, 10.0]), dtype=wp.vec3, device=Example.device)
        self.f = wp.zeros_like(self.v)

        self.use_graph = (Example.device == "cuda")

        if (self.use_graph):

            wp.capture_begin()

            for s in range(Example.sim_substeps):

                with wp.ScopedTimer("forces", active=False):
                    wp.launch(kernel=apply_forces, dim=len(self.x), inputs=[Example.grid.id, self.x, self.v, self.f, Example.point_radius, Example.k_contact, Example.k_damp, Example.k_friction, Example.k_mu], device=Example.device)
                    wp.launch(kernel=integrate, dim=len(self.x), inputs=[self.x, self.v, self.f, (0.0, -9.8, 0.0), Example.sim_dt, Example.inv_mass], device=Example.device)
                
            self.graph = wp.capture_end()

    def update(self):
        with wp.ScopedTimer("simulate", active=True):

            if (self.use_graph):

                with wp.ScopedTimer("grid build", active=False):
                    Example.grid.build(self.x, Example.grid_cell_size)

                with wp.ScopedTimer("solve", active=False):
                    wp.capture_launch(self.graph)
                    wp.synchronize()
                    

            else:
                for s in range(Example.sim_substeps):

                    with wp.ScopedTimer("grid build", active=False):
                        Example.grid.build(self.x, Example.point_radius)

                    with wp.ScopedTimer("forces", active=False):
                        wp.launch(kernel=apply_forces, dim=len(self.x), inputs=[Example.grid.id, self.x, self.v, self.f, Example.point_radius, Example.k_contact, Example.k_damp, Example.k_friction, Example.k_mu], device=Example.device)
                        wp.launch(kernel=integrate, dim=len(self.x), inputs=[self.x, self.v, self.f, (0.0, -9.8, 0.0), Example.sim_dt, Example.inv_mass], device=Example.device)
                
                wp.synchronize()

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else Example.sim_time 
            
            self.renderer.begin_frame(time)
            self.renderer.render_points(points=self.x.numpy(), radius=Example.point_radius, name="points")
            self.renderer.end_frame()

            Example.sim_time += Example.frame_dt


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), "outputs/example_dem.usd")
    stage = Usd.Stage.CreateNew(path)

    example = Example()
    example.init(stage)

    for i in range(example.frame_count):
        example.update()
        example.render()

    example.renderer.save()
