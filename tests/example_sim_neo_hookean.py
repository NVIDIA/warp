# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Neo-Hookean
#
# Shows a simulation of an Neo-Hookean FEM beam being twisted through a
# 180 degree rotation.
#
###########################################################################

import os
import sys
import math

# include parent path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxr import Usd, UsdGeom, Gf, Sdf

import warp as wp
import warp.sim
import warp.sim.render

import numpy as np

wp.init()

# params
sim_width = 8
sim_height = 8

sim_fps = 60.0
sim_substeps = 64
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0
sim_render = True
sim_iterations = 1
sim_relaxation = 1.0

device = wp.get_preferred_device()

builder = wp.sim.ModelBuilder()

cell_dim = 15
cell_size = 2.0/cell_dim

center = cell_size*cell_dim*0.5

builder.add_soft_grid(
    pos=(-center, 0.0, -center), 
    rot=wp.quat_identity(), 
    vel=(0.0, 0.0, 0.0), 
    dim_x=cell_dim, 
    dim_y=cell_dim, 
    dim_z=cell_dim,
    cell_x=cell_size, 
    cell_y=cell_size,
    cell_z=cell_size,
    density=100.0, 
    fix_bottom=True,
    fix_top=True,
    k_mu=1000.0, 
    k_lambda=5000.0,
    k_damp=0.0)

model = builder.finalize(device=device)
model.ground = False
model.gravity[1] = 0.0

integrator = wp.sim.SemiImplicitIntegrator()

rest = model.state()
rest_vol = (cell_size*cell_dim)**3

state_0 = model.state()
state_1 = model.state()

stage = wp.sim.render.SimRenderer(model, "tests/outputs/example_sim_neo_hookean_twist.usd")

@wp.kernel
def twist_points(rest: wp.array(dtype=wp.vec3),
                 points: wp.array(dtype=wp.vec3),
                 mass: wp.array(dtype=float),
                 xform: wp.transform):

    tid = wp.tid()

    r = rest[tid]
    p = points[tid]
    m = mass[tid]

    # twist the top layer of particles in the beam
    if (m == 0 and p[1] != 0.0):
        points[tid] = wp.transform_point(xform, r)

@wp.kernel
def compute_volume(points: wp.array(dtype=wp.vec3),
                 indices: wp.array(dtype=int),
                 volume: wp.array(dtype=float)):

    tid = wp.tid()

    i = indices[tid * 4 + 0]
    j = indices[tid * 4 + 1]
    k = indices[tid * 4 + 2]
    l = indices[tid * 4 + 3]

    x0 = points[i]
    x1 = points[j]
    x2 = points[k]
    x3 = points[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v = wp.dot(x10, wp.cross(x20, x30))/6.0

    wp.atomic_add(volume, 0, v)


volume = wp.zeros(1, dtype=wp.float32, device=device)

lift_speed = 2.5/sim_duration*2.0 # from Smith et al.
rot_speed = math.pi/sim_duration

with wp.ScopedTimer("Total"):
    
    for i in range(sim_frames):
        
        with wp.ScopedTimer("simulate"):

            xform = (*(0.0, lift_speed*sim_time, 0.0), *wp.quat_from_axis_angle((0.0, 1.0, 0.0), rot_speed*sim_time))
            wp.launch(kernel=twist_points, dim=len(state_0.particle_q), inputs=[rest.particle_q, state_0.particle_q, model.particle_mass, xform], device=device)

            for s in range(sim_substeps):

                state_0.clear_forces()
                state_1.clear_forces()

                integrator.simulate(model, state_0, state_1, sim_dt)
                sim_time += sim_dt

                # swap states
                (state_0, state_1) = (state_1, state_0)

            volume.zero_()
            wp.launch(kernel=compute_volume, dim=model.tet_count, inputs=[state_0.particle_q, model.tet_indices, volume], device=device)

        if (sim_render):
        
            with wp.ScopedTimer("render"):

                stage.begin_frame(sim_time)
                stage.render(state_0)
                stage.end_frame()

stage.save()
