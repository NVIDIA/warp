# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cloth
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the wp.sim.ModelBuilder().
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

from pxr import Usd, UsdGeom, Gf, Sdf

wp.init()

sim_width = 64
sim_height = 32

sim_fps = 60.0
sim_substeps = 32
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0
sim_render = True
sim_use_graph = True

device = wp.get_preferred_device()
 
builder = wp.sim.ModelBuilder()

builder.add_cloth_grid(
    pos=(0.0, 4.0, 0.0), 
    rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi*0.5), 
    vel=(0.0, 0.0, 0.0), 
    dim_x=sim_width, 
    dim_y=sim_height, 
    cell_x=0.1, 
    cell_y=0.1, 
    mass=0.1, 
    fix_left=True)


from pxr import Usd, UsdGeom, Gf, Sdf

usd_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets/bunny.usd"))
usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/bunny/bunny"))

mesh_points = np.array(usd_geom.GetPointsAttr().Get())
mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

mesh = wp.sim.Mesh(mesh_points, mesh_indices)

builder.add_shape_mesh(
    body=-1,
    mesh=mesh,
    pos=(1.0, 0.0, 1.0),
    rot=wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.5),
    scale=(2.0, 2.0, 2.0),
    ke=1.e+2,
    kd=1.e+2,
    kf=1.e+1)

model = builder.finalize(device=device)
model.ground = True
model.tri_ke = 1.e+3
model.tri_ka = 1.e+3
model.tri_kb = 1.0
model.tri_kd = 1.e+1
model.soft_contact_kd = 1.e+2

integrator = wp.sim.SemiImplicitIntegrator()

state_0 = model.state()
state_1 = model.state()

stage = wp.sim.render.SimRenderer(model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_cloth.usd"))

if (sim_use_graph):

    # create update graph
    wp.capture_begin()

    wp.sim.collide(model, state_0)

    for s in range(sim_substeps):

        state_0.clear_forces()

        integrator.simulate(model, state_0, state_1, sim_dt)

        # swap states
        (state_0, state_1) = (state_1, state_0)

    graph = wp.capture_end()


# launch simulation
for i in range(sim_frames):
    
    with wp.ScopedTimer("simulate", active=True):

        if (sim_use_graph):
            wp.capture_launch(graph)
        else:

            wp.sim.collide(model, state_0)

            for s in range(sim_substeps):

                state_0.clear_forces()

                integrator.simulate(model, state_0, state_1, sim_dt)

                # swap states
                (state_0, state_1) = (state_1, state_0)

    if (sim_render):

        with wp.ScopedTimer("render", active=True):

            stage.begin_frame(sim_time)
            stage.render(state_0)
            stage.end_frame()

    sim_time += 1.0/sim_fps


stage.save()
