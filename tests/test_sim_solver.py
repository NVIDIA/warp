# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxr import Usd, UsdGeom, Gf, Sdf

import warp as wp
import warp.sim as wpsim

wp.init()

import render

np.random.seed(42)
np.set_printoptions(threshold=sys.maxsize)

# params
sim_width = 8
sim_height = 8

sim_fps = 60.0
sim_substeps = 1
sim_duration = 5.0
sim_frames = 1#int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0
sim_render = True

device = "cpu"

builder = wpsim.ModelBuilder()

builder.add_cloth_grid(
    pos=(0.0, 3.0, 0.0), 
    rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi*0.5), 
    vel=(0.0, 0.0, 0.0), 
    dim_x=sim_width, 
    dim_y=sim_height, 
    cell_x=0.1, 
    cell_y=0.1, 
    mass=0.1, 
    fix_left=True)

# randomize initial positions to create some residual
for i in range(len(builder.particle_q)):
    if (builder.particle_mass[i] > 0.0):
        builder.particle_q[i] += np.random.rand(3)*0.1 
    else:
        builder.particle_q[i] += np.array((-1.0, 0.0, 0.0))


model = builder.finalize(device=device)
model.ground = True
model.tri_ke = 1.e+3
model.tri_ka = 1.e+3
model.tri_kb = 1.0
model.tri_kd = 1.e+1
model.contact_kd = 1.e+2

#integrator = wpsim.SemiImplicitIntegrator()
integrator = wpsim.VariationalImplicitIntegrator(model, solver="nesterov", max_iters=64, alpha=0.01, report=True)

state_0 = model.state()
state_1 = model.state()

stage = render.UsdRenderer("tests/outputs/test_sim_solver.usd")

for i in range(sim_frames):
    
    if (sim_render):
    
        with wp.ScopedTimer("render"):

            stage.begin_frame(sim_time)
            stage.render_mesh(name="cloth", points=state_0.particle_q.to("cpu").numpy(), indices=model.tri_indices.to("cpu").numpy())
            #stage.render_points(name="points", points=state_0.particle_q.to("cpu").numpy(), radius=0.1)
            #stage.render_

            stage.end_frame()

    with wp.ScopedTimer("simulate"):

        for s in range(sim_substeps):

            integrator.simulate(model, state_0, state_1, sim_dt)
            sim_time += sim_dt

            # swap states
            (state_0, state_1) = (state_1, state_0)

        wp.synchronize()




stage.save()
