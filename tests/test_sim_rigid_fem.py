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

from warp.utils import quat_identity

wp.init()

import render

np.random.seed(42)
np.set_printoptions(threshold=sys.maxsize)

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

device = "cuda"

builder = wpsim.ModelBuilder()


builder.add_soft_grid(
    pos=(0.0, 0.0, 0.0), 
    rot=wp.quat_identity(), 
    vel=(0.0, 0.0, 0.0), 
    dim_x=20, 
    dim_y=10, 
    dim_z=10,
    cell_x=0.1, 
    cell_y=0.1,
    cell_z=0.1,
    density=100.0, 
    k_mu=50000.0, 
    k_lambda=20000.0,
    k_damp=0.0)

builder.add_body(X_pj=wp.transform((0.5, 2.5, 0.5), wp.quat_identity()))
builder.add_shape_sphere(body=0, radius=0.75, density=100.0)

model = builder.finalize(device=device)
model.ground = True
model.soft_contact_distance = 0.01
model.soft_contact_ke = 1.e+3
model.soft_contact_kd = 0.0
model.soft_contact_kf = 1.e+3

integrator = wpsim.SemiImplicitIntegrator()

state_0 = model.state()
state_1 = model.state()

model.collide(state_0)

stage = render.UsdRenderer("tests/outputs/test_sim_rigid_fem.usd")

for i in range(sim_frames):
    
    if (sim_render):
    
        with wp.ScopedTimer("render"):

            stage.begin_frame(sim_time)
            stage.render_ground()
            stage.render_mesh(name="fem", points=state_0.particle_q.to("cpu").numpy(), indices=model.tri_indices.to("cpu").numpy())
            
            body_q = state_0.body_q.to("cpu").numpy()
            stage.render_sphere(pos=body_q[0, 0:3], rot=body_q[0, 3:7], radius=float(model.shape_geo_scale.to("cpu").numpy()[0,0]), name="ball")
            stage.end_frame()

    with wp.ScopedTimer("simulate"):

        for s in range(sim_substeps):

            wp.sim.collide(model, state_0)

            state_0.clear_forces()
            state_1.clear_forces()

            integrator.simulate(model, state_0, state_1, sim_dt)
            sim_time += sim_dt

            # swap states
            (state_0, state_1) = (state_1, state_0)

        wp.synchronize()


stage.save()
