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

builder.add_body(X_pj=wp.transform((0.0, 2.0, 0.0), wp.quat_identity()))
builder.add_shape_box(body=0, hx=0.5, hy=0.5, hz=0.5, density=1000.0, ke=2.e+5, kd=1.e+4)

model = builder.finalize(device=device)
model.ground = True

integrator = wpsim.SemiImplicitIntegrator()

state_0 = model.state()
state_1 = model.state()

model.collide(state_0)

stage = render.UsdRenderer("tests/outputs/test_sim_rigid_force.usd")

for i in range(sim_frames):
    
    if (sim_render):
    
        with wp.ScopedTimer("render"):

            stage.begin_frame(sim_time)
            stage.render_ground()
            #stage.render_mesh(name="fem", points=state_0.particle_q.to("cpu").numpy(), indices=model.tri_indices.to("cpu").numpy())
            
            body_q = state_0.body_q.to("cpu").numpy()
            #stage.render_sphere(pos=body_q[0, 0:3], rot=body_q[0, 3:7], radius=float(model.shape_geo_scale.to("cpu").numpy()[0,0]), name="ball")
            stage.render_box(pos=body_q[0, 0:3], rot=body_q[0, 3:7], extents=model.shape_geo_scale.to("cpu").numpy()[0,0:3], name="box")
            stage.end_frame()

    with wp.ScopedTimer("simulate"):

        for s in range(sim_substeps):

            wp.sim.collide(model, state_0)

            state_0.clear_forces()
            state_1.clear_forces()

            state_0.body_f.assign([ [0.0, 0.0, -3000.0, 0.0, 0.0, 0.0], ])
            #state_0.body_f = wp.array([ [0.0, 0.0, -3000.0, 0.0, 0.0, 0.0], ], dtype=wp.spatial_vector, device=device)

            integrator.simulate(model, state_0, state_1, sim_dt)
            sim_time += sim_dt

            # swap states
            (state_0, state_1) = (state_1, state_0)

        wp.synchronize()


stage.save()
