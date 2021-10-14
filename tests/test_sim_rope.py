# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxr import Usd, UsdGeom, Gf, Sdf

import warp as wp
import warp.sim
import warp.sim.render as render


wp.init()
#wp.config.verify_cuda = True

# params
sim_width = 64
sim_height = 32

sim_fps = 60.0
sim_substeps = 32
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0
sim_render = True

device = "cpu"
 
builder = wp.sim.ModelBuilder()

# anchor
builder.add_particle((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

# chain
for i in range(1, 10):
    builder.add_particle((i, 1.0, 0.0), (0.0, 0., 0.0), 1.0)
    builder.add_spring(i - 1, i, 1.e+6, 0.0, 0)

model = builder.finalize(device=device)
model.ground = False

integrator = wp.sim.SemiImplicitIntegrator()

state_0 = model.state()
state_1 = model.state()

renderer = render.SimRenderer(model, "tests/outputs/test_sim_rope.usd")

# launch simulation
for i in range(sim_frames):
    
    with wp.ScopedTimer("simulate", active=True, detailed=False):

        for s in range(sim_substeps):

            state_0.clear_forces()
            state_1.clear_forces()

            integrator.simulate(model, state_0, state_1, sim_dt)
            sim_time += sim_dt

            # swap states
            (state_0, state_1) = (state_1, state_0)

    if (sim_render):

        with wp.ScopedTimer("render", active=False):

            renderer.begin_frame(sim_time)
            renderer.render(state_0)
            renderer.end_frame()

renderer.save()
