import os
import sys

from warp.render import UsdRenderer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import warp.sim

import numpy as np

np.random.seed(532)

wp.config.mode = "release"
wp.config.verify_cuda = False

wp.init()

frame_dt = 1.0/60
frame_count = 400

sim_substeps = 64
sim_dt = frame_dt/sim_substeps
sim_steps = frame_count*sim_substeps
sim_time = 0.0

radius = 0.1

device = "cuda"

builder = wp.sim.ModelBuilder()

builder.add_particle_grid(
    dim_x=16,
    dim_y=32,
    dim_z=16,
    cell_x=radius*2.0,
    cell_y=radius*2.0,
    cell_z=radius*2.0,
    pos=(0.0, 1.0, 0.0),
    rot=wp.quat_identity(),
    vel=(5.0, 0.0, 0.0),
    mass=0.1,
    jitter=radius*0.1)

model = builder.finalize(device)
model.particle_radius = radius
model.particle_kf = 25.0

model.soft_contact_kd = 100.0
model.soft_contact_kf *= 2.0

state_0 = model.state()
state_1 = model.state()

integrator = wp.sim.SemiImplicitIntegrator()

renderer = UsdRenderer("tests/outputs/test_sim_granular.usd")

for i in range(frame_count):

    with wp.ScopedTimer("simulate", active=True):

        model.particle_grid.build(state_0.particle_q, radius*2.0)

        for s in range(sim_substeps):

            state_0.clear_forces()
            state_1.clear_forces()

            integrator.simulate(model, state_0, state_1, sim_dt)
            sim_time += sim_dt

            # swap states
            (state_0, state_1) = (state_1, state_0)

        p = state_0.particle_q.numpy()

    with wp.ScopedTimer("render", active=True):
        renderer.begin_frame(sim_time)
        renderer.render_points(points=p, radius=radius, name="points")
        renderer.end_frame()

    sim_time += frame_dt

renderer.save()

