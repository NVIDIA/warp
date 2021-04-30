# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import oglang as og

import render

@og.kernel
def simulate(positions: og.array(og.vec3),
            velocities: og.array(og.vec3),
            mesh: og.int64,
            dt: float):
    
    tid = og.tid()

    x = og.load(positions, tid)
    v = og.load(velocities, tid)

    v = v + og.vec3(0.0, 0.0-9.8, 0.0)*dt
    xpred = x + v*dt

    if (xpred[1] < 0.0):
        v = og.vec3(v[0], 0.0 - v[1]*0.5, v[2])

    x = x + v*dt

    og.store(positions, tid, x)    
    og.store(velocities, tid, v)


# create og mesh
device = "cuda"

num_particles = 1000

sim_steps = 1000
sim_dt = 1.0/60.0

sim_time = 0.0

#mesh = og.Mesh(points, tris, device=device)


init_pos = np.random.rand(num_particles, 3) + np.array([0.0, 2.0, 0.0])
init_vel = np.random.rand(num_particles, 3)*5.0

positions = og.from_numpy(init_pos.astype(np.float32), dtype=og.vec3, device=device)
velocities = og.from_numpy(init_vel.astype(np.float32), dtype=og.vec3, device=device)

positions_host = og.from_numpy(init_pos.astype(np.float32), dtype=og.vec3, device="cpu")

stage = render.UsdRenderer("tests/outputs/test_mesh.usd")

for i in range(sim_steps):

    with og.ScopedTimer("simulate", detailed=False):

        og.launch(kernel=simulate, dim=num_particles, inputs=[positions, velocities, 0, sim_dt], outputs=[], device=device)
    
    # render
    with og.ScopedTimer("render", detailed=False):

        og.copy(positions_host, positions)

        stage.begin_frame(sim_time)
        
        stage.render_ground()
        stage.render_points(name="points", points=positions_host.numpy(), radius=0.01)

        stage.end_frame()

    sim_time += sim_dt

stage.save()


