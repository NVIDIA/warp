# include parent path
import os
import sys
from warp.utils import quat_identity
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import warp.render

np.random.seed(42)

wp.init()

@wp.func
def volume_grad(volume: wp.uint64,
                p: wp.vec3):
    
    eps = 1.e-2

    dx = wp.volume_sample_world(volume, p + wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR) - wp.volume_sample_world(volume, p - wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR)
    dy = wp.volume_sample_world(volume, p + wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR) - wp.volume_sample_world(volume, p - wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR)
    dz = wp.volume_sample_world(volume, p + wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR) - wp.volume_sample_world(volume, p - wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR)

    return wp.normalize(wp.vec3(dx, dy, dz))

@wp.kernel
def simulate(positions: wp.array(dtype=wp.vec3),
            velocities: wp.array(dtype=wp.vec3),
            volume: wp.uint64,
            restitution: float,
            margin: float,
            dt: float):
    
    
    tid = wp.tid()

    x = wp.load(positions, tid)
    v = wp.load(velocities, tid) 

    v = v + wp.vec3(0.0, 0.0, -980.0)*dt - v*0.1*dt
    xpred = x + v*dt

    d = wp.volume_sample_world(volume, xpred, wp.Volume.LINEAR)

    if (d < margin):
        
        n = volume_grad(volume, xpred)
        err = d - margin

        # mesh collision
        xpred = xpred - n*err

 
    # ground collision
    if (xpred[2] < 0.0):
        xpred = wp.vec3(xpred[0], xpred[1], 0.0)

    # pbd update
    v = (xpred - x)*(1.0/dt)
    x = xpred

    wp.store(positions, tid, x)
    wp.store(velocities, tid, v)


device = "cuda"
num_particles = 10000

sim_steps = 1000
sim_dt = 1.0/60.0
sim_substeps = 3

sim_time = 0.0
sim_timers = {}
sim_render = True

sim_restitution = 0.0
sim_margin = 15.0

from pxr import Usd, UsdGeom, Gf, Sdf

init_pos = 1000.0*(np.random.rand(num_particles, 3)*2.0 - 1.0) + np.array((0.0, 0.0, 3000.0))
init_vel = np.random.rand(num_particles, 3)

positions = wp.from_numpy(init_pos.astype(np.float32), dtype=wp.vec3, device=device)
velocities = wp.from_numpy(init_vel.astype(np.float32), dtype=wp.vec3, device=device)
positions_host = wp.from_numpy(init_pos.astype(np.float32), dtype=wp.vec3, device="cpu")

# load collision volume
file = np.fromfile("C:/Dev/usd/rocks/rocks.nvdb_grid", dtype=np.byte)

# create Volume object
volume = wp.Volume(wp.array(file, device=device))

if (sim_render):
    stage = warp.render.UsdRenderer("C:/Dev/usd/rocks/example_nvdb.usd")    

for i in range(sim_steps):

    with wp.ScopedTimer("simulate", detailed=False, dict=sim_timers):

        for s in range(sim_substeps):
            wp.launch(
                kernel=simulate, 
                dim=num_particles, 
                inputs=[positions, velocities, volume.id, sim_restitution, sim_margin, sim_dt/float(sim_substeps)], 
                device=device)

        wp.synchronize()
    
    # render
    if (sim_render):

        with wp.ScopedTimer("render", detailed=False):

            wp.copy(positions_host, positions)

            stage.begin_frame(sim_time)

            stage.render_ref(name="collision", path="./Rock.usd", pos=(0.0, 0.0, 0.0), rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi), scale=(1.0, 1.0, 1.0))
            stage.render_points(name="points", points=positions_host.numpy(), radius=sim_margin)

            stage.end_frame()

    sim_time += sim_dt

if (sim_render):
    stage.save()

print(np.mean(sim_timers["simulate"]))
print(np.min(sim_timers["simulate"]))
print(np.max(sim_timers["simulate"]))
