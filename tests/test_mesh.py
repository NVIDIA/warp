# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import oglang as og
import render

np.random.seed(42)

og.init()

@og.kernel
def deform(positions: og.array(dtype=og.vec3), t: float):
    
    tid = og.tid()

    x = og.load(positions, tid)
    
#    a = 2.0 + 3.0

    offset = -sin(x[0])*0.01
    scale = sin(t)

    x = x + og.vec3(0.0, offset*scale, 0.0)

    og.store(positions, tid, x)


@og.kernel
def simulate(positions: og.array(dtype=og.vec3),
            velocities: og.array(dtype=og.vec3),
            mesh: og.uint64,
            restitution: float,
            margin: float,
            dt: float):
    
    tid = og.tid()

    x = og.load(positions, tid)
    v = og.load(velocities, tid)

    v = v + og.vec3(0.0, 0.0-9.8, 0.0)*dt - v*0.1*dt
    xpred = x + v*dt

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 1.5
    
    if (og.mesh_query_point(mesh, xpred, max_dist, sign, face_index, face_u, face_v)):
        
        p = og.mesh_eval_position(mesh, face_index, face_u, face_v)

        delta = xpred-p
        
        dist = og.length(delta)*sign
        err = dist - margin

        # mesh collision
        if (err < 0.0):
            n = og.normalize(delta)*sign
            xpred = xpred - n*err

        # # ground collision
        # if (xpred[1] < margin):
        #     xpred = og.vec3(xpred[0], margin, xpred[2])

    # pbd update
    v = (xpred - x)*(1.0/dt)
    x = xpred

    og.store(positions, tid, x)
    og.store(velocities, tid, v)


device = "cpu"
num_particles = 1000

sim_steps = 500
sim_dt = 1.0/60.0

sim_time = 0.0
sim_timers = {}
sim_render = True

sim_restitution = 0.0
sim_margin = 0.1

from pxr import Usd, UsdGeom, Gf, Sdf

torus = Usd.Stage.Open("./tests/assets/suzanne.usda")
torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/World/model/Suzanne"))

points = np.array(torus_geom.GetPointsAttr().Get())
indices = np.array(torus_geom.GetFaceVertexIndicesAttr().Get())

# create og mesh
mesh = og.Mesh(
    points=og.array(points, dtype=og.vec3, device=device),
    velocities=None,
    indices=og.array(indices, dtype=int, device=device))

init_pos = (np.random.rand(num_particles, 3) - np.array([0.5, -0.2, 0.5]))*10.0
init_vel = np.random.rand(num_particles, 3)*0.0

positions = og.from_numpy(init_pos.astype(np.float32), dtype=og.vec3, device=device)
velocities = og.from_numpy(init_vel.astype(np.float32), dtype=og.vec3, device=device)

positions_host = og.from_numpy(init_pos.astype(np.float32), dtype=og.vec3, device="cpu")

if (sim_render):
    stage = render.UsdRenderer("tests/outputs/test_mesh.usd")

for i in range(sim_steps):

    with og.ScopedTimer("simulate", detailed=False, dict=sim_timers):

        og.launch(
            kernel=deform,
            dim=len(mesh.points),
            inputs=[mesh.points, sim_time],
            device=device)

        mesh.refit()

        og.launch(
            kernel=simulate, 
            dim=num_particles, 
            inputs=[positions, velocities, mesh.id, sim_restitution, sim_margin, sim_dt], 
            device=device)

        og.synchronize()
    
    # render
    if (sim_render):

        with og.ScopedTimer("render", detailed=False):

            og.copy(positions_host, positions)

            stage.begin_frame(sim_time)

            stage.render_mesh(name="mesh", points=mesh.points.to("cpu").numpy(), indices=mesh.indices.to("cpu").numpy())
            stage.render_points(name="points", points=positions_host.numpy(), radius=sim_margin)

            stage.end_frame()

    sim_time += sim_dt

if (sim_render):
    stage.save()

print(np.mean(sim_timers["simulate"]))
print(np.min(sim_timers["simulate"]))
print(np.max(sim_timers["simulate"]))
