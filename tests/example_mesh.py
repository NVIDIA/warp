# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import warp.render

np.random.seed(42)

wp.init()

@wp.kernel
def deform(positions: wp.array(dtype=wp.vec3), t: float):
    
    tid = wp.tid()

    x = wp.load(positions, tid)
    
#    a = 2.0 + 3.0

    offset = -sin(x[0])*0.02
    scale = sin(t)

    x = x + wp.vec3(0.0, offset*scale, 0.0)

    wp.store(positions, tid, x)


@wp.kernel
def simulate(positions: wp.array(dtype=wp.vec3),
            velocities: wp.array(dtype=wp.vec3),
            mesh: wp.uint64,
            restitution: float,
            margin: float,
            dt: float):
    
    
    tid = wp.tid()

    x = wp.load(positions, tid)
    v = wp.load(velocities, tid)

    v = v + wp.vec3(0.0, 0.0-9.8, 0.0)*dt - v*0.1*dt
    xpred = x + v*dt

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 1.5
    
    if (wp.mesh_query_point(mesh, xpred, max_dist, sign, face_index, face_u, face_v)):
        
        p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

        delta = xpred-p
        
        dist = wp.length(delta)*sign
        err = dist - margin

        # mesh collision
        if (err < 0.0):
            n = wp.normalize(delta)*sign
            xpred = xpred - n*err

        # # ground collision
        # if (xpred[1] < margin):
        #     xpred = wp.vec3(xpred[0], margin, xpred[2])

    # pbd update
    v = (xpred - x)*(1.0/dt)
    x = xpred

    wp.store(positions, tid, x)
    wp.store(velocities, tid, v)


device = "cuda"
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

# torus = Usd.Stage.Open("./tests/assets/excavator.usda")
# torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/Excavator/Excavator"))

points = np.array(torus_geom.GetPointsAttr().Get())
indices = np.array(torus_geom.GetFaceVertexIndicesAttr().Get())

# create wp mesh
mesh = wp.Mesh(
    points=wp.array(points, dtype=wp.vec3, device=device),
    velocities=None,
    indices=wp.array(indices, dtype=int, device=device))

init_pos = (np.random.rand(num_particles, 3) - np.array([0.5, -0.2, 0.5]))*10.0
init_vel = np.random.rand(num_particles, 3)*0.0

positions = wp.from_numpy(init_pos.astype(np.float32), dtype=wp.vec3, device=device)
velocities = wp.from_numpy(init_vel.astype(np.float32), dtype=wp.vec3, device=device)

positions_host = wp.from_numpy(init_pos.astype(np.float32), dtype=wp.vec3, device="cpu")

if (sim_render):
    stage = render.UsdRenderer("tests/outputs/test_mesh.usd")

for i in range(sim_steps):

    with wp.ScopedTimer("simulate", detailed=False, dict=sim_timers):

        wp.launch(
            kernel=deform,
            dim=len(mesh.points),
            inputs=[mesh.points, sim_time],
            device=device)

        mesh.refit()

        wp.launch(
            kernel=simulate, 
            dim=num_particles, 
            inputs=[positions, velocities, mesh.id, sim_restitution, sim_margin, sim_dt], 
            device=device)

        wp.synchronize()
    
    # render
    if (sim_render):

        with wp.ScopedTimer("render", detailed=False):

            wp.copy(positions_host, positions)

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
