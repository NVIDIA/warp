# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import warp.render

wp.init()

@wp.kernel
def deform(positions: wp.array(dtype=wp.vec3), t: float):
    
    tid = wp.tid()

    x = positions[tid]
    
    offset = -wp.sin(x[0])*0.02
    scale = wp.sin(t)

    x = x + wp.vec3(0.0, offset*scale, 0.0)

    positions[tid] = x


@wp.kernel
def simulate(positions: wp.array(dtype=wp.vec3),
            velocities: wp.array(dtype=wp.vec3),
            mesh: wp.uint64,
            restitution: float,
            margin: float,
            dt: float):
    
    
    tid = wp.tid()

    x = positions[tid]
    v = velocities[tid]

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

    positions[tid] = x
    velocities[tid] = v


num_particles = 1000

sim_steps = 500
sim_dt = 1.0/60.0

sim_time = 0.0
sim_timers = {}
sim_render = True

sim_restitution = 0.0
sim_margin = 0.1

device = wp.get_preferred_device()

from pxr import Usd, UsdGeom, Gf, Sdf

usd_stage = Usd.Stage.Open("./tests/assets/bunny.usd")
usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/bunny/bunny")) 
usd_scale = 10.0

# create collision mesh
mesh = wp.Mesh(
    points=wp.array(usd_geom.GetPointsAttr().Get()*usd_scale, dtype=wp.vec3, device=device),
    indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int, device=device))
 
# random particles
init_pos = (np.random.rand(num_particles, 3) - np.array([0.5, -1.5, 0.5]))*10.0
init_vel = np.random.rand(num_particles, 3)*0.0

positions = wp.from_numpy(init_pos, dtype=wp.vec3, device=device)
velocities = wp.from_numpy(init_vel, dtype=wp.vec3, device=device)

if (sim_render):
    stage = warp.render.UsdRenderer("tests/outputs/example_mesh.usd")

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

            stage.begin_frame(sim_time)

            stage.render_mesh(name="mesh", points=mesh.points.numpy(), indices=mesh.indices.numpy())
            stage.render_points(name="points", points=positions.numpy(), radius=sim_margin)

            stage.end_frame()

    sim_time += sim_dt

if (sim_render):
    stage.save()

