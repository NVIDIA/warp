# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxr import Usd, UsdGeom, Gf, Sdf

import oglang as og
import oglang.sim as ogsim

import render

# params
sim_width = 128
sim_height = 128

sim_fps = 60.0
sim_substeps = 32
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0

device = "cpu"

integrator = ogsim.SemiImplicitIntegrator()

builder = ogsim.ModelBuilder()

builder.add_cloth_grid(
    pos=(0.0, 3.0, 0.0), 
    rot=og.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi*0.5), 
    vel=(0.0, 0.0, 0.0), 
    dim_x=30, 
    dim_y=30, 
    cell_x=0.1, 
    cell_y=0.1, 
    mass=0.1, 
    fix_left=True)

from pxr import Usd, UsdGeom, Gf, Sdf

torus = Usd.Stage.Open("./tests/assets/suzanne.usda")
torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/World/model/Suzanne"))

points = np.array(torus_geom.GetPointsAttr().Get())
indices = np.array(torus_geom.GetFaceVertexIndicesAttr().Get())

mesh = ogsim.Mesh(points, indices)

builder.add_shape_mesh(
    body=-1,
    mesh=mesh,
    pos=(0.0, 0.0, 0.0),
    rot=og.quat_identity(),
    ke=1.e+3,
    kd=1.e+2,
    kf=1.e+2)


model = builder.finalize(adapter=device)
model.ground = False
model.tri_ke = 1.e+4
model.tri_kb = 1.e+1

state = model.state()


stage = render.UsdRenderer("tests/outputs/test_sim_cloth.usd")

for i in range(sim_frames):
    
    with og.ScopedTimer("simulate"):

        for s in range(sim_substeps):

            integrator.simulate(model, state, state, sim_dt)
            sim_time += sim_dt

    with og.ScopedTimer("render"):

        stage.begin_frame(sim_time)
        stage.render_mesh(name="cloth", points=state.particle_q.to("cpu").numpy(), indices=model.tri_indices.to("cpu").numpy())
        stage.render_mesh(name="mesh", points=points, indices=indices)
        stage.end_frame()


stage.save()
