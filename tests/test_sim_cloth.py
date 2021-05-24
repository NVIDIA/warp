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
sim_width = 64
sim_height = 32

sim_fps = 60.0
sim_substeps = 2
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0
sim_render = True

device = "cpu"

builder = ogsim.ModelBuilder()

builder.add_cloth_grid(
    pos=(0.0, 3.0, 0.0), 
    rot=og.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi*0.5), 
    vel=(0.0, 0.0, 0.0), 
    dim_x=sim_width, 
    dim_y=sim_height, 
    cell_x=0.1, 
    cell_y=0.1, 
    mass=0.1, 
    fix_left=True)


from pxr import Usd, UsdGeom, Gf, Sdf

torus = Usd.Stage.Open("./tests/assets/suzanne_small.usda")
torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/Suzanne/Suzanne"))

# torus = Usd.Stage.Open("./tests/assets/suzanne.usda")
# torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/World/model/Suzanne"))

# torus = Usd.Stage.Open("./tests/assets/suzanne_two.usda")
# torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/World/model/Suzanne"))


#torus = Usd.Stage.Open("./tests/assets/bunny.usda")
#torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/bunny/bunny"))

#torus = Usd.Stage.Open("./tests/assets/sphere_high.usda")
#torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/Icosphere/Icosphere"))

points = np.array(torus_geom.GetPointsAttr().Get())
indices = np.array(torus_geom.GetFaceVertexIndicesAttr().Get())

mesh = ogsim.Mesh(points, indices)

builder.add_shape_mesh(
    body=-1,
    mesh=mesh,
    pos=(0.0, 0.0, 0.0),
    rot=og.quat_identity(),
    scale=(1.0, 1.0, 1.0),
    ke=1.e+2,
    kd=1.e+2,
    kf=1.e+1)

#builder.add_shape_sphere(body=-1,)
#builder.add_shape_box(body=-1)

model = builder.finalize(device=device)
model.ground = True
model.tri_ke = 1.e+3
model.tri_ka = 1.e+3
model.tri_kb = 1.0
model.tri_kd = 1.e+1

model.contact_kd = 1.e+2

# disable cloth
#model.edge_count = 0
#model.tri_count = 0

#integrator = ogsim.SemiImplicitIntegrator()
integrator = ogsim.VariationalImplicitIntegrator(model)


state_0 = model.state()
state_1 = model.state()

stage = render.UsdRenderer("tests/outputs/test_sim_cloth.usd")

for i in range(sim_frames):
    
    with og.ScopedTimer("simulate"):

        for s in range(sim_substeps):

            integrator.simulate(model, state_0, state_1, sim_dt)
            sim_time += sim_dt

            # swap states
            (state_0, state_1) = (state_1, state_0)

        og.synchronize()

    if (sim_render):

        with og.ScopedTimer("render"):

            stage.begin_frame(sim_time)
            #stage.render_mesh(name="cloth", points=state.particle_q.to("cpu").numpy(), indices=model.tri_indices.to("cpu").numpy())
            stage.render_points(name="points", points=state_0.particle_q.to("cpu").numpy(), radius=0.1)
            
            # render static geometry once
            if (i == 0):
                #stage.render_box(name="box", pos=(0.0, 0.0, 0.0), extents=(0.5, 0.5, 0.5))
                #stage.render_sphere(name="sphere", pos=(0.0, 0.0, 0.0), radius=1.0)
                stage.render_mesh(name="mesh", points=points, indices=indices)

            stage.end_frame()


# history = np.genfromtxt("test_history.txt", delimiter=",")
# history_indices = []

# stage.begin_frame(sim_time)

# i = 0
# for line in history:
#     tri = int(line[0])

#     i = indices[tri*3+0]
#     j = indices[tri*3+1]
#     k = indices[tri*3+2]

#     history_indices.append(int(i))
#     history_indices.append(int(j))
#     history_indices.append(int(k))

#     p = (line[1], line[2], line[3])
#     e = (line[4]*0.5, line[5]*0.5, line[6]*0.5)

#     stage.render_box(name="box_" + str(i), pos=p, extents=e)

#     i += 1

# stage.render_mesh(name="history", points=points, indices=history_indices)
# stage.render_sphere(name="worst", pos=(0.307443, -0.668166, 0.270352), radius=0.2)
# stage.end_frame()



stage.save()
