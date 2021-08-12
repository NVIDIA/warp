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

import render

wp.init()

# params
sim_width = 128
sim_height = 128 

sim_fps = 60.0
sim_substeps = 32
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0
sim_render = True
sim_use_graph = False

device = "cuda"

builder = wp.sim.ModelBuilder()

builder.add_cloth_grid(
    pos=(0.0, 3.0, 0.0), 
    rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi*0.5), 
    vel=(0.0, 0.0, 0.0), 
    dim_x=sim_width, 
    dim_y=sim_height, 
    cell_x=0.1, 
    cell_y=0.1, 
    mass=0.1, 
    fix_left=False)


from pxr import Usd, UsdGeom, Gf, Sdf

#torus = Usd.Stage.Open("./tests/assets/suzanne_small.usda")
#torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/Suzanne/Suzanne"))

# torus = Usd.Stage.Open("./tests/assets/suzanne.usda")
# torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/World/model/Suzanne"))

torus = Usd.Stage.Open("./tests/assets/suzanne_two.usda")
torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/World/model/Suzanne"))


#torus = Usd.Stage.Open("./tests/assets/bunny.usda")
#torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/bunny/bunny"))

#torus = Usd.Stage.Open("./tests/assets/sphere_high.usda")
#torus_geom = UsdGeom.Mesh(torus.GetPrimAtPath("/Icosphere/Icosphere"))

points = np.array(torus_geom.GetPointsAttr().Get())
indices = np.array(torus_geom.GetFaceVertexIndicesAttr().Get())

mesh = wp.sim.Mesh(points, indices)

builder.add_shape_mesh(
    body=-1,
    mesh=mesh,
    pos=(5.0, -2.0, 7.0),
    rot=wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.5),
    scale=(4.0, 4.0, 4.0),
    ke=1.e+2,
    kd=1.e+2,
    kf=1.e+1)

#builder.add_shape_sphere(body=-1, pos=(1.0, 0.0, 1.0))
#builder.add_shape_box(body=-1)

model = builder.finalize(device=device)
model.ground = False
model.tri_ke = 1.e+4
model.tri_ka = 1.e+4
model.tri_kb = 1.0
model.tri_kd = 1.e+1

model.soft_contact_kd = 1.e+2
#model.soft_contact_margin = 3.0
#model.soft_contact_ke = 

integrator = wp.sim.SemiImplicitIntegrator()
#integrator = wp.sim.VariationalImplicitIntegrator(model)


state_0 = model.state()
state_1 = model.state()

stage = render.UsdRenderer("tests/outputs/test_sim_cloth.usd")

if (sim_use_graph):
    # create update graph
    wp.capture_begin()

    wp.sim.collide(model, state_0)

    for s in range(sim_substeps):

        state_0.clear_forces()
        state_1.clear_forces()

        integrator.simulate(model, state_0, state_1, sim_dt)
        sim_time += sim_dt

        # swap states
        (state_0, state_1) = (state_1, state_0)

    graph = wp.capture_end()


# launch simulation
for i in range(sim_frames):
    
    with wp.ScopedTimer("simulate", active=True):

        if (sim_use_graph):
            wp.capture_launch(graph)
            sim_time += 1.0/60.0
        else:

            wp.sim.collide(model, state_0)

            for s in range(sim_substeps):

                integrator.simulate(model, state_0, state_1, sim_dt)
                sim_time += sim_dt

                # swap states
                (state_0, state_1) = (state_1, state_0)

    # if (i == sim_frames-1):

    #     xform = builder.shape_transform[0]

    #     # write collider mesh
    #     collider = open("tests/outputs/ingo_suzanne.obj", "w")
        
    #     for v in mesh.vertices:
    #         s = builder.shape_geo_scale
    #         p = wp.transform_point(xform, v*s[0])#(v[0]*s[0], v[1]*s[1], v[2]*s[2]))
            
    #         collider.write("v {} {} {}\n".format(p[0], p[1], p[2]))
        
    #     for f in range(0, len(mesh.indices), 3):
    #         collider.write("f {} {} {}\n".format(mesh.indices[f+0] + 1, mesh.indices[f+1] + 1, mesh.indices[f+2] + 1))

    #     collider.close()

    #     # write cloth mesh
    #     cloth = open("tests/outputs/ingo_cloth.obj", "w")
    #     cloth_verts = state_1.particle_q.to("cpu").numpy()
    #     cloth_indices = model.tri_indices.to("cpu").numpy().reshape((model.tri_count, 3))

    #     for v in cloth_verts:
    #         cloth.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        
    #     for f in cloth_indices:
    #         cloth.write("f {} {} {}\n".format(f[0] + 1, f[1]+ 1, f[2]+ 1))

    #     cloth.close()

    #     # write results
    #     contact_count = model.soft_contact_count.to("cpu").numpy()[0]
    #     contact_indices = model.soft_contact_particle.to("cpu").numpy()
    #     contact_pos = model.soft_contact_body_pos.to("cpu").numpy()

    #     assert(contact_count == model.particle_count)

    #     results = open("tests/outputs/ingo_results.txt", "w")
    #     results.write("# test_x test_y test_z, result_x result_y result_z\n")

    #     for i in range(contact_count[0]):
    #         cloth_x = cloth_verts[contact_indices[i][0]]
    #         contact_x = contact_pos[i]

    #         results.write("{} {} {}, {} {} {}\n".format(cloth_x[0], cloth_x[1], cloth_x[2], contact_x[0], contact_x[1], contact_x[2]))

    #     results.close()

    if (sim_render):

        with wp.ScopedTimer("render", active=False):

            stage.begin_frame(sim_time)
            stage.render_mesh(name="cloth", points=state_0.particle_q.to("cpu").numpy(), indices=model.tri_indices.to("cpu").numpy())
            #stage.render_points(name="points", points=state_0.particle_q.to("cpu").numpy(), radius=0.1)
            
            # render static geometry once
            if (i == 0):
                #stage.render_box(name="box", pos=(0.0, 0.0, 0.0), extents=(0.5, 0.5, 0.5))
                #stage.render_sphere(name="sphere", pos=(1.0, 0.0, 1.0), radius=1.0)
                stage.render_mesh(name="mesh", points=points, indices=indices, pos=(5.0, -2.0, 7.0), rot=wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.5), scale=(4.0, 4.0, 4.0))

            stage.end_frame()


stage.save()
