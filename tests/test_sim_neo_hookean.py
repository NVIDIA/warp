# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxr import Usd, UsdGeom, Gf, Sdf

import warp as wp
import warp.sim as wpsim

wp.init()

import render

np.random.seed(42)
np.set_printoptions(threshold=sys.maxsize)

# params
sim_width = 8
sim_height = 8

sim_fps = 60.0
sim_substeps = 64
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0
sim_render = True
sim_iterations = 1
sim_relaxation = 1.0

device = "cuda"

builder = wpsim.ModelBuilder()


cell_dim = 15
cell_size = 2.0/cell_dim

center = cell_size*cell_dim*0.5

builder.add_soft_grid(
    pos=(-center, 0.0, -center), 
    rot=wp.quat_identity(), 
    vel=(0.0, 0.0, 0.0), 
    dim_x=cell_dim, 
    dim_y=cell_dim, 
    dim_z=cell_dim,
    cell_x=cell_size, 
    cell_y=cell_size,
    cell_z=cell_size,
    density=100.0, 
    fix_bottom=True,
    fix_top=True,
    k_mu=1000.0, 
    k_lambda=5000.0,
    k_damp=0.0)

model = builder.finalize(device=device)
model.ground = False
model.gravity[1] = 0.0

#integrator = wpsim.SemiImplicitIntegrator()
integrator = wpsim.XPBDIntegrator(iterations=sim_iterations, relaxation=sim_relaxation)

rest = model.state()
rest_vol = (cell_size*cell_dim)**3

state_0 = model.state()
state_1 = model.state()

stage = render.UsdRenderer("tests/outputs/test_sim_neo_hookean_twist.usd")

@wp.kernel
def twist_points(rest: wp.array(dtype=wp.vec3),
                 points: wp.array(dtype=wp.vec3),
                 mass: wp.array(dtype=float),
                 xform: wp.spatial_transform):

    tid = wp.tid()

    r = rest[tid]
    p = points[tid]
    m = mass[tid]

    if (m == 0 and p[1] != 0.0):

        points[tid] = wp.spatial_transform_point(xform, r)

@wp.kernel
def compute_volume(points: wp.array(dtype=wp.vec3),
                 indices: wp.array(dtype=int),
                 volume: wp.array(dtype=float)):

    tid = wp.tid()

    i = wp.load(indices, tid * 4 + 0)
    j = wp.load(indices, tid * 4 + 1)
    k = wp.load(indices, tid * 4 + 2)
    l = wp.load(indices, tid * 4 + 3)

    x0 = wp.load(points, i)
    x1 = wp.load(points, j)
    x2 = wp.load(points, k)
    x3 = wp.load(points, l)

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v = wp.dot(x10, wp.cross(x20, x30))/6.0

    wp.atomic_add(volume, 0, v)


volume = wp.zeros(1, dtype=wp.float32, device=device)

lift_speed = 2.5/sim_duration*2.0 # from Smith et al.
rot_speed = 0.0 #math.pi/sim_duration

with wp.ScopedTimer("Total"):
    
    for i in range(sim_frames):
        
        if (sim_render):
        
            with wp.ScopedTimer("render"):

                stage.begin_frame(sim_time)
                stage.render_mesh(name="cloth", points=state_0.particle_q.to("cpu").numpy(), indices=model.tri_indices.to("cpu").numpy())
                #stage.render_points(name="points", points=state_0.particle_q.to("cpu").numpy(), radius=0.1)
                #stage.render_

                stage.end_frame()

        with wp.ScopedTimer("simulate"):

            xform = (*(0.0, lift_speed*sim_time, 0.0), *wp.quat_from_axis_angle((0.0, 1.0, 0.0), rot_speed*sim_time))
            wp.launch(kernel=twist_points, dim=len(state_0.particle_q), inputs=[rest.particle_q, state_0.particle_q, model.particle_mass, xform], device=device)

            for s in range(sim_substeps):

                state_0.clear_forces()
                state_1.clear_forces()

                integrator.simulate(model, state_0, state_1, sim_dt)
                sim_time += sim_dt

                # swap states
                (state_0, state_1) = (state_1, state_0)

            volume.zero_()
            wp.launch(kernel=compute_volume, dim=model.tet_count, inputs=[state_0.particle_q, model.tet_indices, volume], device=device)

            v = volume.to("cpu").numpy()
            print(v[0]/rest_vol)

            wp.synchronize()

stage.save()



# Pixar ref.

# lambda, time, err
# 5000, 1m, 1.24
# 10000, 1m20, 1.13
# 50000, 3m20, 1.02 * (solver failed)
# 100000, 7m10, 1.013 * (solver failed)

# Ours

# lambda, time, err
# 5000, 28s, 1.2
# 10000, 28s 1.13
# 50000, 28s, 1.02
# 100000, 28s, 1.01
# 1000000, 28s, 1.001