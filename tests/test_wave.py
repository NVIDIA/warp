# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxr import Usd, UsdGeom, Gf, Sdf


import oglang as og
import tests.util

@og.func
def sample(f: og.array(float),
           x: int,
           y: int,
           width: int,
           height: int):

    # clamp texture coords
    x = og.clamp(x, 0, width-1)
    y = og.clamp(y, 0, height-1)
    
    s = og.load(f, y*width + x)
    return s

@og.func
def laplacian(f: og.array(float),
              x: int,
              y: int,
              width: int,
              height: int):
    
    ddx = sample(f, x+1, y, width, height) - 2.0*sample(f, x,y, width, height) + sample(f, x-1, y, width, height)
    ddy = sample(f, x, y+1, width, height) - 2.0*sample(f, x,y, width, height) + sample(f, x, y-1, width, height)

    return (ddx + ddy)

@og.kernel
def wave_displace(hcurrent: og.array(float),
                  hprevious: og.array(float),
                  width: int,
                  height: int,
                  center_x: float,
                  center_y: float,
                  r: float,
                  mag: float,
                  t: float):

    tid = og.tid()

    x = tid%width
    y = tid//width

    dx = float(x) - center_x
    dy = float(y) - center_y

    dist_sq = float(dx*dx + dy*dy)

    if (dist_sq < r*r):

        h = mag*og.sin(t)

        og.store(hcurrent, tid, h)
        og.store(hprevious, tid, h)


@og.kernel
def wave_solve(hprevious: og.array(float),
               hcurrent: og.array(float),
               width: int,
               height: int,
               inv_cell: float,
               k_speed: float,
               k_damp: float,
               dt: float):

    tid = og.tid()

    x = tid%width
    y = tid//width

    l = laplacian(hcurrent, x, y, width, height)*inv_cell*inv_cell

    # integrate 
    h1 = og.load(hcurrent, tid)
    h0 = og.load(hprevious, tid)
    
    h = 2.0*h1 - h0 + dt*dt*(k_speed*l - k_damp*(h1-h0))

    # buffers get swapped each iteration
    og.store(hprevious, tid, h)


# simple kernel to apply height deltas to a vertex array
@og.kernel
def grid_update(heights: og.array(float),
                vertices: og.array(og.float3)):

    tid = og.tid()

    h = og.load(heights, tid)
    v = og.load(vertices, tid)

    v_new = og.float3(v[0], h, v[2])

    og.store(vertices, tid, v_new)


# params
sim_width = 128
sim_height = 128

sim_fps = 60.0
sim_substeps = 16
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0

# wave constants
k_speed = 1.0
k_damp = 0.0

# set up grid for visualization
stage = Usd.Stage.CreateNew("tests/outputs/wave.usd")
stage.SetStartTimeCode(0.0)
stage.SetEndTimeCode(sim_duration*sim_fps)
stage.SetTimeCodesPerSecond(sim_fps)

grid = UsdGeom.Mesh.Define(stage, "/root")
grid_size = 0.1
grid_displace = 0.5

# simulation context
context = og.Context("cuda")

# simulation grids
sim_grid0 = context.zeros(sim_width*sim_height, dtype=float, device="cuda")
sim_grid1 = context.zeros(sim_width*sim_height, dtype=float, device="cuda")

sim_host = context.zeros(sim_width*sim_height, dtype=float, device="cpu")
verts_host = context.zeros(sim_width*sim_height, dtype=og.float3, device="cpu")

vertices = verts_host.numpy().reshape((sim_width*sim_height, 3))
indices = []
counts = []

def add_sphere(stage, pos: tuple, radius: float, time: float=0.0):
    """Debug helper to add a sphere for visualization
    
    Args:
        pos: The position of the sphere
        radius: The radius of the sphere
        name: A name for the USD prim on the stage
    """

    sphere_path = "/sphere"
    sphere = UsdGeom.Sphere.Get(stage, sphere_path)
    if not sphere:
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    
    sphere.GetRadiusAttr().Set(radius, time)

    mat = Gf.Matrix4d()
    mat.SetIdentity()
    mat.SetTranslateOnly(Gf.Vec3d(pos))

    op = sphere.MakeMatrixXform()
    op.Set(mat, time)

def grid_index(x, y, stride):
    return y*stride + x

for z in range(sim_height):
    for x in range(sim_width):

        pos = Gf.Vec3f(float(x)*grid_size, 0.0, float(z)*grid_size)# - Gf.Vec3f(float(sim_width)/2*grid_size, 0.0, float(sim_height)/2*grid_size)

        vertices[z*sim_width + x] = pos#append(pos)
            
        if (x > 0 and z > 0):
            
            indices.append(grid_index(x-1, z-1, sim_width))
            indices.append(grid_index(x, z, sim_width))
            indices.append(grid_index(x, z-1, sim_width))

            indices.append(grid_index(x-1, z-1, sim_width))
            indices.append(grid_index(x-1, z, sim_width))
            indices.append(grid_index(x, z, sim_width))

            counts.append(3)
            counts.append(3)

grid.GetPointsAttr().Set(vertices, 0.0)
grid.GetFaceVertexIndicesAttr().Set(indices, 0.0)
grid.GetFaceVertexCountsAttr().Set(counts, 0.0)



for i in range(sim_frames):

    # simulate
    with tests.util.ScopedTimer("Simulate"):

        for s in range(sim_substeps):

            #create surface displacment around a point
            cx = sim_width/2 + math.sin(sim_time)*sim_width/3
            cy = sim_height/2 + math.cos(sim_time)*sim_height/3

            context.launch(
                kernel=wave_displace, 
                dim=sim_width*sim_height, 
                inputs=[sim_grid0, sim_grid1, sim_width, sim_height, cx, cy, 10.0, grid_displace, -math.pi*0.5],  
                outputs=[],
                device="cuda")


            # integrate wave equation
            context.launch(
                kernel=wave_solve, 
                dim=sim_width*sim_height, 
                inputs=[sim_grid0, sim_grid1, sim_width, sim_height, 1.0/grid_size, k_speed, k_damp, sim_dt], 
                outputs=[],
                device="cuda")

            # swap grids
            (sim_grid0, sim_grid1) = (sim_grid1, sim_grid0)

            sim_time += sim_dt


        # copy data back to host
        context.copy(sim_grid0, sim_host)
        context.synchronize()

    # render
    with tests.util.ScopedTimer("Render"):

        # update vertices (CPU)
        with tests.util.ScopedTimer("Mesh"):
            
            context.launch(kernel=grid_update,
                        dim=sim_width*sim_height,
                        inputs=[sim_host, verts_host],
                        outputs=[],
                        device="cpu")

        # render
        # sim_vew = sim_grid0.numpy()

        # with tests.util.ScopedTimer("Mesh"):
        #     for v in range(sim_width*sim_height):
        #         vertices[v] = Gf.Vec3f(vertices[v][0], float(sim_view[v]), vertices[v][2])

        with tests.util.ScopedTimer("Usd"):
            
            vertices = verts_host.numpy()

            grid.GetPointsAttr().Set(vertices, sim_time*sim_fps)

            add_sphere(stage, (cx*grid_size, 0.0, cy*grid_size), 10.0*grid_size, sim_time*sim_fps)



stage.Save()
