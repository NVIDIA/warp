# include parent path
import os
import sys
import numpy as np
import math
import ctypes

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import oglang as og

og.init()

@og.kernel
def render(mesh: og.uint64,
           cam_pos: og.vec3,
           width: int,
           height: int,
           pixels: og.array(dtype=og.vec3)):
    
    tid = og.tid()

    x = tid%width
    y = tid//width

    sx = 2.0*float(x)/float(height) - 1.0
    sy = 2.0*float(y)/float(height) - 1.0

    # compute view ray
    ro = cam_pos
    rd = og.normalize(og.vec3(sx, sy, -1.0))
    
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = og.vec3(0.0, 0.0, 0.0)

    color = og.vec3(0.0, 0.0, 0.0)

    if (og.mesh_query_ray(mesh, ro, rd, 1.e+6, t, u, v, sign, n)):
        color = n*0.5 + og.vec3(0.5, 0.5, 0.5)
        
    og.store(pixels, tid, color)


device = "cuda"
width = 1024
height = 1024
cam_pos = np.array((0.01, 0.1, 2.0))

from pxr import Usd, UsdGeom, Gf, Sdf

stage = Usd.Stage.Open("./tests/assets/suzanne_two.usda")
mesh_geom = UsdGeom.Mesh(stage.GetPrimAtPath("/World/model/Suzanne"))

points = np.array(mesh_geom.GetPointsAttr().Get())
indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

pixels = og.zeros(width*height, dtype=og.vec3, device=device)

# create og mesh
mesh = og.Mesh(
    points=og.array(points, dtype=og.vec3, device=device),
    velocities=None,
    indices=og.array(indices, dtype=int, device=device))

with og.ScopedTimer("render"):

    og.launch(
        kernel=render,
        dim=width*height,
        inputs=[mesh.id, cam_pos, width, height, pixels],
        device=device)

    og.synchronize()

plt.imshow(pixels.to("cpu").numpy().reshape((height, width, 3)), origin="lower", interpolation="antialiased")
plt.show()