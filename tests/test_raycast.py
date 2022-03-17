# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import os
import sys
import numpy as np
import math
import ctypes

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp

wp.init()

@wp.kernel
def render(mesh: wp.uint64,
           cam_pos: wp.vec3,
           width: int,
           height: int,
           pixels: wp.array(dtype=wp.vec3)):
    
    tid = wp.tid()

    x = tid%width
    y = tid//width

    sx = 2.0*float(x)/float(height) - 1.0
    sy = 2.0*float(y)/float(height) - 1.0

    # compute view ray
    ro = cam_pos
    rd = wp.normalize(wp.vec3(sx, sy, -1.0))    
    
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    color = wp.vec3(0.0, 0.0, 0.0)

    if wp.mesh_query_ray(mesh, ro, rd, 1.e+6, t, u, v, sign, n, f):
        color = n*0.5 + wp.vec3(0.5, 0.5, 0.5)
        
    pixels[tid] = color


device = "cuda"
width = 1024
height = 1024
cam_pos = (0.0, 1.0, 2.0)

from pxr import Usd, UsdGeom, Gf, Sdf

stage = Usd.Stage.Open("./tests/assets/bunny.usd")
mesh_geom = UsdGeom.Mesh(stage.GetPrimAtPath("/bunny/bunny"))

points = np.array(mesh_geom.GetPointsAttr().Get())
indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

pixels = wp.zeros(width*height, dtype=wp.vec3, device=device)

# create wp mesh
mesh = wp.Mesh(
    points=wp.array(points, dtype=wp.vec3, device=device),
    velocities=None,
    indices=wp.array(indices, dtype=int, device=device))


with wp.ScopedTimer("render"):

    wp.launch(
        kernel=render,
        dim=width*height,
        inputs=[mesh.id, cam_pos, width, height, pixels],
        device=device)

    wp.synchronize()

plt.imshow(pixels.numpy().reshape((height, width, 3)), origin="lower", interpolation="antialiased")
plt.show()