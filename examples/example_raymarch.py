# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Ray March
#
# Shows how to implement an SDF ray marching based renderer. Please see
# https://iquilezles.org/www/articles/distfunctions/distfunctions.htm 
# for reference on different distance functions.
#
###########################################################################


import matplotlib.pyplot as plt

import warp as wp
import numpy as np

import os

wp.init()

# signed sphere 
@wp.func
def sdf_sphere(p: wp.vec3, r: float):
    return wp.length(p) - r

# signed box 
@wp.func
def sdf_box(upper: wp.vec3, p: wp.vec3):

    qx = wp.abs(p[0])-upper[0]
    qy = wp.abs(p[1])-upper[1]
    qz = wp.abs(p[2])-upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))
    
    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)

@wp.func
def sdf_plane(p: wp.vec3, plane: wp.vec4):
    return plane[0]*p[0] + plane[1]*p[1] + plane[2]*p[2] + plane[3]

# union 
@wp.func
def op_union(d1: float, d2: float):
    return wp.min(d1, d2)
    
# subtraction
@wp.func
def op_subtract(d1: float, d2: float):
    return wp.max(-d1, d2)

# intersection
@wp.func
def op_intersect(d1: float, d2: float):
    return wp.max(d1, d2)
    

# simple scene 
@wp.func 
def sdf(p: wp.vec3):

    # intersection of two spheres
    sphere_1 = wp.vec3(0.0, 0.0, 0.0)
    sphere_2 = wp.vec3(0.0, 0.75, 0.0)

    d = op_subtract(sdf_sphere(p - sphere_2, 0.75), 
                    sdf_box(wp.vec3(1.0, 0.5, 0.5), p))
                    #sdf_sphere(p + sphere_1, 1.0))


    # ground plane
    d = op_union(d, sdf_plane(p, wp.vec4(0.0, 1.0, 0.0, 1.0)))

    return d

@wp.func
def normal(p: wp.vec3):

    eps = 1.e-5

    # compute gradient of the SDF using finite differences
    dx = wp.sdf(p + wp.vec3(eps, 0.0, 0.0)) - wp.sdf(p - wp.vec3(eps, 0.0, 0.0))
    dy = wp.sdf(p + wp.vec3(0.0, eps, 0.0)) - wp.sdf(p - wp.vec3(0.0, eps, 0.0))
    dz = wp.sdf(p + wp.vec3(0.0, 0.0, eps)) - wp.sdf(p - wp.vec3(0.0, 0.0, eps))

    return wp.normalize(wp.vec3(dx, dy, dz))

@wp.func
def shadow(ro: wp.vec3, rd: wp.vec3):

    t = float(0.0)
    s = float(1.0)

    for i in range(64):

        d = sdf(ro + t*rd)
        t = t + wp.clamp(d, 0.0001, 2.0)

        h = wp.clamp(4.0*d/t, 0.0, 1.0)
        s = wp.min(s, h*h*(3.0-2.0*h))

        if (t > 8.0):
            return 1.0

    return s


@wp.kernel
def render(cam_pos: wp.vec3,
           cam_rot: wp.quat,
           width: int,
           height: int,
           pixels: wp.array(dtype=wp.vec3)):
    
    tid = wp.tid()

    x = tid%width
    y = tid//width

    # compute pixel coordinates
    sx = (2.0*float(x) - float(width))/float(height)
    sy = (2.0*float(y) - float(height))/float(height)

    # compute view ray
    ro = cam_pos
    rd = wp.quat_rotate(cam_rot, wp.normalize(wp.vec3(sx, sy, -1.0)))
        
    t = float(0.0)

    # ray march
    for i in range(128):

        d = sdf(ro + rd*t)
        t = t + d

    if d < 0.01:

        p = ro + rd*t
        n = wp.normal(p)        
        l = wp.normalize(wp.vec3(0.6, 0.4, 0.5))
        
        # half-vector
        h = wp.normalize(l-rd)

        diffuse = wp.dot(n, l)
        specular = wp.clamp(wp.dot(n, h), 0.0, 1.0)**80.0
        fresnel = 0.04 + 0.96*wp.clamp(1.0-wp.dot(h, l), 0.0, 1.0)**5.0

        intensity = 2.0
        result = wp.vec3(0.6, 0.6, 0.59)*(diffuse*(1.0-fresnel) + specular*fresnel*10.0)*shadow(p, l)*intensity

        # gamma
        pixels[tid] = wp.vec3(result[0]**2.2, result[1]**2.2, result[2]**2.2)

    else:

        pixels[tid] = wp.vec3(0.4, 0.45, 0.5)*1.5


device = wp.get_preferred_device()
width = 2048
height = 1024
cam_pos = (0.0, 1.0, 2.0)
cam_rot = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -0.5)

pixels = wp.zeros(width*height, dtype=wp.vec3, device=device)

with wp.ScopedTimer("render"):

    wp.launch(
        kernel=render,
        dim=width*height,
        inputs=[cam_pos, cam_rot, width, height, pixels],
        device=device)

    wp.synchronize()

plt.imshow(pixels.numpy().reshape((height, width, 3)), origin="lower", interpolation="antialiased")
plt.show()