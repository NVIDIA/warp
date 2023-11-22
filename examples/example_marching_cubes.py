# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Marching Cubes
#
# Shows how use the built-in marching cubes functionality to extract
# the iso-surface from a density field.
#
###########################################################################


import math
import os

import warp as wp
import warp.render

wp.init()


# signed sphere
@wp.func
def sdf_sphere(p: wp.vec3, r: float):
    return wp.length(p) - r


# signed box
@wp.func
def sdf_box(upper: wp.vec3, p: wp.vec3):
    qx = wp.abs(p[0]) - upper[0]
    qy = wp.abs(p[1]) - upper[1]
    qz = wp.abs(p[2]) - upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def op_union(d1: float, d2: float):
    return wp.min(d1, d2)


@wp.func
def op_smooth_union(d1: float, d2: float, k: float):
    a = d1
    b = d2

    h = wp.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return wp.lerp(b, a, h) - k * h * (1.0 - h)


@wp.func
def op_subtract(d1: float, d2: float):
    return wp.max(-d1, d2)


@wp.func
def op_intersect(d1: float, d2: float):
    return wp.max(d1, d2)


@wp.kernel
def make_field(field: wp.array3d(dtype=float), center: wp.vec3, radius: float, time: float):
    i, j, k = wp.tid()

    p = wp.vec3(float(i), float(j), float(k))

    rng = wp.rand_init(42)
    noise = wp.noise(rng, wp.vec4(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5, time) * 0.25)

    sphere = 2.0 * noise + wp.length(p - center) - radius
    box = sdf_box(wp.vec3(16.0, 48.0, 16.0), p - center)

    d = op_smooth_union(sphere, box, 4.0)

    field[i, j, k] = d


class Example:
    def __init__(self, stage):
        self.dim = 128
        self.max_verts = 10**6
        self.max_tris = 10**6

        self.time = 0.0
        self.frame_dt = 1.0 / 60.0

        self.field = wp.zeros(shape=(self.dim, self.dim, self.dim), dtype=float)

        self.iso = wp.MarchingCubes(
            nx=self.dim, ny=self.dim, nz=self.dim, max_verts=self.max_verts, max_tris=self.max_tris
        )

        self.renderer = wp.render.UsdRenderer(stage)

    def update(self):
        with wp.ScopedTimer("Update Field"):
            wp.launch(
                make_field,
                dim=self.field.shape,
                inputs=[self.field, wp.vec3(self.dim / 2, self.dim / 2, self.dim / 2), self.dim / 4, self.time],
            )
            self.time += self.frame_dt

        with wp.ScopedTimer("Surface Extraction"):
            self.iso.surface(field=self.field, threshold=math.sin(self.time) * self.dim / 8)

    def render(self, is_live=False):
        with wp.ScopedTimer("Render"):
            self.renderer.begin_frame(self.time)
            self.renderer.render_mesh("surface", self.iso.verts.numpy(), self.iso.indices.numpy(), update_topology=True)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_marching_cubes.usd")

    example = Example(stage_path)

    for i in range(240):
        example.update()
        example.render()

    example.renderer.save()
