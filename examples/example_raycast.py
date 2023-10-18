# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#############################################################################
# Example Ray Cast
#
# Shows how to use the built-in wp.Mesh data structure and wp.mesh_query_ray()
# function to implement a basic ray-tracer.
#
##############################################################################

import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp

wp.init()


@wp.kernel
def draw(mesh: wp.uint64, cam_pos: wp.vec3, width: int, height: int, pixels: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    x = tid % width
    y = tid // width

    sx = 2.0 * float(x) / float(height) - 1.0
    sy = 2.0 * float(y) / float(height) - 1.0

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

    if wp.mesh_query_ray(mesh, ro, rd, 1.0e6, t, u, v, sign, n, f):
        color = n * 0.5 + wp.vec3(0.5, 0.5, 0.5)

    pixels[tid] = color


class Example:
    def __init__(self, **kwargs):
        self.width = 1024
        self.height = 1024
        self.cam_pos = (0.0, 1.0, 2.0)

        asset_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets/bunny.usd"))
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/bunny/bunny"))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)

        # create wp mesh
        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3), velocities=None, indices=wp.array(indices, dtype=int)
        )

    def update(self):
        pass

    def render(self):
        with wp.ScopedTimer("render"):
            wp.launch(
                kernel=draw,
                dim=self.width * self.height,
                inputs=[self.mesh.id, self.cam_pos, self.width, self.height, self.pixels],
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    example = Example()
    example.render()

    wp.synchronize_device()

    plt.imshow(
        example.pixels.numpy().reshape((example.height, example.width, 3)), origin="lower", interpolation="antialiased"
    )
    plt.show()
