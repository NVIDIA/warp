# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import warp.examples


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

    color = wp.vec3(0.0, 0.0, 0.0)

    query = wp.mesh_query_ray(mesh, ro, rd, 1.0e6)
    if query.result:
        color = query.normal * 0.5 + wp.vec3(0.5, 0.5, 0.5)

    pixels[tid] = color


class Example:
    def __init__(self, height=1024, width=1024):
        self.height = height
        self.width = width
        self.cam_pos = (0.0, 1.0, 2.0)

        asset_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bunny"))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)

        # create wp mesh
        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3), velocities=None, indices=wp.array(indices, dtype=int)
        )

    def render(self):
        with wp.ScopedTimer("render"):
            wp.launch(
                kernel=draw,
                dim=self.width * self.height,
                inputs=[self.mesh.id, self.cam_pos, self.width, self.height, self.pixels],
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height in pixels.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(height=args.height, width=args.width)
        example.render()

        if not args.headless:
            import matplotlib.pyplot as plt

            plt.imshow(
                example.pixels.numpy().reshape((example.height, example.width, 3)),
                origin="lower",
                interpolation="antialiased",
            )
            plt.show()
