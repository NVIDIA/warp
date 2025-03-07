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

###########################################################################
# Example Ray March
#
# Shows how to implement an SDF ray marching based renderer. Please see
# https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
# for reference on different distance functions.
#
###########################################################################

import warp as wp


@wp.func
def sdf_sphere(p: wp.vec3, r: float):
    return wp.length(p) - r


@wp.func
def sdf_box(upper: wp.vec3, p: wp.vec3):
    qx = wp.abs(p[0]) - upper[0]
    qy = wp.abs(p[1]) - upper[1]
    qz = wp.abs(p[2]) - upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def sdf_plane(p: wp.vec3, plane: wp.vec4):
    return plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3]


@wp.func
def op_union(d1: float, d2: float):
    return wp.min(d1, d2)


@wp.func
def op_subtract(d1: float, d2: float):
    return wp.max(-d1, d2)


@wp.func
def op_intersect(d1: float, d2: float):
    return wp.max(d1, d2)


# simple scene
@wp.func
def sdf(p: wp.vec3):
    sphere_1 = wp.vec3(0.0, 0.75, 0.0)

    d = op_subtract(sdf_sphere(p - sphere_1, 0.75), sdf_box(wp.vec3(1.0, 0.5, 0.5), p))

    # ground plane
    d = op_union(d, sdf_plane(p, wp.vec4(0.0, 1.0, 0.0, 1.0)))

    return d


@wp.func
def normal(p: wp.vec3):
    eps = 1.0e-5

    # compute gradient of the SDF using finite differences
    dx = sdf(p + wp.vec3(eps, 0.0, 0.0)) - sdf(p - wp.vec3(eps, 0.0, 0.0))
    dy = sdf(p + wp.vec3(0.0, eps, 0.0)) - sdf(p - wp.vec3(0.0, eps, 0.0))
    dz = sdf(p + wp.vec3(0.0, 0.0, eps)) - sdf(p - wp.vec3(0.0, 0.0, eps))

    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def shadow(ro: wp.vec3, rd: wp.vec3):
    t = float(0.0)
    s = float(1.0)

    for _ in range(64):
        d = sdf(ro + t * rd)
        t = t + wp.clamp(d, 0.0001, 2.0)

        h = wp.clamp(4.0 * d / t, 0.0, 1.0)
        s = wp.min(s, h * h * (3.0 - 2.0 * h))

        if t > 8.0:
            return 1.0

    return s


@wp.kernel
def draw(cam_pos: wp.vec3, cam_rot: wp.quat, width: int, height: int, pixels: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    x = tid % width
    y = tid // width

    # compute pixel coordinates
    sx = (2.0 * float(x) - float(width)) / float(height)
    sy = (2.0 * float(y) - float(height)) / float(height)

    # compute view ray
    ro = cam_pos
    rd = wp.quat_rotate(cam_rot, wp.normalize(wp.vec3(sx, sy, -2.0)))

    t = float(0.0)

    # ray march
    for _ in range(128):
        d = sdf(ro + rd * t)
        t = t + d

    if d < 0.01:
        p = ro + rd * t
        n = normal(p)
        l = wp.normalize(wp.vec3(0.6, 0.4, 0.5))

        # half-vector
        h = wp.normalize(l - rd)

        diffuse = wp.dot(n, l)
        specular = wp.clamp(wp.dot(n, h), 0.0, 1.0) ** 80.0
        fresnel = 0.04 + 0.96 * wp.clamp(1.0 - wp.dot(h, l), 0.0, 1.0) ** 5.0

        intensity = 2.0
        result = (
            wp.vec3(0.85, 0.9, 0.95)
            * (diffuse * (1.0 - fresnel) + specular * fresnel * 10.0)
            * shadow(p, l)
            * intensity
        )

        # gamma
        pixels[tid] = wp.vec3(
            wp.clamp(result[0] ** 2.2, 0.0, 1.0),
            wp.clamp(result[1] ** 2.2, 0.0, 1.0),
            wp.clamp(result[2] ** 2.2, 0.0, 1.0),
        )

    else:
        pixels[tid] = wp.vec3(0.4, 0.45, 0.5) * 1.5


class Example:
    def __init__(self, height=1024, width=2048):
        self.width = width
        self.height = height
        self.cam_pos = (-1.25, 1.0, 2.0)
        self.cam_rot = wp.quat_rpy(-0.5, -0.5, 0.0)

        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)

    def render(self):
        with wp.ScopedTimer("render"):
            wp.launch(
                kernel=draw,
                dim=self.width * self.height,
                inputs=[self.cam_pos, self.cam_rot, self.width, self.height, self.pixels],
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--width", type=int, default=2048, help="Output image width in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height in pixels.")

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
