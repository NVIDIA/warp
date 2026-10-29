# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Ray March
#
# Shows how to implement an SDF ray marching based renderer. Please see
# https://iquilezles.org/articles/distfunctions/
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
def draw(cam_pos: wp.vec3, cam_rot: wp.quat, width: int, height: int, pixels: wp.array[wp.vec3]):
    tid = wp.tid()

    x = tid % width
    y = tid // width

    sx = (2.0 * float(x) - float(width)) / float(height)
    sy = (2.0 * float(y) - float(height)) / float(height)

    ro = cam_pos
    rd = wp.quat_rotate(cam_rot, wp.normalize(wp.vec3(sx, sy, -2.0)))

    t = float(0.0)

    d = float(0.0)
    march_stable = bool(False)
    for iteration in range(128):
        if not march_stable:
            march_point = ro + rd * t
            plane_distance = sdf_plane(march_point, wp.vec4(0.0, 1.0, 0.0, 1.0))
            if plane_distance <= 0.25:
                d = plane_distance
            else:
                box_bound = wp.max(
                    wp.abs(march_point[0]) - 1.0,
                    wp.max(wp.abs(march_point[1]) - 0.5, wp.abs(march_point[2]) - 0.5),
                )
                if plane_distance <= box_bound:
                    d = plane_distance
                else:
                    d = sdf(march_point)
        next_t = t + d
        march_stable = next_t == t
        t = next_t

    if d < 0.01:
        p = ro + rd * t
        normal_eps = 1.0e-5
        normal_top = p + wp.vec3(0.0, normal_eps, 0.0)
        if normal_top[1] <= -0.75 and wp.isfinite(p):
            normal_dx = sdf_plane(p + wp.vec3(normal_eps, 0.0, 0.0), wp.vec4(0.0, 1.0, 0.0, 1.0)) - sdf_plane(p - wp.vec3(normal_eps, 0.0, 0.0), wp.vec4(0.0, 1.0, 0.0, 1.0))
            normal_dy = sdf_plane(normal_top, wp.vec4(0.0, 1.0, 0.0, 1.0)) - sdf_plane(p - wp.vec3(0.0, normal_eps, 0.0), wp.vec4(0.0, 1.0, 0.0, 1.0))
            normal_dz = sdf_plane(p + wp.vec3(0.0, 0.0, normal_eps), wp.vec4(0.0, 1.0, 0.0, 1.0)) - sdf_plane(p - wp.vec3(0.0, 0.0, normal_eps), wp.vec4(0.0, 1.0, 0.0, 1.0))
            n = wp.normalize(wp.vec3(normal_dx, normal_dy, normal_dz))
        else:
            n = normal(p)
        l = wp.normalize(wp.vec3(0.6, 0.4, 0.5))

        h = wp.normalize(l - rd)

        diffuse = wp.dot(n, l)
        specular = wp.clamp(wp.dot(n, h), 0.0, 1.0) ** 80.0
        fresnel = 0.04 + 0.96 * wp.clamp(1.0 - wp.dot(h, l), 0.0, 1.0) ** 5.0

        intensity = 2.0
        shadow_t = float(0.0)
        shadow_factor = float(1.0)
        for shadow_iteration in range(64):
            shadow_point = p + shadow_t * l
            shadow_plane = sdf_plane(shadow_point, wp.vec4(0.0, 1.0, 0.0, 1.0))
            if shadow_plane <= 0.25:
                shadow_distance = shadow_plane
            else:
                shadow_bound = wp.max(
                    wp.abs(shadow_point[0]) - 1.0,
                    wp.max(wp.abs(shadow_point[1]) - 0.5, wp.abs(shadow_point[2]) - 0.5),
                )
                if shadow_plane <= shadow_bound:
                    shadow_distance = shadow_plane
                else:
                    shadow_distance = sdf(shadow_point)
            shadow_t = shadow_t + wp.clamp(shadow_distance, 0.0001, 2.0)
            shadow_h = wp.clamp(4.0 * shadow_distance / shadow_t, 0.0, 1.0)
            shadow_factor = wp.min(shadow_factor, shadow_h * shadow_h * (3.0 - 2.0 * shadow_h))
            if shadow_t > 8.0:
                shadow_factor = 1.0
                break
        result = (
            wp.vec3(0.85, 0.9, 0.95)
            * (diffuse * (1.0 - fresnel) + specular * fresnel * 10.0)
            * shadow_factor
            * intensity
        )

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
