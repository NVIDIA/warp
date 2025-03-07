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
# Example Differentiable Ray Caster
#
# Shows how to use the built-in wp.Mesh data structure and wp.mesh_query_ray()
# function to implement a basic differentiable ray caster
#
##############################################################################

import math
import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
from warp.optim import SGD


class RenderMode:
    """Rendering modes
    grayscale: Lambertian shading from multiple directional lights
    texture: 2D texture map
    normal_map: mesh normal computed from interpolated vertex normals
    """

    grayscale = 0
    texture = 1
    normal_map = 2


@wp.struct
class RenderMesh:
    """Mesh to be ray casted.
    Assumes a triangle mesh as input.
    Per-vertex normals are computed with compute_vertex_normals()
    """

    id: wp.uint64
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=int)
    tex_coords: wp.array(dtype=wp.vec2)
    tex_indices: wp.array(dtype=int)
    vertex_normals: wp.array(dtype=wp.vec3)
    pos: wp.array(dtype=wp.vec3)
    rot: wp.array(dtype=wp.quat)


@wp.struct
class Camera:
    """Basic camera for ray casting"""

    horizontal: float
    vertical: float
    aspect: float
    e: float
    tan: float
    pos: wp.vec3
    rot: wp.quat


@wp.struct
class DirectionalLights:
    """Stores arrays of directional light directions and intensities."""

    dirs: wp.array(dtype=wp.vec3)
    intensities: wp.array(dtype=float)
    num_lights: int


@wp.kernel
def vertex_normal_sum_kernel(
    verts: wp.array(dtype=wp.vec3), indices: wp.array(dtype=int), normal_sums: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()

    i = indices[tid * 3]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]

    a = verts[i]
    b = verts[j]
    c = verts[k]

    ab = b - a
    ac = c - a

    area_normal = wp.cross(ab, ac)
    wp.atomic_add(normal_sums, i, area_normal)
    wp.atomic_add(normal_sums, j, area_normal)
    wp.atomic_add(normal_sums, k, area_normal)


@wp.kernel
def normalize_kernel(
    normal_sums: wp.array(dtype=wp.vec3),
    vertex_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    vertex_normals[tid] = wp.normalize(normal_sums[tid])


@wp.func
def texture_interpolation(tex_interp: wp.vec2, texture: wp.array2d(dtype=wp.vec3)):
    tex_width = texture.shape[1]
    tex_height = texture.shape[0]
    tex = wp.vec2(tex_interp[0] * float(tex_width - 1), (1.0 - tex_interp[1]) * float(tex_height - 1))

    x0 = int(tex[0])
    x1 = x0 + 1
    alpha_x = tex[0] - float(x0)
    y0 = int(tex[1])
    y1 = y0 + 1
    alpha_y = tex[1] - float(y0)
    c00 = texture[y0, x0]
    c10 = texture[y0, x1]
    c01 = texture[y1, x0]
    c11 = texture[y1, x1]
    lower = (1.0 - alpha_x) * c00 + alpha_x * c10
    upper = (1.0 - alpha_x) * c01 + alpha_x * c11
    color = (1.0 - alpha_y) * lower + alpha_y * upper

    return color


@wp.kernel
def draw_kernel(
    mesh: RenderMesh,
    camera: Camera,
    texture: wp.array2d(dtype=wp.vec3),
    rays_width: int,
    rays_height: int,
    rays: wp.array(dtype=wp.vec3),
    lights: DirectionalLights,
    mode: int,
):
    tid = wp.tid()

    x = tid % rays_width
    y = rays_height - tid // rays_width

    sx = 2.0 * float(x) / float(rays_width) - 1.0
    sy = 2.0 * float(y) / float(rays_height) - 1.0

    # compute view ray in world space
    ro_world = camera.pos
    rd_world = wp.normalize(wp.quat_rotate(camera.rot, wp.vec3(sx * camera.tan * camera.aspect, sy * camera.tan, -1.0)))

    # compute view ray in mesh space
    inv = wp.transform_inverse(wp.transform(mesh.pos[0], mesh.rot[0]))
    ro = wp.transform_point(inv, ro_world)
    rd = wp.transform_vector(inv, rd_world)

    color = wp.vec3(0.0, 0.0, 0.0)

    query = wp.mesh_query_ray(mesh.id, ro, rd, 1.0e6)
    if query.result:
        i = mesh.indices[query.face * 3]
        j = mesh.indices[query.face * 3 + 1]
        k = mesh.indices[query.face * 3 + 2]

        a_n = mesh.vertex_normals[i]
        b_n = mesh.vertex_normals[j]
        c_n = mesh.vertex_normals[k]

        # vertex normal interpolation
        normal = query.u * a_n + query.v * b_n + (1.0 - query.u - query.v) * c_n

        if mode == 0 or mode == 1:
            if mode == 0:  # grayscale
                color = wp.vec3(1.0)

            elif mode == 1:  # texture interpolation
                tex_a = mesh.tex_coords[mesh.tex_indices[query.face * 3]]
                tex_b = mesh.tex_coords[mesh.tex_indices[query.face * 3 + 1]]
                tex_c = mesh.tex_coords[mesh.tex_indices[query.face * 3 + 2]]

                tex = query.u * tex_a + query.v * tex_b + (1.0 - query.u - query.v) * tex_c

                color = texture_interpolation(tex, texture)

            # lambertian directional lighting
            lambert = float(0.0)
            for i in range(lights.num_lights):
                dir = wp.transform_vector(inv, lights.dirs[i])
                val = lights.intensities[i] * wp.dot(normal, dir)
                if val < 0.0:
                    val = 0.0
                lambert = lambert + val

            color = lambert * color

        elif mode == 2:  # normal map
            color = normal * 0.5 + wp.vec3(0.5, 0.5, 0.5)

        if color[0] > 1.0:
            color = wp.vec3(1.0, color[1], color[2])
        if color[1] > 1.0:
            color = wp.vec3(color[0], 1.0, color[2])
        if color[2] > 1.0:
            color = wp.vec3(color[0], color[1], 1.0)

    rays[tid] = color


@wp.kernel
def downsample_kernel(
    rays: wp.array(dtype=wp.vec3), pixels: wp.array(dtype=wp.vec3), rays_width: int, num_samples: int
):
    tid = wp.tid()

    pixels_width = rays_width / num_samples
    px = tid % pixels_width
    py = tid // pixels_width
    start_idx = py * num_samples * rays_width + px * num_samples

    color = wp.vec3(0.0, 0.0, 0.0)

    for i in range(0, num_samples):
        for j in range(0, num_samples):
            ray = rays[start_idx + i * rays_width + j]
            color = wp.vec3(color[0] + ray[0], color[1] + ray[1], color[2] + ray[2])

    num_samples_sq = float(num_samples * num_samples)
    color = wp.vec3(color[0] / num_samples_sq, color[1] / num_samples_sq, color[2] / num_samples_sq)
    pixels[tid] = color


@wp.kernel
def loss_kernel(pixels: wp.array(dtype=wp.vec3), target_pixels: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    tid = wp.tid()

    pixel = pixels[tid]
    target_pixel = target_pixels[tid]

    diff = target_pixel - pixel

    # pseudo Huber loss
    delta = 1.0
    x = delta * delta * (wp.sqrt(1.0 + (diff[0] / delta) * (diff[0] / delta)) - 1.0)
    y = delta * delta * (wp.sqrt(1.0 + (diff[1] / delta) * (diff[1] / delta)) - 1.0)
    z = delta * delta * (wp.sqrt(1.0 + (diff[2] / delta) * (diff[2] / delta)) - 1.0)
    sum = x + y + z

    wp.atomic_add(loss, 0, sum)


@wp.kernel
def normalize(x: wp.array(dtype=wp.quat)):
    tid = wp.tid()

    x[tid] = wp.normalize(x[tid])


class Example:
    """
    Non-differentiable variables:
    camera.horizontal: camera horizontal aperture size
    camera.vertical: camera vertical aperture size
    camera.aspect: camera aspect ratio
    camera.e: focal length
    camera.pos: camera displacement
    camera.rot: camera rotation (quaternion)
    pix_width: final image width in pixels
    pix_height: final image height in pixels
    num_samples: anti-aliasing. calculated as pow(2, num_samples)
    directional_lights: characterized by intensity (scalar) and direction (vec3)
    render_mesh.indices: mesh vertex indices
    render_mesh.tex_indices: texture indices

    Differentiable variables:
    render_mesh.pos: parent transform displacement
    render_mesh.quat: parent transform rotation (quaternion)
    render_mesh.vertices: mesh vertex positions
    render_mesh.vertex_normals: mesh vertex normals
    render_mesh.tex_coords: 2D texture coordinates
    """

    def __init__(self, height=1024, train_iters=150, rot_array=None):
        cam_pos = wp.vec3(0.0, 0.75, 7.0)
        cam_rot = wp.quat(0.0, 0.0, 0.0, 1.0)
        horizontal_aperture = 36.0
        vertical_aperture = 20.25
        aspect = horizontal_aperture / vertical_aperture
        focal_length = 50.0
        self.height = height
        self.width = int(aspect * self.height)
        self.num_pixels = self.width * self.height

        if rot_array is None:
            rot_array = [0.0, 0.0, 0.0, 1.0]

        asset_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bunny"))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())
        num_points = points.shape[0]
        num_faces = int(indices.shape[0] / 3)

        # manufacture texture coordinates + indices for this asset
        distance = np.linalg.norm(points, axis=1)
        radius = np.max(distance)
        distance = distance / radius
        tex_coords = np.stack((distance, distance), axis=1)
        tex_indices = indices

        # manufacture texture for this asset
        x = np.arange(256.0)
        xx, yy = np.meshgrid(x, x)
        zz = np.zeros_like(xx)
        texture_host = np.stack((xx, yy, zz), axis=2) / 255.0

        # set anti-aliasing
        self.num_samples = 1

        # set render mode
        self.render_mode = RenderMode.texture

        # set training iterations
        self.train_rate = 5.00e-8
        self.momentum = 0.5
        self.dampening = 0.1
        self.weight_decay = 0.0
        self.train_iters = train_iters
        self.period = 10  # Training iterations between render() calls
        self.iter = 0

        # storage for training animation
        self.images = np.zeros((self.height, self.width, 3, max(int(self.train_iters / self.period), 1)))
        self.image_counter = 0

        # construct RenderMesh
        self.render_mesh = RenderMesh()
        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, requires_grad=True),
            indices=wp.array(indices, dtype=int),
        )
        self.render_mesh.id = self.mesh.id
        self.render_mesh.vertices = self.mesh.points
        self.render_mesh.indices = self.mesh.indices
        self.render_mesh.tex_coords = wp.array(tex_coords, dtype=wp.vec2, requires_grad=True)
        self.render_mesh.tex_indices = wp.array(tex_indices, dtype=int)
        self.normal_sums = wp.zeros(num_points, dtype=wp.vec3, requires_grad=True)
        self.render_mesh.vertex_normals = wp.zeros(num_points, dtype=wp.vec3, requires_grad=True)
        self.render_mesh.pos = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
        self.render_mesh.rot = wp.array(np.array(rot_array), dtype=wp.quat, requires_grad=True)

        # compute vertex normals
        wp.launch(
            kernel=vertex_normal_sum_kernel,
            dim=num_faces,
            inputs=[self.render_mesh.vertices, self.render_mesh.indices, self.normal_sums],
        )
        wp.launch(
            kernel=normalize_kernel,
            dim=num_points,
            inputs=[self.normal_sums, self.render_mesh.vertex_normals],
        )

        # construct camera
        self.camera = Camera()
        self.camera.horizontal = horizontal_aperture
        self.camera.vertical = vertical_aperture
        self.camera.aspect = aspect
        self.camera.e = focal_length
        self.camera.tan = vertical_aperture / (2.0 * focal_length)
        self.camera.pos = cam_pos
        self.camera.rot = cam_rot

        # construct texture
        self.texture = wp.array2d(texture_host, dtype=wp.vec3, requires_grad=True)

        # construct lights
        self.lights = DirectionalLights()
        self.lights.dirs = wp.array(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), dtype=wp.vec3, requires_grad=True)
        self.lights.intensities = wp.array(np.array([2.0, 0.2]), dtype=float, requires_grad=True)
        self.lights.num_lights = 2

        # construct rays
        self.rays_width = self.width * pow(2, self.num_samples)
        self.rays_height = self.height * pow(2, self.num_samples)
        self.num_rays = self.rays_width * self.rays_height
        self.rays = wp.zeros(self.num_rays, dtype=wp.vec3, requires_grad=True)

        # construct pixels
        self.pixels = wp.zeros(self.num_pixels, dtype=wp.vec3, requires_grad=True)
        self.target_pixels = wp.zeros(self.num_pixels, dtype=wp.vec3)

        # loss array
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # capture graph
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)
            self.graph = capture.graph

        self.optimizer = SGD(
            [self.render_mesh.rot],
            self.train_rate,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
        )

    def ray_cast(self):
        # raycast
        wp.launch(
            kernel=draw_kernel,
            dim=self.num_rays,
            inputs=[
                self.render_mesh,
                self.camera,
                self.texture,
                self.rays_width,
                self.rays_height,
                self.rays,
                self.lights,
                self.render_mode,
            ],
        )

        # downsample
        wp.launch(
            kernel=downsample_kernel,
            dim=self.num_pixels,
            inputs=[self.rays, self.pixels, self.rays_width, pow(2, self.num_samples)],
        )

    def forward(self):
        self.ray_cast()

        # compute pixel loss
        wp.launch(loss_kernel, dim=self.num_pixels, inputs=[self.pixels, self.target_pixels, self.loss])

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)

            rot_grad = self.tape.gradients[self.render_mesh.rot]
            self.optimizer.step([rot_grad])
            wp.launch(normalize, dim=1, inputs=[self.render_mesh.rot])

            if self.iter % self.period == 0:
                print(f"Iter: {self.iter} Loss: {self.loss}")

            self.tape.zero()
            self.loss.zero_()

            self.iter = self.iter + 1

    def render(self):
        with wp.ScopedTimer("render"):
            self.images[:, :, :, self.image_counter] = self.get_image()
            self.image_counter += 1

    def get_image(self):
        return self.pixels.numpy().reshape((self.height, self.width, 3))

    def get_animation(self):
        fig, ax = plt.subplots()
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        frames = []
        for i in range(self.images.shape[3]):
            frame = ax.imshow(self.images[:, :, :, i], animated=True)
            frames.append([frame])

        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
        return ani


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--train_iters", type=int, default=150, help="Total number of training iterations.")
    parser.add_argument("--height", type=int, default=1024, help="Height of rendered image in pixels.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        reference_example = Example(height=args.height)

        # render target rotation
        reference_example.ray_cast()

        # offset mesh rotation
        example = Example(
            train_iters=args.train_iters,
            height=args.height,
            rot_array=[
                0.0,
                (math.sqrt(3) - 1) / (2.0 * math.sqrt(2.0)),
                0.0,
                (math.sqrt(3) + 1) / (2.0 * math.sqrt(2.0)),
            ],
        )

        wp.copy(example.target_pixels, reference_example.pixels)

        # recover target rotation
        for i in range(example.train_iters):
            example.step()

            if i % example.period == 0:
                example.render()

        if not args.headless:
            import matplotlib.animation as animation
            import matplotlib.image as img
            import matplotlib.pyplot as plt

            target_image = reference_example.get_image()
            target_image_filename = "example_diffray_target_image.png"
            img.imsave(target_image_filename, target_image)
            print(f"Saved the target image at `{target_image_filename}`")

            final_image = example.get_image()
            final_image_filename = "example_diffray_final_image.png"
            img.imsave(final_image_filename, final_image)
            print(f"Saved the final image at `{final_image_filename}`")

            anim = example.get_animation()
            anim_filename = "example_diffray_animation.gif"
            anim.save(anim_filename, dpi=300, writer=animation.PillowWriter(fps=5))
            print(f"Saved the animation at `{anim_filename}`")
