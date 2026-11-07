# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark texture sampling inside a large camera-rendering kernel.

The kernel follows the structure of Newton's camera sensor render kernel:
closest-hit and shadow rays traverse a BVH of primitive and mesh shapes,
textured hits sample a 2D texture through one of several projections, and
each pixel writes color and depth. The sampling code is inlined into a
kernel that already needs many registers, so extra work in
``texture_sample()`` can lower occupancy even when no shape is textured.
The kernels in ``texture.py`` use few registers, so the same change does
not show up there.

Keep the mesh shapes and the separate texture call sites even though the
sampling code is the target. Together they raise the kernel's register use
enough for sampling overhead to change occupancy; a version without them
did not detect the regression this benchmark was added for.
"""

import math

import numpy as np

import warp as wp

from .benchmarks_utils import CudaEventTimer, setup_once

RESOLUTION = 512
GRID_SIZE = 16
NUM_TEXTURES = 8
TEXTURE_SIZE = 256
TEXTURE_REPEAT = 2.0
# Newton launches its render kernel with 64-thread blocks.
BLOCK_DIM = 64
MAX_DISTANCE = wp.constant(1000.0)

SHAPE_PLANE = wp.constant(0)
SHAPE_SPHERE = wp.constant(1)
SHAPE_BOX = wp.constant(2)
SHAPE_CAPSULE = wp.constant(3)
SHAPE_CYLINDER = wp.constant(4)
SHAPE_MESH = wp.constant(5)


@wp.func
def safe_inverse(x: float) -> float:
    if wp.abs(x) < 1.0e-12:
        return 1.0e12
    return 1.0 / x


@wp.func
def intersect_sphere(o: wp.vec3f, d: wp.vec3f, radius: float):
    b = wp.dot(o, d)
    discriminant = b * b - (wp.dot(o, o) - radius * radius)
    if discriminant < 0.0:
        return -1.0, wp.vec3f(0.0)
    t = -b - wp.sqrt(discriminant)
    return t, wp.normalize(o + d * t)


@wp.func
def intersect_box(o: wp.vec3f, d: wp.vec3f, half: wp.vec3f):
    inv = wp.vec3f(safe_inverse(d[0]), safe_inverse(d[1]), safe_inverse(d[2]))
    t1 = wp.cw_mul(-half - o, inv)
    t2 = wp.cw_mul(half - o, inv)
    tmin = wp.min(t1, t2)
    tmax = wp.max(t1, t2)
    t_near = wp.max(wp.max(tmin[0], tmin[1]), tmin[2])
    t_far = wp.min(wp.min(tmax[0], tmax[1]), tmax[2])
    if t_near > t_far or t_far < 0.0:
        return -1.0, wp.vec3f(0.0)
    normal = wp.vec3f(0.0, 0.0, -wp.sign(d[2]))
    if t_near == tmin[0]:
        normal = wp.vec3f(-wp.sign(d[0]), 0.0, 0.0)
    elif t_near == tmin[1]:
        normal = wp.vec3f(0.0, -wp.sign(d[1]), 0.0)
    return t_near, normal


@wp.func
def intersect_capsule(o: wp.vec3f, d: wp.vec3f, radius: float, half_height: float):
    pa = wp.vec3f(0.0, 0.0, -half_height)
    ba = wp.vec3f(0.0, 0.0, 2.0 * half_height)
    oa = o - pa
    baba = wp.dot(ba, ba)
    bard = wp.dot(ba, d)
    baoa = wp.dot(ba, oa)
    a = baba - bard * bard
    b = baba * wp.dot(d, oa) - baoa * bard
    c = baba * wp.dot(oa, oa) - baoa * baoa - radius * radius * baba
    discriminant = b * b - a * c
    if discriminant < 0.0:
        return -1.0, wp.vec3f(0.0)
    t = (-b - wp.sqrt(discriminant)) / a
    y = baoa + t * bard
    if y <= 0.0 or y >= baba:
        # The ray misses the side, so test the nearer hemispherical cap.
        oc = oa
        if y > 0.0:
            oc = o - (pa + ba)
        bc = wp.dot(d, oc)
        cap_discriminant = bc * bc - (wp.dot(oc, oc) - radius * radius)
        if cap_discriminant < 0.0:
            return -1.0, wp.vec3f(0.0)
        t = -bc - wp.sqrt(cap_discriminant)
    p = o + d * t - pa
    s = wp.clamp(wp.dot(p, ba) / baba, 0.0, 1.0)
    return t, (p - ba * s) / radius


@wp.func
def intersect_cylinder(o: wp.vec3f, d: wp.vec3f, radius: float, half_height: float):
    pa = wp.vec3f(0.0, 0.0, -half_height)
    ba = wp.vec3f(0.0, 0.0, 2.0 * half_height)
    oc = o - pa
    baba = wp.dot(ba, ba)
    bard = wp.dot(ba, d)
    baoc = wp.dot(ba, oc)
    k2 = baba - bard * bard
    k1 = baba * wp.dot(oc, d) - baoc * bard
    k0 = baba * wp.dot(oc, oc) - baoc * baoc - radius * radius * baba
    discriminant = k1 * k1 - k2 * k0
    if discriminant < 0.0:
        return -1.0, wp.vec3f(0.0)
    sqrt_discriminant = wp.sqrt(discriminant)
    t = (-k1 - sqrt_discriminant) / k2
    y = baoc + t * bard
    if y > 0.0 and y < baba:
        return t, (oc + d * t - ba * (y / baba)) / radius
    cap = 0.0
    if y >= baba:
        cap = baba
    t = (cap - baoc) / bard
    if wp.abs(k1 + k2 * t) < sqrt_discriminant:
        return t, ba * (wp.sign(y) / wp.sqrt(baba))
    return -1.0, wp.vec3f(0.0)


@wp.func
def intersect_plane(o: wp.vec3f, d: wp.vec3f):
    if o[2] <= 0.0 or d[2] >= 0.0:
        return -1.0, wp.vec3f(0.0)
    return -o[2] / d[2], wp.vec3f(0.0, 0.0, 1.0)


@wp.func
def intersect_mesh(
    o: wp.vec3f, d: wp.vec3f, scale: wp.vec3f, mesh: wp.uint64, normals: wp.array[wp.vec3f], max_t: float
):
    inv_scale = wp.vec3f(safe_inverse(scale[0]), safe_inverse(scale[1]), safe_inverse(scale[2]))
    o_mesh = wp.cw_mul(o, inv_scale)
    d_mesh = wp.cw_mul(d, inv_scale)
    d_length = wp.length(d_mesh)
    d_mesh = d_mesh / d_length
    query = wp.mesh_query_ray(mesh, o_mesh, d_mesh, max_t * d_length)
    if not query.result or wp.dot(d_mesh, query.normal) >= 0.0:
        return -1.0, wp.vec3f(0.0), 0.0, 0.0, -1
    face = query.face
    i0 = wp.mesh_get_index(mesh, face * 3 + 0)
    i1 = wp.mesh_get_index(mesh, face * 3 + 1)
    i2 = wp.mesh_get_index(mesh, face * 3 + 2)
    n = normals[i0] * query.u + normals[i1] * query.v + normals[i2] * (1.0 - query.u - query.v)
    return query.t / d_length, wp.normalize(wp.cw_mul(n, inv_scale)), query.u, query.v, face


@wp.func
def intersect_shape(
    shape_type: int,
    size: wp.vec3f,
    xform: wp.transformf,
    mesh: wp.uint64,
    normals: wp.array[wp.vec3f],
    origin: wp.vec3f,
    direction: wp.vec3f,
    max_t: float,
):
    inv = wp.transform_inverse(xform)
    o = wp.transform_point(inv, origin)
    d = wp.transform_vector(inv, direction)
    t = float(-1.0)
    n = wp.vec3f(0.0)
    u = float(0.0)
    v = float(0.0)
    face = int(-1)
    if shape_type == SHAPE_MESH:
        t, n, u, v, face = intersect_mesh(o, d, size, mesh, normals, max_t)
    elif shape_type == SHAPE_SPHERE:
        t, n = intersect_sphere(o, d, size[0])
    elif shape_type == SHAPE_BOX:
        t, n = intersect_box(o, d, size)
    elif shape_type == SHAPE_CAPSULE:
        t, n = intersect_capsule(o, d, size[0], size[1])
    elif shape_type == SHAPE_CYLINDER:
        t, n = intersect_cylinder(o, d, size[0], size[1])
    elif shape_type == SHAPE_PLANE:
        t, n = intersect_plane(o, d)
    return t, wp.transform_vector(xform, n), u, v, face


@wp.func
def srgb_to_linear(c: wp.vec3f) -> wp.vec3f:
    out = wp.vec3f(0.0)
    for i in range(3):
        if c[i] <= 0.04045:
            out[i] = c[i] / 12.92
        else:
            out[i] = wp.pow((c[i] + 0.055) / 1.055, 2.4)
    return out


@wp.func
def linear_to_srgb(c: wp.vec3f) -> wp.vec3f:
    out = wp.vec3f(0.0)
    for i in range(3):
        x = wp.clamp(c[i], 0.0, 1.0)
        if x <= 0.0031308:
            out[i] = x * 12.92
        else:
            out[i] = 1.055 * wp.pow(x, 1.0 / 2.4) - 0.055
    return out


@wp.func
def texture_rgb(tex: wp.Texture2D, uv: wp.vec2f) -> wp.vec3f:
    color = wp.texture_sample(tex, wp.vec2f(uv[0], 1.0 - uv[1]), dtype=wp.vec4f)
    return wp.vec3f(color[0], color[1], color[2])


@wp.func
def texture_cubic_projection(
    tex: wp.Texture2D, xform: wp.transformf, hit_point: wp.vec3f, normal: wp.vec3f
) -> wp.vec3f:
    # Project onto the face that best matches the local normal.
    inv = wp.transform_inverse(xform)
    local = wp.transform_point(inv, hit_point)
    n = wp.transform_vector(inv, normal)
    uv = wp.vec2f(local[0], local[1])
    if wp.abs(n[0]) >= wp.abs(n[1]) and wp.abs(n[0]) >= wp.abs(n[2]):
        uv = wp.vec2f(local[1], local[2])
    elif wp.abs(n[1]) >= wp.abs(n[2]):
        uv = wp.vec2f(local[0], local[2])
    return texture_rgb(tex, uv * TEXTURE_REPEAT)


@wp.func
def surface_texture(
    tex: wp.Texture2D,
    shape_type: int,
    xform: wp.transformf,
    mesh: wp.uint64,
    uvs: wp.array[wp.vec2f],
    hit_point: wp.vec3f,
    normal: wp.vec3f,
    u: float,
    v: float,
    face: int,
) -> wp.vec3f:
    # Separate call sites per projection, as in Newton, so the kernel inlines
    # several texture fetches.
    if shape_type == SHAPE_PLANE:
        local = wp.transform_point(wp.transform_inverse(xform), hit_point)
        return texture_rgb(tex, wp.vec2f(local[0], local[1]) * TEXTURE_REPEAT)
    if shape_type == SHAPE_MESH:
        if face < 0:
            return wp.vec3f(1.0)
        if uvs.shape[0] > 0:
            i0 = wp.mesh_get_index(mesh, face * 3 + 0)
            i1 = wp.mesh_get_index(mesh, face * 3 + 1)
            i2 = wp.mesh_get_index(mesh, face * 3 + 2)
            uv = uvs[i0] * u + uvs[i1] * v + uvs[i2] * (1.0 - u - v)
            return texture_rgb(tex, uv * TEXTURE_REPEAT)
        return texture_cubic_projection(tex, xform, hit_point, normal)
    return texture_cubic_projection(tex, xform, hit_point, normal)


@wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": True})
def render_shapes(
    width: int,
    camera: wp.transformf,
    tan_half_fov: float,
    bvh: wp.uint64,
    shape_types: wp.array[int],
    shape_sizes: wp.array[wp.vec3f],
    shape_xforms: wp.array[wp.transformf],
    shape_colors: wp.array[wp.vec3f],
    shape_texture_ids: wp.array[int],
    shape_meshes: wp.array[wp.uint64],
    mesh_normals: wp.array[wp.vec3f],
    mesh_uvs: wp.array[wp.vec2f],
    textures: wp.array[wp.Texture2D],
    light_direction: wp.vec3f,
    light_color: wp.vec3f,
    sky_color: wp.vec3f,
    ground_color: wp.vec3f,
    out_color: wp.array[wp.uint32],
    out_depth: wp.array[float],
):
    tid = wp.tid()
    x = float(tid % width) + 0.5
    y = float(tid // width) + 0.5
    scale = 2.0 / float(width)
    ray_camera = wp.vec3f((x * scale - 1.0) * tan_half_fov, (1.0 - y * scale) * tan_half_fov, -1.0)
    origin = wp.transform_get_translation(camera)
    direction = wp.normalize(wp.transform_vector(camera, ray_camera))

    closest = float(MAX_DISTANCE)
    hit_shape = int(-1)
    hit_normal = wp.vec3f(0.0)
    hit_u = float(0.0)
    hit_v = float(0.0)
    hit_face = int(-1)
    query = wp.bvh_query_ray(bvh, origin, direction)
    shape = int(0)
    while wp.bvh_query_next(query, shape, closest):
        t, n, u, v, face = intersect_shape(
            shape_types[shape],
            shape_sizes[shape],
            shape_xforms[shape],
            shape_meshes[shape],
            mesh_normals,
            origin,
            direction,
            closest,
        )
        if t >= 0.0 and t < closest:
            closest = t
            hit_shape = shape
            hit_normal = n
            hit_u = u
            hit_v = v
            hit_face = face

    if hit_shape < 0:
        out_color[tid] = wp.uint32(0xFF000000)
        out_depth[tid] = 0.0
        return

    hit_point = origin + direction * closest
    normal = wp.normalize(hit_normal)
    albedo = srgb_to_linear(shape_colors[hit_shape])
    texture_id = shape_texture_ids[hit_shape]
    if texture_id >= 0:
        tex_color = surface_texture(
            textures[texture_id],
            shape_types[hit_shape],
            shape_xforms[hit_shape],
            shape_meshes[hit_shape],
            mesh_uvs,
            hit_point,
            normal,
            hit_u,
            hit_v,
            hit_face,
        )
        albedo = wp.cw_mul(albedo, srgb_to_linear(tex_color))

    hemisphere = 0.5 * (normal[2] + 1.0)
    color = wp.cw_mul(albedo, sky_color * hemisphere + ground_color * (1.0 - hemisphere))

    to_light = -light_direction
    n_dot_l = wp.dot(normal, to_light)
    if n_dot_l > 0.0:
        shadow_origin = hit_point + normal * 1.0e-3
        occluded = bool(False)
        shadow_query = wp.bvh_query_ray(bvh, shadow_origin, to_light)
        blocker = int(0)
        while wp.bvh_query_next(shadow_query, blocker, MAX_DISTANCE):
            t, n, u, v, face = intersect_shape(
                shape_types[blocker],
                shape_sizes[blocker],
                shape_xforms[blocker],
                shape_meshes[blocker],
                mesh_normals,
                shadow_origin,
                to_light,
                MAX_DISTANCE,
            )
            if t >= 0.0:
                occluded = True
                break
        if not occluded:
            color += wp.cw_mul(albedo, light_color) * n_dot_l

    srgb = linear_to_srgb(color)
    r = wp.uint32(srgb[0] * 255.0 + 0.5)
    g = wp.uint32(srgb[1] * 255.0 + 0.5)
    b = wp.uint32(srgb[2] * 255.0 + 0.5)
    out_color[tid] = wp.uint32(0xFF000000) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r
    out_depth[tid] = closest


def _look_at(eye, target) -> wp.transformf:
    """Return a camera-to-world transform with the camera looking down its -Z axis."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack([right, up, -forward], axis=1).astype(np.float32)
    return wp.transformf(wp.vec3f(*eye.astype(np.float32)), wp.quat_from_matrix(wp.mat33f(*rotation.flatten())))


def _make_sphere_mesh(rings: int = 12, segments: int = 24):
    """Return points, texture coordinates, and triangle indices of a unit UV sphere."""
    theta, phi = np.meshgrid(
        np.linspace(0.0, np.pi, rings + 1), np.linspace(0.0, 2.0 * np.pi, segments + 1), indexing="ij"
    )
    points = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    uvs = np.stack([phi / (2.0 * np.pi), theta / np.pi], axis=-1)
    a = (np.arange(rings)[:, None] * (segments + 1) + np.arange(segments)[None, :]).ravel()
    b = a + segments + 1
    indices = np.stack([a, b, a + 1, a + 1, b, b + 1], axis=-1).ravel()
    return points.reshape(-1, 3).astype(np.float32), uvs.reshape(-1, 2).astype(np.float32), indices.astype(np.int32)


class TextureRender:
    """Render color and depth for a grid of shapes in one large kernel, as camera sensors do."""

    params = ["untextured", "textured"]
    param_names = ["surfaces"]
    timer = CudaEventTimer()
    # Calibrated on a Blackwell MIG slice. Averaging several calls per sample
    # cost more for the same run-to-run precision, fewer samples let single
    # outliers set the confidence interval, and more samples stopped helping.
    repeat = 16
    number = 1

    @setup_once
    def setup(self, surfaces):
        wp.init()
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(42)

        # Shape 0 is the ground plane. The rest sit on a grid with random
        # orientations. Primitive sizes match Newton's 256-shape camera
        # benchmark scene, which has no meshes; the mesh spheres are added
        # here to exercise the mesh code path.
        count = GRID_SIZE * GRID_SIZE + 1
        types = np.concatenate([[SHAPE_PLANE], rng.integers(SHAPE_SPHERE, SHAPE_MESH + 1, count - 1)])
        sizes = np.zeros((count, 3), dtype=np.float32)
        radii = np.zeros(count, dtype=np.float32)
        for shape_type, size in (
            (SHAPE_SPHERE, (0.18, 0.0, 0.0)),
            (SHAPE_BOX, (0.16, 0.16, 0.16)),
            (SHAPE_CAPSULE, (0.1, 0.15, 0.0)),
            (SHAPE_CYLINDER, (0.14, 0.15, 0.0)),
            (SHAPE_MESH, (0.18, 0.18, 0.18)),
        ):
            mask = types == shape_type
            sizes[mask] = size
        radii[types == SHAPE_SPHERE] = 0.18
        radii[types == SHAPE_BOX] = 0.16 * math.sqrt(3.0)
        radii[types == SHAPE_CAPSULE] = 0.25
        radii[types == SHAPE_CYLINDER] = math.hypot(0.14, 0.15)
        radii[types == SHAPE_MESH] = 0.18

        coords = (np.arange(GRID_SIZE) - 0.5 * (GRID_SIZE - 1)) * 0.6
        grid_x, grid_y = np.meshgrid(coords, coords, indexing="ij")
        positions = np.zeros((count, 3), dtype=np.float32)
        positions[1:, 0] = grid_x.ravel()
        positions[1:, 1] = grid_y.ravel()
        positions[1:, 2] = 0.3
        rotations = np.zeros((count, 4), dtype=np.float32)
        rotations[0, 3] = 1.0
        quats = rng.normal(size=(count - 1, 4))
        rotations[1:] = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        lowers = positions - radii[:, None]
        uppers = positions + radii[:, None]
        lowers[0] = (-50.0, -50.0, -0.01)
        uppers[0] = (50.0, 50.0, 0.01)
        self.lowers = wp.array(lowers, dtype=wp.vec3f, device=device)
        self.uppers = wp.array(uppers, dtype=wp.vec3f, device=device)
        self.bvh = wp.Bvh(self.lowers, self.uppers)

        points, uvs, indices = _make_sphere_mesh()
        self.mesh_points = wp.array(points, dtype=wp.vec3f, device=device)
        self.mesh_indices = wp.array(indices, dtype=wp.int32, device=device)
        self.mesh = wp.Mesh(self.mesh_points, self.mesh_indices)
        # Unit-sphere points double as smooth vertex normals.
        self.mesh_normals = wp.array(points, dtype=wp.vec3f, device=device)
        self.mesh_uvs = wp.array(uvs, dtype=wp.vec2f, device=device)

        self.textures = [
            wp.Texture2D(
                rng.integers(0, 256, (TEXTURE_SIZE, TEXTURE_SIZE, 4), dtype=np.uint8),
                filter_mode=wp.TextureFilterMode.LINEAR,
                address_mode=wp.TextureAddressMode.WRAP,
                device=device,
            )
            for _ in range(NUM_TEXTURES)
        ]
        if surfaces == "textured":
            texture_ids = np.arange(count, dtype=np.int32) % NUM_TEXTURES
        else:
            # The sampling code stays compiled into the kernel but never runs.
            texture_ids = np.full(count, -1, dtype=np.int32)

        light = np.array([-0.5, -0.3, -0.8], dtype=np.float32)
        self.out_color = wp.empty(RESOLUTION * RESOLUTION, dtype=wp.uint32, device=device)
        self.out_depth = wp.empty(RESOLUTION * RESOLUTION, dtype=float, device=device)
        self.command = wp.launch(
            render_shapes,
            dim=RESOLUTION * RESOLUTION,
            inputs=[
                RESOLUTION,
                _look_at((9.0, 9.0, 6.0), (0.0, 0.0, 0.0)),
                math.tan(math.radians(45.0) * 0.5),
                self.bvh.id,
                wp.array(types, dtype=int, device=device),
                wp.array(sizes, dtype=wp.vec3f, device=device),
                wp.array(np.concatenate([positions, rotations], axis=1), dtype=wp.transformf, device=device),
                wp.array(rng.random((count, 3), dtype=np.float32), dtype=wp.vec3f, device=device),
                wp.array(texture_ids, dtype=int, device=device),
                wp.array(
                    np.where(types == SHAPE_MESH, self.mesh.id, 0).astype(np.uint64), dtype=wp.uint64, device=device
                ),
                self.mesh_normals,
                self.mesh_uvs,
                wp.array(self.textures, dtype=wp.Texture2D, device=device),
                wp.vec3f(*(light / np.linalg.norm(light))),
                wp.vec3f(1.0, 0.95, 0.9),
                wp.vec3f(0.35, 0.4, 0.5),
                wp.vec3f(0.2, 0.18, 0.15),
            ],
            outputs=[self.out_color, self.out_depth],
            device=device,
            block_dim=BLOCK_DIM,
            record_cmd=True,
        )
        for _ in range(5):
            self.command.launch()

        # Guard against measuring an empty frame: the camera should see the
        # ground plane or a shape in almost every pixel.
        depth = self.out_depth.numpy()
        if np.count_nonzero(depth > 0.0) < 0.9 * depth.size:
            raise RuntimeError("TextureRender camera rays missed the scene")

    def time_cuda(self, surfaces):
        self.command.launch()
