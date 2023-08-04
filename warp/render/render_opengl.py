# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import time

import warp as wp
from .utils import tab10_color_map

from collections import defaultdict
from typing import List, Tuple, Union, Optional

import numpy as np
import ctypes

Mat44 = Union[List[float], List[List[float]], np.ndarray]

wp.set_module_options({"enable_backward": False})

shape_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

// column vectors of the instance transform matrix
layout (location = 3) in vec4 aInstanceTransform0;
layout (location = 4) in vec4 aInstanceTransform1;
layout (location = 5) in vec4 aInstanceTransform2;
layout (location = 6) in vec4 aInstanceTransform3;

// colors to use for the checkerboard pattern
layout (location = 7) in vec3 aObjectColor1;
layout (location = 8) in vec3 aObjectColor2;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoord;
out vec3 ObjectColor1;
out vec3 ObjectColor2;

void main()
{
    mat4 transform = model * mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);
    vec4 worldPos = transform * vec4(aPos, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = vec3(worldPos);
    Normal = mat3(transpose(inverse(transform))) * aNormal;
    TexCoord = aTexCoord;
    ObjectColor1 = aObjectColor1;
    ObjectColor2 = aObjectColor2;
}
"""

shape_fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;
in vec3 ObjectColor1;
in vec3 ObjectColor2;

uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 sunDirection;

void main()
{
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    vec3 norm = normalize(Normal);

    float diff = max(dot(norm, sunDirection), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 lightDir2 = normalize(vec3(1.0, 0.3, -0.3));
    diff = max(dot(norm, lightDir2), 0.0);
    diffuse += diff * lightColor * 0.3;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 reflectDir = reflect(-sunDirection, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    reflectDir = reflect(-lightDir2, norm);
    spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    specular += specularStrength * spec * lightColor * 0.3;

    // checkerboard pattern
    float u = TexCoord.x;
    float v = TexCoord.y;
    // blend the checkerboard pattern dependent on the gradient of the texture coordinates
    // to void Moire patterns
    vec2 grad = abs(dFdx(TexCoord)) + abs(dFdy(TexCoord));
    float blendRange = 1.5;
    float blendFactor = max(grad.x, grad.y) * blendRange;
    float scale = 2.0;
    float checker = mod(floor(u * scale) + floor(v * scale), 2.0);
    checker = mix(checker, 0.5, smoothstep(0.0, 1.0, blendFactor));
    vec3 checkerColor = mix(ObjectColor1, ObjectColor2, checker);

    vec3 result = (ambient + diffuse + specular) * checkerColor;
    FragColor = vec4(result, 1.0);
}
"""

grid_vertex_shader = """
#version 330 core

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

in vec3 position;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

# Fragment shader source code
grid_fragment_shader = """
#version 330 core

out vec4 outColor;

void main() {
    outColor = vec4(0.5, 0.5, 0.5, 1.0);
}
"""

sky_vertex_shader = """
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;
uniform vec3 viewPos;

out vec3 FragPos;
out vec2 TexCoord;

void main()
{
    vec4 worldPos = vec4(aPos + viewPos, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = vec3(worldPos);
    TexCoord = aTexCoord;
}
"""

sky_fragment_shader = """
#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoord;

uniform vec3 color1;
uniform vec3 color2;

uniform vec3 sunDirection;

void main()
{
    float y = tanh(FragPos.y*0.01)*0.5+0.5;
    float height = sqrt(1.0-y);

    float s = pow(0.5, 1.0 / 10.0);
    s = 1.0 - clamp(s, 0.75, 1.0);

    vec3 haze = mix(vec3(1.0), color2 * 1.3, s);
    vec3 sky = mix(color1, haze, height / 1.3);

    float diff = max(dot(sunDirection, normalize(FragPos)), 0.0);
    vec3 sun = pow(diff, 32) * vec3(1.0, 0.8, 0.6) * 0.5;

    FragColor = vec4(sky + sun, 1.0);
}
"""

frame_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

frame_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D textureSampler;

void main() {
    FragColor = texture(textureSampler, TexCoord);
}
"""


@wp.kernel
def update_vbo_transforms(
    instance_id: wp.array(dtype=int),
    instance_body: wp.array(dtype=int),
    instance_transforms: wp.array(dtype=wp.transform),
    instance_scalings: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    i = instance_id[tid]
    X_ws = instance_transforms[i]
    if instance_body:
        body = instance_body[i]
        if body >= 0:
            if body_q:
                X_ws = body_q[body] * X_ws
            else:
                return
    p = wp.transform_get_translation(X_ws)
    q = wp.transform_get_rotation(X_ws)
    s = instance_scalings[i]
    rot = wp.quat_to_matrix(q)
    # transposed definition
    vbo_transforms[tid] = wp.mat44(
        rot[0, 0] * s[0],
        rot[1, 0] * s[0],
        rot[2, 0] * s[0],
        0.0,
        rot[0, 1] * s[1],
        rot[1, 1] * s[1],
        rot[2, 1] * s[1],
        0.0,
        rot[0, 2] * s[2],
        rot[1, 2] * s[2],
        rot[2, 2] * s[2],
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


@wp.kernel
def update_vbo_vertices(
    points: wp.array(dtype=wp.vec3),
    # outputs
    vbo_vertices: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    p = points[tid]
    vbo_vertices[tid, 0] = p[0]
    vbo_vertices[tid, 1] = p[1]
    vbo_vertices[tid, 2] = p[2]


@wp.kernel
def update_points_positions(
    instance_positions: wp.array(dtype=wp.vec3),
    instance_scalings: wp.array(dtype=wp.vec3),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    p = instance_positions[tid]
    s = wp.vec3(1.0)
    if instance_scalings:
        s = instance_scalings[tid]
    # transposed definition
    # fmt: off
    vbo_transforms[tid] = wp.mat44(
        s[0],  0.0,  0.0, 0.0,
         0.0, s[1],  0.0, 0.0,
         0.0,  0.0, s[2], 0.0,
        p[0], p[1], p[2], 1.0)
    # fmt: on


@wp.kernel
def update_line_transforms(
    lines: wp.array(dtype=wp.vec3, ndim=2),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    p0 = lines[tid, 0]
    p1 = lines[tid, 1]
    p = 0.5 * (p0 + p1)
    d = p1 - p0
    s = wp.length(d)
    axis = wp.normalize(d)
    y_up = wp.vec3(0.0, 1.0, 0.0)
    angle = wp.acos(wp.dot(axis, y_up))
    axis = wp.normalize(wp.cross(axis, y_up))
    q = wp.quat_from_axis_angle(axis, -angle)
    rot = wp.quat_to_matrix(q)
    # transposed definition
    # fmt: off
    vbo_transforms[tid] = wp.mat44(
            rot[0, 0],     rot[1, 0],     rot[2, 0], 0.0,
        s * rot[0, 1], s * rot[1, 1], s * rot[2, 1], 0.0,
            rot[0, 2],     rot[1, 2],     rot[2, 2], 0.0,
                 p[0],          p[1],          p[2], 1.0,
    )
    # fmt: on


@wp.kernel
def compute_gfx_vertices(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    # outputs
    gfx_vertices: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    v0 = vertices[indices[tid, 0]]
    v1 = vertices[indices[tid, 1]]
    v2 = vertices[indices[tid, 2]]
    i = tid * 3
    j = i + 1
    k = i + 2
    gfx_vertices[i, 0] = v0[0]
    gfx_vertices[i, 1] = v0[1]
    gfx_vertices[i, 2] = v0[2]
    gfx_vertices[j, 0] = v1[0]
    gfx_vertices[j, 1] = v1[1]
    gfx_vertices[j, 2] = v1[2]
    gfx_vertices[k, 0] = v2[0]
    gfx_vertices[k, 1] = v2[1]
    gfx_vertices[k, 2] = v2[2]
    n = wp.normalize(wp.cross(v1 - v0, v2 - v0))
    gfx_vertices[i, 3] = n[0]
    gfx_vertices[i, 4] = n[1]
    gfx_vertices[i, 5] = n[2]
    gfx_vertices[j, 3] = n[0]
    gfx_vertices[j, 4] = n[1]
    gfx_vertices[j, 5] = n[2]
    gfx_vertices[k, 3] = n[0]
    gfx_vertices[k, 4] = n[1]
    gfx_vertices[k, 5] = n[2]


@wp.kernel
def compute_average_normals(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3),
    # outputs
    normals: wp.array(dtype=wp.vec3),
    faces_per_vertex: wp.array(dtype=int),
):
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    v0 = vertices[i]
    v1 = vertices[j]
    v2 = vertices[k]
    n = wp.normalize(wp.cross(v1 - v0, v2 - v0))
    wp.atomic_add(normals, i, n)
    wp.atomic_add(faces_per_vertex, i, 1)
    wp.atomic_add(normals, j, n)
    wp.atomic_add(faces_per_vertex, j, 1)
    wp.atomic_add(normals, k, n)
    wp.atomic_add(faces_per_vertex, k, 1)


@wp.kernel
def assemble_gfx_vertices(
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    normals: wp.array(dtype=wp.vec3),
    faces_per_vertex: wp.array(dtype=int),
    # outputs
    gfx_vertices: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    v = vertices[tid]
    n = normals[tid] / float(faces_per_vertex[tid])
    gfx_vertices[tid, 0] = v[0]
    gfx_vertices[tid, 1] = v[1]
    gfx_vertices[tid, 2] = v[2]
    gfx_vertices[tid, 3] = n[0]
    gfx_vertices[tid, 4] = n[1]
    gfx_vertices[tid, 5] = n[2]


@wp.kernel
def copy_frame(
    input_img: wp.array(dtype=wp.uint8),
    width: int,
    height: int,
    # outputs
    output_img: wp.array(dtype=float, ndim=3),
):
    w, v = wp.tid()
    pixel = v * width + w
    pixel *= 3
    r = float(input_img[pixel + 0])
    g = float(input_img[pixel + 1])
    b = float(input_img[pixel + 2])
    # flip vertically (OpenGL coordinates start at bottom)
    v = height - v - 1
    output_img[v, w, 0] = r / 255.0
    output_img[v, w, 1] = g / 255.0
    output_img[v, w, 2] = b / 255.0


@wp.kernel
def copy_frame_tiles(
    input_img: wp.array(dtype=wp.uint8),
    positions: wp.array(dtype=int, ndim=2),
    screen_width: int,
    screen_height: int,
    tile_height: int,
    # outputs
    output_img: wp.array(dtype=float, ndim=4),
):
    tile, x, y = wp.tid()
    p = positions[tile]
    qx = x + p[0]
    qy = y + p[1]
    pixel = qy * screen_width + qx
    # flip vertically (OpenGL coordinates start at bottom)
    y = tile_height - y - 1
    if qx >= screen_width or qy >= screen_height:
        output_img[tile, y, x, 0] = 0.0
        output_img[tile, y, x, 1] = 0.0
        output_img[tile, y, x, 2] = 0.0
        return  # prevent out-of-bounds access
    pixel *= 3
    r = float(input_img[pixel + 0])
    g = float(input_img[pixel + 1])
    b = float(input_img[pixel + 2])
    output_img[tile, y, x, 0] = r / 255.0
    output_img[tile, y, x, 1] = g / 255.0
    output_img[tile, y, x, 2] = b / 255.0


@wp.kernel
def copy_frame_tile(
    input_img: wp.array(dtype=wp.uint8),
    offset_x: int,
    offset_y: int,
    screen_width: int,
    screen_height: int,
    tile_height: int,
    # outputs
    output_img: wp.array(dtype=float, ndim=4),
):
    tile, x, y = wp.tid()
    qx = x + offset_x
    qy = y + offset_y
    pixel = qy * screen_width + qx
    # flip vertically (OpenGL coordinates start at bottom)
    y = tile_height - y - 1
    if qx >= screen_width or qy >= screen_height:
        output_img[tile, y, x, 0] = 0.0
        output_img[tile, y, x, 1] = 0.0
        output_img[tile, y, x, 2] = 0.0
        return  # prevent out-of-bounds access
    pixel *= 3
    r = float(input_img[pixel + 0])
    g = float(input_img[pixel + 1])
    b = float(input_img[pixel + 2])
    output_img[tile, y, x, 0] = r / 255.0
    output_img[tile, y, x, 1] = g / 255.0
    output_img[tile, y, x, 2] = b / 255.0


def check_gl_error():
    from pyglet import gl

    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print(f"OpenGL error: {error}")


class ShapeInstancer:
    """
    Handles instanced rendering for a mesh.
    Note the vertices must be in the 8-dimensional format:
        [3D point, 3D normal, UV texture coordinates]
    """

    def __init__(self, shape_shader, device):
        self.shape_shader = shape_shader
        self.device = device
        self.face_count = 0
        self.vao = None
        self.instance_transform_gl_buffer = None
        self.instance_color1_buffer = None
        self.instance_color2_buffer = None
        self.color1 = (1.0, 1.0, 1.0)
        self.color2 = (0.0, 0.0, 0.0)
        self.num_instances = 0
        self.transforms = None
        self.scalings = None
        self._instance_transform_cuda_buffer = None

    def __del__(self):
        from pyglet import gl

        if self.instance_transform_gl_buffer is not None:
            try:
                gl.glDeleteBuffers(1, self.instance_transform_gl_buffer)
                gl.glDeleteBuffers(1, self.instance_color1_buffer)
                gl.glDeleteBuffers(1, self.instance_color2_buffer)
            except gl.GLException:
                pass
        if self.vao is not None:
            try:
                gl.glDeleteVertexArrays(1, self.vao)
                gl.glDeleteBuffers(1, self.vbo)
                gl.glDeleteBuffers(1, self.ebo)
            except gl.GLException:
                pass

    def register_shape(self, vertices, indices, color1=(1.0, 1.0, 1.0), color2=(0.0, 0.0, 0.0)):
        from pyglet import gl

        if color1 is not None and color2 is None:
            color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
        self.color1 = color1
        self.color2 = color2

        gl.glUseProgram(self.shape_shader.id)

        # Create VAO, VBO, and EBO
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.GLuint()
        gl.glGenBuffers(1, self.vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

        self.ebo = gl.GLuint()
        gl.glGenBuffers(1, self.ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        self.face_count = len(indices)

    def allocate_instances(self, positions, rotations=None, colors1=None, colors2=None, scalings=None):
        from pyglet import gl

        gl.glBindVertexArray(self.vao)

        self.num_instances = len(positions)

        # Create instance buffer and bind it as an instanced array
        if self.instance_transform_gl_buffer is None:
            self.instance_transform_gl_buffer = gl.GLuint()
            gl.glGenBuffers(1, self.instance_transform_gl_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_gl_buffer)

        self.instance_ids = wp.array(np.arange(self.num_instances), dtype=wp.int32, device=self.device)
        if rotations is None:
            self.instance_transforms = wp.array(
                [(*pos, 0.0, 0.0, 0.0, 1.0) for pos in positions], dtype=wp.transform, device=self.device
            )
        else:
            self.instance_transforms = wp.array(
                [(*pos, *rot) for pos, rot in zip(positions, rotations)], dtype=wp.transform, device=self.device
            )

        if scalings is None:
            self.instance_scalings = wp.array(
                np.tile((1.0, 1.0, 1.0), (self.num_instances, 1)), dtype=wp.vec3, device=self.device
            )
        else:
            self.instance_scalings = wp.array(scalings, dtype=wp.vec3, device=self.device)

        vbo_transforms = wp.zeros(dtype=wp.mat44, shape=(self.num_instances,), device=self.device)

        wp.launch(
            update_vbo_transforms,
            dim=self.num_instances,
            inputs=[
                self.instance_ids,
                None,
                self.instance_transforms,
                self.instance_scalings,
                None,
            ],
            outputs=[
                vbo_transforms,
            ],
            device=self.device,
        )

        vbo_transforms = vbo_transforms.numpy()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vbo_transforms.nbytes, vbo_transforms.ctypes.data, gl.GL_DYNAMIC_DRAW)

        # Create CUDA buffer for instance transforms
        self._instance_transform_cuda_buffer = wp.RegisteredGLBuffer(
            int(self.instance_transform_gl_buffer.value), self.device
        )

        if colors1 is None:
            colors1 = np.tile(self.color1, (self.num_instances, 1))
        if colors2 is None:
            colors2 = np.tile(self.color2, (self.num_instances, 1))
        colors1 = np.array(colors1, dtype=np.float32)
        colors2 = np.array(colors2, dtype=np.float32)

        # create buffer for checkerboard colors
        if self.instance_color1_buffer is None:
            self.instance_color1_buffer = gl.GLuint()
            gl.glGenBuffers(1, self.instance_color1_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color1_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors1.nbytes, colors1.ctypes.data, gl.GL_STATIC_DRAW)

        if self.instance_color2_buffer is None:
            self.instance_color2_buffer = gl.GLuint()
            gl.glGenBuffers(1, self.instance_color2_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color2_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors2.nbytes, colors2.ctypes.data, gl.GL_STATIC_DRAW)

        # Set up instance attribute pointers
        matrix_size = vbo_transforms[0].nbytes

        gl.glBindVertexArray(self.vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_gl_buffer)

        # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
        for i in range(4):
            gl.glVertexAttribPointer(
                3 + i, 4, gl.GL_FLOAT, gl.GL_FALSE, matrix_size, ctypes.c_void_p(i * matrix_size // 4)
            )
            gl.glEnableVertexAttribArray(3 + i)
            gl.glVertexAttribDivisor(3 + i, 1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color1_buffer)
        gl.glVertexAttribPointer(7, 3, gl.GL_FLOAT, gl.GL_FALSE, colors1[0].nbytes, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(7)
        gl.glVertexAttribDivisor(7, 1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color2_buffer)
        gl.glVertexAttribPointer(8, 3, gl.GL_FLOAT, gl.GL_FALSE, colors2[0].nbytes, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(8)
        gl.glVertexAttribDivisor(8, 1)

        gl.glBindVertexArray(0)

    def update_instances(self, transforms: wp.array = None, scalings: wp.array = None, colors1=None, colors2=None):
        from pyglet import gl

        if transforms is not None:
            if transforms.device.is_cuda:
                wp_transforms = transforms
            else:
                wp_transforms = transforms.to(self.device)
            self.transforms = wp_transforms
        if scalings is not None:
            if transforms.device.is_cuda:
                wp_scalings = scalings
            else:
                wp_scalings = scalings.to(self.device)
            self.scalings = wp_scalings

        if transforms is not None or scalings is not None:
            gl.glBindVertexArray(self.vao)
            vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self.num_instances,))

            wp.launch(
                update_vbo_transforms,
                dim=self.num_instances,
                inputs=[
                    self.instance_ids,
                    None,
                    self.instance_transforms,
                    self.instance_scalings,
                    None,
                ],
                outputs=[
                    vbo_transforms,
                ],
                device=self.device,
            )

            self._instance_transform_cuda_buffer.unmap()

    def render(self):
        from pyglet import gl

        gl.glUseProgram(self.shape_shader.id)

        gl.glBindVertexArray(self.vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.face_count, gl.GL_UNSIGNED_INT, None, self.num_instances)
        gl.glBindVertexArray(0)

    # scope exposes VBO transforms to be set directly by a warp kernel
    def __enter__(self):
        from pyglet import gl

        gl.glBindVertexArray(self.vao)
        self.vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self.num_instances,))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._instance_transform_cuda_buffer.unmap()


def str_buffer(string: str):
    return ctypes.c_char_p(string.encode("utf-8"))


def arr_pointer(arr: np.ndarray):
    return arr.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class OpenGLRenderer:
    """
    OpenGLRenderer is a simple OpenGL renderer for rendering 3D shapes and meshes.
    """

    # number of segments to use for rendering spheres, capsules, cones and cylinders
    default_num_segments = 32

    def __init__(
        self,
        title="Warp sim",
        scaling=1.0,
        fps=60,
        up_axis="y",
        screen_width=1024,
        screen_height=768,
        near_plane=0.01,
        far_plane=1000.0,
        camera_fov=45.0,
        background_color=(0.53, 0.8, 0.92),
        draw_grid=True,
        draw_sky=True,
        draw_axis=True,
        show_info=True,
        render_wireframe=False,
        axis_scale=1.0,
        vsync=False,
        headless=False,
        enable_backface_culling=True,
    ):
        try:
            import pyglet

            # disable error checking for performance
            pyglet.options["debug_gl"] = False

            from pyglet import gl
            from pyglet.math import Vec3 as PyVec3
            from pyglet.graphics.shader import Shader, ShaderProgram
        except ImportError:
            raise Exception("OpenGLRenderer requires pyglet (version >= 2.0) to be installed.")

        self.camera_near_plane = near_plane
        self.camera_far_plane = far_plane
        self.camera_fov = camera_fov

        self.background_color = background_color
        self.draw_grid = draw_grid
        self.draw_sky = draw_sky
        self.draw_axis = draw_axis
        self.show_info = show_info
        self.render_wireframe = render_wireframe
        self.enable_backface_culling = enable_backface_culling

        self._device = wp.get_cuda_device()
        self._title = title

        self.window = pyglet.window.Window(
            width=screen_width, height=screen_height, caption=title, resizable=True, vsync=vsync, visible=not headless
        )
        self.app = pyglet.app

        # making window current opengl rendering context
        self.window.switch_to()

        self.screen_width, self.screen_height = self.window.get_framebuffer_size()

        self._camera_pos = PyVec3(0.0, 2.0, 10.0)
        self._camera_front = PyVec3(0.0, 0.0, -1.0)
        self._camera_up = PyVec3(0.0, 1.0, 0.0)
        self._camera_speed = 0.04
        if isinstance(up_axis, int):
            self._camera_axis = up_axis
        else:
            self._camera_axis = "xyz".index(up_axis.lower())
        self._yaw, self._pitch = -90.0, 0.0
        self._last_x, self._last_y = self.screen_width // 2, self.screen_height // 2
        self._first_mouse = True
        self._left_mouse_pressed = False
        self._keys_pressed = defaultdict(bool)

        self.update_view_matrix()
        self.update_projection_matrix()

        self._frame_dt = 1.0 / fps
        self.time = 0.0
        self._start_time = time.time()
        self.clock_time = 0.0
        self._paused = False
        self._frame_speed = 0.0
        self.skip_rendering = False
        self._skip_frame_counter = 0
        self._fps_update = 0.0
        self._fps_render = 0.0
        self._fps_alpha = 0.1  # low pass filter rate to update FPS stats

        self._body_name = {}
        self._shapes = []
        self._shape_geo_hash = {}
        self._shape_gl_buffers = {}
        self._shape_instances = defaultdict(list)
        self._instances = {}
        self._instance_shape = {}
        self._instance_gl_buffers = {}
        self._instance_transform_gl_buffer = None
        self._instance_transform_cuda_buffer = None
        self._instance_color1_buffer = None
        self._instance_color2_buffer = None
        self._instance_count = 0
        self._wp_instance_ids = None
        self._instance_ids = None
        self._inverse_instance_ids = None
        self._wp_instance_transforms = None
        self._wp_instance_scalings = None
        self._wp_instance_bodies = None
        self._update_shape_instances = False
        self._add_shape_instances = False

        # additional shape instancer used for points and line rendering
        self._shape_instancers = {}

        # instancer for the arrow shapes sof the coordinate system axes
        self._axis_instancer = None

        # toggle tiled rendering
        self._tiled_rendering = False
        self._tile_instances = None
        self._tile_ncols = 0
        self._tile_nrows = 0
        self._tile_width = 0
        self._tile_height = 0
        self._tile_viewports = None
        self._tile_view_matrices = None
        self._tile_projection_matrices = None

        self._frame_texture = None
        self._frame_fbo = None
        self._frame_pbo = None

        self.window.push_handlers(on_draw=self._draw)
        self.window.push_handlers(on_resize=self._window_resize_callback)
        self.window.push_handlers(on_key_press=self._key_press_callback)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self._key_handler)

        self.window.on_mouse_scroll = self._scroll_callback
        self.window.on_mouse_drag = self._mouse_drag_callback

        gl.glClearColor(*self.background_color, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self._shape_shader = ShaderProgram(
            Shader(shape_vertex_shader, "vertex"), Shader(shape_fragment_shader, "fragment")
        )
        self._grid_shader = ShaderProgram(
            Shader(grid_vertex_shader, "vertex"), Shader(grid_fragment_shader, "fragment")
        )

        self._sun_direction = np.array((-0.2, 0.8, 0.3))
        self._sun_direction /= np.linalg.norm(self._sun_direction)
        with self._shape_shader:
            gl.glUniform3f(
                gl.glGetUniformLocation(self._shape_shader.id, str_buffer("sunDirection")), *self._sun_direction
            )
            gl.glUniform3f(gl.glGetUniformLocation(self._shape_shader.id, str_buffer("lightColor")), 1, 1, 1)
            self._loc_shape_model = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("model"))
            self._loc_shape_view = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("view"))
            self._loc_shape_projection = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("projection"))
            self._loc_shape_view_pos = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("viewPos"))
            gl.glUniform3f(self._loc_shape_view_pos, 0, 0, 10)

        # create grid data
        limit = 10.0
        ticks = np.linspace(-limit, limit, 21)
        grid_vertices = []
        for i in ticks:
            if self._camera_axis == 0:
                grid_vertices.extend([0, -limit, i, 0, limit, i])
                grid_vertices.extend([0, i, -limit, 0, i, limit])
            elif self._camera_axis == 1:
                grid_vertices.extend([-limit, 0, i, limit, 0, i])
                grid_vertices.extend([i, 0, -limit, i, 0, limit])
            elif self._camera_axis == 2:
                grid_vertices.extend([-limit, i, 0, limit, i, 0])
                grid_vertices.extend([i, -limit, 0, i, limit, 0])
        grid_vertices = np.array(grid_vertices, dtype=np.float32)
        self._grid_vertex_count = len(grid_vertices) // 3

        with self._grid_shader:
            self._grid_vao = gl.GLuint()
            gl.glGenVertexArrays(1, self._grid_vao)
            gl.glBindVertexArray(self._grid_vao)

            self._grid_vbo = gl.GLuint()
            gl.glGenBuffers(1, self._grid_vbo)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._grid_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, grid_vertices.nbytes, grid_vertices.ctypes.data, gl.GL_STATIC_DRAW)

            self._loc_grid_view = gl.glGetUniformLocation(self._grid_shader.id, str_buffer("view"))
            self._loc_grid_model = gl.glGetUniformLocation(self._grid_shader.id, str_buffer("model"))
            self._loc_grid_projection = gl.glGetUniformLocation(self._grid_shader.id, str_buffer("projection"))

            self._loc_grid_pos_attribute = gl.glGetAttribLocation(self._grid_shader.id, str_buffer("position"))
            gl.glVertexAttribPointer(self._loc_grid_pos_attribute, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(self._loc_grid_pos_attribute)

        # create sky data
        self._sky_shader = ShaderProgram(Shader(sky_vertex_shader, "vertex"), Shader(sky_fragment_shader, "fragment"))

        with self._sky_shader:
            self._loc_sky_view = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("view"))
            self._loc_sky_model = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("model"))
            self._loc_sky_projection = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("projection"))

            self._loc_sky_color1 = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("color1"))
            self._loc_sky_color2 = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("color2"))
            gl.glUniform3f(self._loc_sky_color1, *background_color)
            # glUniform3f(self._loc_sky_color2, *np.clip(np.array(background_color)+0.5, 0.0, 1.0))
            gl.glUniform3f(self._loc_sky_color2, 0.8, 0.4, 0.05)
            self._loc_sky_view_pos = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("viewPos"))
            gl.glUniform3f(
                gl.glGetUniformLocation(self._sky_shader.id, str_buffer("sunDirection")), *self._sun_direction
            )

        # Create VAO, VBO, and EBO
        self._sky_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._sky_vao)
        gl.glBindVertexArray(self._sky_vao)

        vertices, indices = self._create_sphere_mesh(self.camera_far_plane * 0.9, 32, 32, reverse_winding=True)
        self._sky_tri_count = len(indices)

        self._sky_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._sky_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._sky_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

        self._sky_ebo = gl.GLuint()
        gl.glGenBuffers(1, self._sky_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._sky_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        self._last_time = time.time()
        self._last_begin_frame_time = self._last_time
        self._last_end_frame_time = self._last_time

        # create arrow shapes for the coordinate system axes
        vertices, indices = self._create_arrow_mesh(
            base_radius=0.02 * axis_scale, base_height=0.85 * axis_scale, cap_height=0.15 * axis_scale
        )
        self._axis_instancer = ShapeInstancer(self._shape_shader, self._device)
        self._axis_instancer.register_shape(vertices, indices)
        sqh = np.sqrt(0.5)
        self._axis_instancer.allocate_instances(
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            rotations=[(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, -sqh, sqh), (sqh, 0.0, 0.0, sqh)],
            colors1=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            colors2=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        )

        # create frame buffer for rendering to a texture
        self._frame_texture = None
        self._frame_fbo = None
        self._setup_framebuffer()

        # fmt: off
        # set up VBO for the quad that is rendered to the user window with the texture
        self._frame_vertices = np.array([
            # Positions  TexCoords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ], dtype=np.float32)
        # fmt: on

        self._frame_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self._frame_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._frame_vao)
        gl.glBindVertexArray(self._frame_vao)

        self._frame_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._frame_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, self._frame_vertices.nbytes, self._frame_vertices.ctypes.data, gl.GL_STATIC_DRAW
        )

        self._frame_ebo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._frame_ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, self._frame_indices.nbytes, self._frame_indices.ctypes.data, gl.GL_STATIC_DRAW
        )

        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * self._frame_vertices.itemsize, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * self._frame_vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize)
        )
        gl.glEnableVertexAttribArray(1)

        self._frame_shader = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"), Shader(frame_fragment_shader, "fragment")
        )
        gl.glUseProgram(self._frame_shader.id)
        self._frame_loc_texture = gl.glGetUniformLocation(self._frame_shader.id, str_buffer("textureSampler"))

        # Unbind the VBO and VAO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # update model matrix
        self.scaling = scaling

        check_gl_error()

        # create text to render stats on the screen
        self._info_label = pyglet.text.Label(
            "",
            font_name="Arial",
            font_size=12,
            color=(255, 255, 255, 255),
            x=10,
            y=10,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            width=400,
        )

        # set up our own event handling so we can synchronously render frames
        # by calling update() in a loop
        from pyglet.window import Window

        Window._enable_event_queue = False

        self.window.switch_to()
        self.window.dispatch_pending_events()

        platform_event_loop = self.app.platform_event_loop
        platform_event_loop.start()

        # start event loop
        self.app.event_loop.dispatch_event("on_enter")

    @property
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, value):
        self._paused = value
        if value:
            self.window.set_caption(f"{self._title} (paused)")
        else:
            self.window.set_caption(self._title)

    @property
    def has_exit(self):
        return self.app.event_loop.has_exit

    def clear(self):
        from pyglet import gl

        self.app.event_loop.dispatch_event("on_exit")
        self.app.platform_event_loop.stop()

        if self._instance_transform_gl_buffer is not None:
            try:
                gl.glDeleteBuffers(1, self._instance_transform_gl_buffer)
                gl.glDeleteBuffers(1, self._instance_color1_buffer)
                gl.glDeleteBuffers(1, self._instance_color2_buffer)
            except gl.GLException:
                pass
        for vao, vbo, ebo, _, vertex_cuda_buffer in self._shape_gl_buffers.values():
            try:
                gl.glDeleteVertexArrays(1, vao)
                gl.glDeleteBuffers(1, vbo)
                gl.glDeleteBuffers(1, ebo)
            except gl.GLException:
                pass

        self._body_name.clear()
        self._shapes.clear()
        self._shape_geo_hash.clear()
        self._shape_gl_buffers.clear()
        self._shape_instances.clear()
        self._instances.clear()
        self._instance_shape.clear()
        self._instance_gl_buffers.clear()
        self._instance_transform_gl_buffer = None
        self._instance_transform_cuda_buffer = None
        self._instance_color1_buffer = None
        self._instance_color2_buffer = None
        self._wp_instance_ids = None
        self._wp_instance_transforms = None
        self._wp_instance_scalings = None
        self._wp_instance_bodies = None
        self._update_shape_instances = False

    @property
    def tiled_rendering(self):
        return self._tiled_rendering

    @tiled_rendering.setter
    def tiled_rendering(self, value):
        if value:
            assert self._tile_instances is not None, "Tiled rendering is not set up. Call setup_tiled_rendering first."
        self._tiled_rendering = value

    def setup_tiled_rendering(
        self,
        instances: List[List[int]],
        rescale_window: bool = False,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        tile_ncols: Optional[int] = None,
        tile_nrows: Optional[int] = None,
        tile_positions: Optional[List[Tuple[int]]] = None,
        tile_sizes: Optional[List[Tuple[int]]] = None,
        projection_matrices: Optional[List[Mat44]] = None,
        view_matrices: Optional[List[Mat44]] = None,
    ):
        """
        Set up tiled rendering where the render buffer is split into multiple tiles that can visualize
        different shape instances of the scene with different view and projection matrices.
        See `get_pixels` which allows to retrieve the pixels of for each tile.

        :param instances: A list of lists of shape instance ids. Each list of shape instance ids
            will be rendered into a separate tile.
        :param rescale_window: If True, the window will be resized to fit the tiles.
        :param tile_width: The width of each tile in pixels (optional).
        :param tile_height: The height of each tile in pixels (optional).
        :param tile_ncols: The number of tiles rendered horizontally (optional). Will be considered
            if `tile_width` is set to compute the tile positions, unless `tile_positions` is defined.
        :param tile_positions: A list of (x, y) tuples specifying the position of each tile in pixels.
            If None, the tiles will be arranged in a square grid, or, if `tile_ncols` and `tile_nrows`
            is set, in a grid with the specified number of columns and rows.
        :param tile_sizes: A list of (width, height) tuples specifying the size of each tile in pixels.
            If None, the tiles will have the same size as specified by `tile_width` and `tile_height`.
        :param projection_matrices: A list of projection matrices for each tile (each view matrix is
            either a flattened 16-dimensional array or a 4x4 matrix).
            If the entire array is None, or only a view instances, the projection matrices for all, or these
            instances, respectively, will be derived from the current render settings.
        :param view_matrices: A list of view matrices for each tile (each view matrix is either a flattened
            16-dimensional array or a 4x4 matrix).
            If the entire array is None, or only a view instances, the view matrices for all, or these
            instances, respectively, will be derived from the current camera settings and be
            updated when the camera is moved.
        """

        assert len(instances) > 0 and all(isinstance(i, list) for i in instances), "Invalid tile instances."

        self._tile_instances = instances
        n = len(self._tile_instances)

        if tile_positions is None or tile_sizes is None:
            if tile_ncols is None or tile_nrows is None:
                # try to fit the tiles into a square
                self._tile_ncols = int(np.ceil(np.sqrt(n)))
                self._tile_nrows = int(np.ceil(n / float(self._tile_ncols)))
            else:
                self._tile_ncols = tile_ncols
                self._tile_nrows = tile_nrows
            self._tile_width = tile_width or max(32, self.screen_width // self._tile_ncols)
            self._tile_height = tile_height or max(32, self.screen_height // self._tile_nrows)
            self._tile_viewports = [
                (i * self._tile_width, j * self._tile_height, self._tile_width, self._tile_height)
                for i in range(self._tile_ncols)
                for j in range(self._tile_nrows)
            ]
            if rescale_window:
                self.window.set_size(self._tile_width * self._tile_ncols, self._tile_height * self._tile_nrows)
        else:
            assert (
                len(tile_positions) == n and len(tile_sizes) == n
            ), "Number of tiles does not match number of instances."
            self._tile_ncols = None
            self._tile_nrows = None
            self._tile_width = None
            self._tile_height = None
            if all([tile_sizes[i][0] == tile_sizes[0][0] for i in range(n)]):
                # tiles all have the same width
                self._tile_width = tile_sizes[0][0]
            if all([tile_sizes[i][1] == tile_sizes[0][1] for i in range(n)]):
                # tiles all have the same height
                self._tile_height = tile_sizes[0][1]
            self._tile_viewports = [(x, y, w, h) for (x, y), (w, h) in zip(tile_positions, tile_sizes)]

        if projection_matrices is None:
            projection_matrices = [None] * n
        self._tile_projection_matrices = []
        for i, p in enumerate(projection_matrices):
            if p is None:
                w, h = self._tile_viewports[i][2:]
                self._tile_projection_matrices.append(
                    self.compute_projection_matrix(
                        self.camera_fov, w / h, self.camera_near_plane, self.camera_far_plane
                    )
                )
            else:
                self._tile_projection_matrices.append(np.array(p).flatten())

        if view_matrices is None:
            self._tile_view_matrices = [None] * n
        else:
            self._tile_view_matrices = [np.array(m).flatten() for m in view_matrices]

        self._tiled_rendering = True

    def update_tile(
        self,
        tile_id,
        instances: Optional[List[int]] = None,
        projection_matrix: Optional[Mat44] = None,
        view_matrix: Optional[Mat44] = None,
        tile_size: Optional[Tuple[int]] = None,
        tile_position: Optional[Tuple[int]] = None,
    ):
        """
        Update the shape instances, projection matrix, view matrix, tile size, or tile position
        for a given tile given its index.

        :param tile_id: The index of the tile to update.
        :param instances: A list of shape instance ids (optional).
        :param projection_matrix: A projection matrix (optional).
        :param view_matrix: A view matrix (optional).
        :param tile_size: A (width, height) tuple specifying the size of the tile in pixels (optional).
        :param tile_position: A (x, y) tuple specifying the position of the tile in pixels (optional).
        """

        assert self._tile_instances is not None, "Tiled rendering is not set up. Call setup_tiled_rendering first."
        assert tile_id < len(self._tile_instances), "Invalid tile id."

        if instances is not None:
            self._tile_instances[tile_id] = instances
        if projection_matrix is not None:
            self._tile_projection_matrices[tile_id] = np.array(projection_matrix).flatten()
        if view_matrix is not None:
            self._tile_view_matrices[tile_id] = np.array(view_matrix).flatten()
        (x, y, w, h) = self._tile_viewports[tile_id]
        if tile_size is not None:
            w, h = tile_size
        if tile_position is not None:
            x, y = tile_position
        self._tile_viewports[tile_id] = (x, y, w, h)

    def _setup_framebuffer(self):
        from pyglet import gl

        if self._frame_texture is None:
            self._frame_texture = gl.GLuint()
            gl.glGenTextures(1, self._frame_texture)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            self.screen_width,
            self.screen_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # create a framebuffer object (FBO)
        if self._frame_fbo is None:
            self._frame_fbo = gl.GLuint()
            gl.glGenFramebuffers(1, self._frame_fbo)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)

            # attach the texture to the FBO as its color attachment
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._frame_texture, 0
            )

            self._frame_depth_renderbuffer = gl.GLuint()
            gl.glGenRenderbuffers(1, self._frame_depth_renderbuffer)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._frame_depth_renderbuffer)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, self.screen_width, self.screen_height)

            # attach the depth renderbuffer to the FBO
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self._frame_depth_renderbuffer
            )

            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                print("Framebuffer is not complete!")
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                sys.exit(1)

            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
        else:
            # rescale framebuffer
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._frame_depth_renderbuffer)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, self.screen_width, self.screen_height)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        # unbind the FBO (switch back to the default framebuffer)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        if self._frame_pbo is None:
            self._frame_pbo = gl.GLuint()
            gl.glGenBuffers(1, self._frame_pbo)  # generate 1 buffer reference
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._frame_pbo)  # binding to this buffer

        # allocate memory for PBO
        pixels = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, pixels.nbytes, pixels.ctypes.data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

    @staticmethod
    def compute_projection_matrix(
        fov: float,
        aspect_ratio: float,
        near_plane: float,
        far_plane: float,
    ) -> Mat44:
        """
        Compute a projection matrix given the field of view, aspect ratio, near plane, and far plane.

        :param fov: The field of view in degrees.
        :param aspect_ratio: The aspect ratio (width / height).
        :param near_plane: The near plane.
        :param far_plane: The far plane.
        :return: A projection matrix.
        """

        from pyglet.math import Mat4 as PyMat4

        return np.array(PyMat4.perspective_projection(aspect_ratio, near_plane, far_plane, fov))

    def update_projection_matrix(self):
        if self.screen_height == 0:
            return
        aspect_ratio = self.screen_width / self.screen_height
        self._projection_matrix = self.compute_projection_matrix(
            self.camera_fov, aspect_ratio, self.camera_near_plane, self.camera_far_plane
        )

    def update_view_matrix(self):
        from pyglet.math import Mat4 as PyMat4

        cam_pos = self._camera_pos
        self._view_matrix = np.array(PyMat4.look_at(cam_pos, cam_pos + self._camera_front, self._camera_up))

    def update_model_matrix(self):
        from pyglet import gl

        # fmt: off
        if self._camera_axis == 0:
            self._model_matrix = np.array((
                0, 0, self._scaling, 0,
                self._scaling, 0, 0, 0,
                0, self._scaling, 0, 0,
                0, 0, 0, 1
            ))
        elif self._camera_axis == 2:
            self._model_matrix = np.array((
                -self._scaling, 0, 0, 0,
                0, 0, self._scaling, 0,
                0, self._scaling, 0, 0,
                0, 0, 0, 1
            ))
        else:
            self._model_matrix = np.array((
                self._scaling, 0, 0, 0,
                0, self._scaling, 0, 0,
                0, 0, self._scaling, 0,
                0, 0, 0, 1
            ))
        # fmt: on
        ptr = arr_pointer(self._model_matrix)
        gl.glUseProgram(self._shape_shader.id)
        gl.glUniformMatrix4fv(self._loc_shape_model, 1, gl.GL_FALSE, ptr)
        gl.glUseProgram(self._grid_shader.id)
        gl.glUniformMatrix4fv(self._loc_grid_model, 1, gl.GL_FALSE, ptr)
        gl.glUseProgram(self._sky_shader.id)
        gl.glUniformMatrix4fv(self._loc_sky_model, 1, gl.GL_FALSE, ptr)

    @property
    def num_tiles(self):
        return len(self._tile_instances)

    @property
    def tile_width(self):
        return self._tile_width

    @property
    def tile_height(self):
        return self._tile_height

    @property
    def num_shapes(self):
        return len(self._shapes)

    @property
    def num_instances(self):
        return self._instance_count

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = scaling
        self.update_model_matrix()

    def begin_frame(self, t: float = None):
        self._last_begin_frame_time = time.time()
        self.time = t or self.clock_time

    def end_frame(self):
        self._last_end_frame_time = time.time()
        if self._add_shape_instances:
            self.allocate_shape_instances()
        if self._update_shape_instances:
            self.update_shape_instances()
        self.update()
        while self.paused and self.is_running():
            self.update()

    def update(self):
        self.clock_time = time.time() - self._start_time
        update_duration = self.clock_time - self._last_time
        frame_duration = self._last_end_frame_time - self._last_begin_frame_time
        self._last_time = self.clock_time
        self._frame_speed = update_duration * 100.0

        # self.app.event_loop.idle()
        self.app.platform_event_loop.step(self._frame_dt * 1e-3)

        if not self.skip_rendering:
            self._skip_frame_counter += 1
            if self._skip_frame_counter > 100:
                self._skip_frame_counter = 0

            if frame_duration > 0.0:
                if self._fps_update is None:
                    self._fps_update = 1.0 / frame_duration
                else:
                    update = 1.0 / frame_duration
                    self._fps_update = (1.0 - self._fps_alpha) * self._fps_update + self._fps_alpha * update
            if update_duration > 0.0:
                if self._fps_render is None:
                    self._fps_render = 1.0 / update_duration
                else:
                    update = 1.0 / update_duration
                    self._fps_render = (1.0 - self._fps_alpha) * self._fps_render + self._fps_alpha * update

            self.app.event_loop._redraw_windows(self._frame_dt * 1e-3)

    def _draw(self):
        from pyglet import gl

        # catch key hold events
        self._process_inputs()

        if self.enable_backface_culling:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        if self._frame_fbo is not None:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self._frame_fbo)

        gl.glClearColor(*self.background_color, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(0)

        if not self._tiled_rendering:
            if self.draw_grid:
                self._draw_grid()

            if self.draw_sky:
                self._draw_sky()

        view_mat_ptr = arr_pointer(self._view_matrix)
        projection_mat_ptr = arr_pointer(self._projection_matrix)
        gl.glUseProgram(self._shape_shader.id)
        gl.glUniformMatrix4fv(self._loc_shape_view, 1, gl.GL_FALSE, view_mat_ptr)
        gl.glUniform3f(self._loc_shape_view_pos, *self._camera_pos)
        gl.glUniformMatrix4fv(self._loc_shape_view, 1, gl.GL_FALSE, view_mat_ptr)
        gl.glUniformMatrix4fv(self._loc_shape_projection, 1, gl.GL_FALSE, projection_mat_ptr)

        if self.render_wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        if self._tiled_rendering:
            self._render_scene_tiled()
        else:
            self._render_scene()

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self.screen_width, self.screen_height)

        # render frame buffer texture to screen
        if self._frame_fbo is not None:
            with self._frame_shader:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
                gl.glUniform1i(self._frame_loc_texture, 0)

                gl.glBindVertexArray(self._frame_vao)
                gl.glDrawElements(gl.GL_TRIANGLES, len(self._frame_indices), gl.GL_UNSIGNED_INT, None)
                gl.glBindVertexArray(0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # check for OpenGL errors
        # check_gl_error()

        if self.show_info:
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)

            text = f"""Sim Time: {self.time:.1f}
Update FPS: {self._fps_update:.1f}
Render FPS: {self._fps_render:.1f}

Shapes: {len(self._shapes)}
Instances: {len(self._instances)}"""
            if self.paused:
                text += "\nPaused (press space to resume)"

            self._info_label.text = text
            self._info_label.y = self.screen_height - 5
            self._info_label.draw()

    def _draw_grid(self, is_tiled=False):
        from pyglet import gl

        if not is_tiled:
            gl.glUseProgram(self._grid_shader.id)

            gl.glUniformMatrix4fv(self._loc_grid_view, 1, gl.GL_FALSE, arr_pointer(self._view_matrix))
            gl.glUniformMatrix4fv(self._loc_grid_projection, 1, gl.GL_FALSE, arr_pointer(self._projection_matrix))

        gl.glBindVertexArray(self._grid_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self._grid_vertex_count)
        gl.glBindVertexArray(0)

    def _draw_sky(self, is_tiled=False):
        from pyglet import gl

        if not is_tiled:
            gl.glUseProgram(self._sky_shader.id)

            gl.glUniformMatrix4fv(self._loc_sky_view, 1, gl.GL_FALSE, arr_pointer(self._view_matrix))
            gl.glUniformMatrix4fv(self._loc_sky_projection, 1, gl.GL_FALSE, arr_pointer(self._projection_matrix))
            gl.glUniform3f(self._loc_sky_view_pos, *self._camera_pos)

        gl.glBindVertexArray(self._sky_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._sky_tri_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def _render_scene(self):
        from pyglet import gl

        start_instance_idx = 0

        for shape, (vao, _, _, tri_count, _) in self._shape_gl_buffers.items():
            num_instances = len(self._shape_instances[shape])

            gl.glBindVertexArray(vao)
            gl.glDrawElementsInstancedBaseInstance(
                gl.GL_TRIANGLES, tri_count, gl.GL_UNSIGNED_INT, None, num_instances, start_instance_idx
            )

            start_instance_idx += num_instances

        if self.draw_axis:
            self._axis_instancer.render()

        for instancer in self._shape_instancers.values():
            instancer.render()

        gl.glBindVertexArray(0)

    def _render_scene_tiled(self):
        from pyglet import gl

        for i, viewport in enumerate(self._tile_viewports):
            projection_matrix_ptr = arr_pointer(self._tile_projection_matrices[i])
            view_matrix_ptr = arr_pointer(self._tile_view_matrices[i] or self._view_matrix)

            gl.glViewport(*viewport)
            if self.draw_grid:
                gl.glUseProgram(self._grid_shader.id)
                gl.glUniformMatrix4fv(self._loc_grid_projection, 1, gl.GL_FALSE, projection_matrix_ptr)
                gl.glUniformMatrix4fv(self._loc_grid_view, 1, gl.GL_FALSE, view_matrix_ptr)
                self._draw_grid(is_tiled=True)

            if self.draw_sky:
                gl.glUseProgram(self._sky_shader.id)
                gl.glUniformMatrix4fv(self._loc_sky_projection, 1, gl.GL_FALSE, projection_matrix_ptr)
                gl.glUniformMatrix4fv(self._loc_sky_view, 1, gl.GL_FALSE, view_matrix_ptr)
                self._draw_sky(is_tiled=True)

            gl.glUseProgram(self._shape_shader.id)
            gl.glUniformMatrix4fv(self._loc_shape_projection, 1, gl.GL_FALSE, projection_matrix_ptr)
            gl.glUniformMatrix4fv(self._loc_shape_view, 1, gl.GL_FALSE, view_matrix_ptr)

            instances = self._tile_instances[i]

            for instance in instances:
                shape = self._instance_shape[instance]

                vao, _, _, tri_count, _ = self._shape_gl_buffers[shape]

                start_instance_idx = self._inverse_instance_ids[instance]

                gl.glBindVertexArray(vao)
                gl.glDrawElementsInstancedBaseInstance(
                    gl.GL_TRIANGLES, tri_count, gl.GL_UNSIGNED_INT, None, 1, start_instance_idx
                )

            if self.draw_axis:
                self._axis_instancer.render()

            for instancer in self._shape_instancers.values():
                instancer.render()

        gl.glBindVertexArray(0)

    def _mouse_drag_callback(self, x, y, dx, dy, buttons, modifiers):
        import pyglet

        if buttons & pyglet.window.mouse.LEFT:
            sensitivity = 0.1
            dx *= sensitivity
            dy *= sensitivity

            self._yaw += dx
            self._pitch += dy

            self._pitch = max(min(self._pitch, 89.0), -89.0)

            self._camera_front.x = np.cos(np.deg2rad(self._yaw)) * np.cos(np.deg2rad(self._pitch))
            self._camera_front.y = np.sin(np.deg2rad(self._pitch))
            self._camera_front.z = np.sin(np.deg2rad(self._yaw)) * np.cos(np.deg2rad(self._pitch))
            self._camera_front = self._camera_front.normalize()
            self.update_view_matrix()

    def _scroll_callback(self, x, y, scroll_x, scroll_y):
        self.camera_fov -= scroll_y
        self.camera_fov = max(min(self.camera_fov, 90.0), 15.0)
        self.update_projection_matrix()

    def _process_inputs(self):
        import pyglet
        from pyglet.math import Vec3 as PyVec3

        if self._key_handler[pyglet.window.key.W] or self._key_handler[pyglet.window.key.UP]:
            self._camera_pos += self._camera_front * (self._camera_speed * self._frame_speed)
            self.update_view_matrix()
        if self._key_handler[pyglet.window.key.S] or self._key_handler[pyglet.window.key.DOWN]:
            self._camera_pos -= self._camera_front * (self._camera_speed * self._frame_speed)
            self.update_view_matrix()
        if self._key_handler[pyglet.window.key.A] or self._key_handler[pyglet.window.key.LEFT]:
            camera_side = PyVec3.cross(self._camera_front, self._camera_up).normalize()
            self._camera_pos -= camera_side * (self._camera_speed * self._frame_speed)
            self.update_view_matrix()
        if self._key_handler[pyglet.window.key.D] or self._key_handler[pyglet.window.key.RIGHT]:
            camera_side = PyVec3.cross(self._camera_front, self._camera_up).normalize()
            self._camera_pos += camera_side * (self._camera_speed * self._frame_speed)
            self.update_view_matrix()

    def _key_press_callback(self, symbol, modifiers):
        import pyglet

        if symbol == pyglet.window.key.ESCAPE:
            self.window.close()
        if symbol == pyglet.window.key.SPACE:
            self.paused = not self.paused
        if symbol == pyglet.window.key.TAB:
            self.skip_rendering = not self.skip_rendering
        if symbol == pyglet.window.key.C:
            self.draw_axis = not self.draw_axis
        if symbol == pyglet.window.key.G:
            self.draw_grid = not self.draw_grid
        if symbol == pyglet.window.key.I:
            self.show_info = not self.show_info
        if symbol == pyglet.window.key.X:
            self.render_wireframe = not self.render_wireframe
        if symbol == pyglet.window.key.B:
            self.enable_backface_culling = not self.enable_backface_culling

    def _window_resize_callback(self, width, height):
        self._first_mouse = True
        self.screen_width, self.screen_height = self.window.get_framebuffer_size()
        self.update_projection_matrix()
        self._setup_framebuffer()

    def register_shape(self, geo_hash, vertices, indices, color1=None, color2=None):
        from pyglet import gl

        shape = len(self._shapes)
        if color1 is None:
            color1 = tab10_color_map(len(self._shape_geo_hash))
        if color2 is None:
            color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
        # TODO check if we actually need to store the shape data
        self._shapes.append((vertices, indices, color1, color2, geo_hash))
        self._shape_geo_hash[geo_hash] = shape

        gl.glUseProgram(self._shape_shader.id)

        # Create VAO, VBO, and EBO
        vao = gl.GLuint()
        gl.glGenVertexArrays(1, vao)
        gl.glBindVertexArray(vao)

        vbo = gl.GLuint()
        gl.glGenBuffers(1, vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

        vertex_cuda_buffer = wp.RegisteredGLBuffer(int(vbo.value), self._device)

        ebo = gl.GLuint()
        gl.glGenBuffers(1, ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        self._shape_gl_buffers[shape] = (vao, vbo, ebo, len(indices), vertex_cuda_buffer)

        return shape

    def add_shape_instance(
        self, name: str, shape: int, body, pos, rot, scale=(1.0, 1.0, 1.0), color1=None, color2=None
    ):
        if color1 is None:
            color1 = self._shapes[shape][2]
        if color2 is None:
            color2 = self._shapes[shape][3]
        instance = len(self._instances)
        self._shape_instances[shape].append(instance)
        body = self._resolve_body_id(body)
        self._instances[name] = (instance, body, shape, [*pos, *rot], scale, color1, color2)
        self._instance_shape[instance] = shape
        self._add_shape_instances = True
        self._instance_count = len(self._instances)
        return instance

    def allocate_shape_instances(self):
        from pyglet import gl

        self._add_shape_instances = False
        self._wp_instance_transforms = wp.array(
            [instance[3] for instance in self._instances.values()], dtype=wp.transform, device=self._device
        )
        self._wp_instance_scalings = wp.array(
            [instance[4] for instance in self._instances.values()], dtype=wp.vec3, device=self._device
        )
        self._wp_instance_bodies = wp.array(
            [instance[1] for instance in self._instances.values()], dtype=wp.int32, device=self._device
        )

        gl.glUseProgram(self._shape_shader.id)
        if self._instance_transform_gl_buffer is not None:
            gl.glDeleteBuffers(1, self._instance_transform_gl_buffer)
            gl.glDeleteBuffers(1, self._instance_color1_buffer)
            gl.glDeleteBuffers(1, self._instance_color2_buffer)

        # Create instance buffer and bind it as an instanced array
        self._instance_transform_gl_buffer = gl.GLuint()
        gl.glGenBuffers(1, self._instance_transform_gl_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

        transforms = np.tile(np.diag(np.ones(4, dtype=np.float32)), (len(self._instances), 1, 1))
        gl.glBufferData(gl.GL_ARRAY_BUFFER, transforms.nbytes, transforms.ctypes.data, gl.GL_DYNAMIC_DRAW)

        # Create CUDA buffer for instance transforms
        self._instance_transform_cuda_buffer = wp.RegisteredGLBuffer(
            int(self._instance_transform_gl_buffer.value), self._device
        )

        colors1, colors2 = [], []
        all_instances = list(self._instances.values())
        for shape, instances in self._shape_instances.items():
            for i in instances:
                if i >= len(all_instances):
                    continue
                instance = all_instances[i]
                colors1.append(instance[5])
                colors2.append(instance[6])
        colors1 = np.array(colors1, dtype=np.float32)
        colors2 = np.array(colors2, dtype=np.float32)

        # create buffer for checkerboard colors
        self._instance_color1_buffer = gl.GLuint()
        gl.glGenBuffers(1, self._instance_color1_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color1_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors1.nbytes, colors1.ctypes.data, gl.GL_STATIC_DRAW)

        self._instance_color2_buffer = gl.GLuint()
        gl.glGenBuffers(1, self._instance_color2_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color2_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors2.nbytes, colors2.ctypes.data, gl.GL_STATIC_DRAW)

        # Set up instance attribute pointers
        matrix_size = transforms[0].nbytes

        instance_ids = []
        inverse_instance_ids = {}
        instance_count = 0
        for shape, (vao, vbo, ebo, tri_count, vertex_cuda_buffer) in self._shape_gl_buffers.items():
            gl.glBindVertexArray(vao)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

            # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
            for i in range(4):
                gl.glVertexAttribPointer(
                    3 + i, 4, gl.GL_FLOAT, gl.GL_FALSE, matrix_size, ctypes.c_void_p(i * matrix_size // 4)
                )
                gl.glEnableVertexAttribArray(3 + i)
                gl.glVertexAttribDivisor(3 + i, 1)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color1_buffer)
            gl.glVertexAttribPointer(7, 3, gl.GL_FLOAT, gl.GL_FALSE, colors1[0].nbytes, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(7)
            gl.glVertexAttribDivisor(7, 1)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color2_buffer)
            gl.glVertexAttribPointer(8, 3, gl.GL_FLOAT, gl.GL_FALSE, colors2[0].nbytes, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(8)
            gl.glVertexAttribDivisor(8, 1)

            instance_ids.extend(self._shape_instances[shape])
            for i in self._shape_instances[shape]:
                inverse_instance_ids[i] = instance_count
                instance_count += 1

        # trigger update to the instance transforms
        self._update_shape_instances = True

        self._wp_instance_ids = wp.array(instance_ids, dtype=wp.int32, device=self._device)
        self._instance_ids = instance_ids
        self._inverse_instance_ids = inverse_instance_ids

        gl.glBindVertexArray(0)

    def update_shape_instance(self, name, pos, rot, color1=None, color2=None):
        """Update the instance transform of the shape

        Args:
            name: The name of the shape
            pos: The position of the shape
            rot: The rotation of the shape
        """
        if name in self._instances:
            i, body, shape, _, scale, old_color1, old_color2 = self._instances[name]
            self._instances[name] = (i, body, shape, [*pos, *rot], scale, color1 or old_color1, color2 or old_color2)
            self._update_shape_instances = True
            return True
        return False

    def update_shape_instances(self):
        with self._shape_shader:
            self._update_shape_instances = False
            self._wp_instance_transforms = wp.array(
                [instance[3] for instance in self._instances.values()], dtype=wp.transform, device=self._device
            )
            self.update_body_transforms(None)

    def update_body_transforms(self, body_tf: wp.array):
        if self._instance_transform_cuda_buffer is None:
            return

        body_q = None
        if body_tf is not None:
            if body_tf.device.is_cuda:
                body_q = body_tf
            else:
                body_q = body_tf.to(self._device)

        vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self._instance_count,))

        wp.launch(
            update_vbo_transforms,
            dim=self._instance_count,
            inputs=[
                self._wp_instance_ids,
                self._wp_instance_bodies,
                self._wp_instance_transforms,
                self._wp_instance_scalings,
                body_q,
            ],
            outputs=[
                vbo_transforms,
            ],
            device=self._device,
        )

        self._instance_transform_cuda_buffer.unmap()

    def register_body(self, name):
        # register body name and return its ID
        if name not in self._body_name:
            self._body_name[name] = len(self._body_name)
        return self._body_name[name]

    def _resolve_body_id(self, body):
        if body is None:
            return -1
        if isinstance(body, int):
            return body
        return self._body_name[body]

    def is_running(self):
        return not self.app.event_loop.has_exit

    def save(self):
        # save just keeps the window open to allow the user to interact with the scene
        while not self.app.event_loop.has_exit:
            self.update()
        if self.app.event_loop.has_exit:
            self.clear()
            self.app.event_loop.exit()

    def get_pixels(self, target_image: wp.array, split_up_tiles=True):
        from pyglet import gl

        if split_up_tiles:
            assert (
                self._tile_width is not None and self._tile_height is not None
            ), "Tile width and height are not set, tiles must all have the same size"
            assert all(
                vp[2] == self._tile_width for vp in self._tile_viewports
            ), "Tile widths do not all equal global tile_width, use `get_tile_pixels` instead to retrieve pixels for a single tile"
            assert all(
                vp[3] == self._tile_height for vp in self._tile_viewports
            ), "Tile heights do not all equal global tile_height, use `get_tile_pixels` instead to retrieve pixels for a single tile"
            assert target_image.shape == (
                self.num_tiles,
                self._tile_height,
                self._tile_width,
                3,
            ), f"Shape of `target_image` array does not match {self.num_tiles} x {self.screen_height} x {self.screen_width} x 3"
        else:
            assert target_image.shape == (
                self.screen_height,
                self.screen_width,
                3,
            ), f"Shape of `target_image` array does not match {self.screen_height} x {self.screen_width} x 3"

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._frame_pbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
        try:
            # read screen texture into PBO
            gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        except gl.GLException:
            # this can happen if the window is closed/being moved to a different display
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
            return False
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

        pbo_buffer = wp.RegisteredGLBuffer(
            int(self._frame_pbo.value), self._device, wp.RegisteredGLBuffer.WRITE_DISCARD
        )
        screen_size = self.screen_height * self.screen_width
        img = pbo_buffer.map(dtype=wp.uint8, shape=(screen_size * 3))
        img = img.to(target_image.device)
        if split_up_tiles:
            positions = wp.array(self._tile_viewports, ndim=2, dtype=wp.int32, device=target_image.device)
            wp.launch(
                copy_frame_tiles,
                dim=(self.num_tiles, self._tile_width, self._tile_height),
                inputs=[img, positions, self.screen_width, self.screen_height, self._tile_height],
                outputs=[target_image],
                device=target_image.device,
            )
        else:
            wp.launch(
                copy_frame,
                dim=(self.screen_width, self.screen_height),
                inputs=[img, self.screen_width, self.screen_height],
                outputs=[target_image],
                device=target_image.device,
            )
        pbo_buffer.unmap()
        return True

    def get_tile_pixels(self, tile_id: int, target_image: wp.array):
        from pyglet import gl

        viewport = self._tile_viewports[tile_id]
        assert target_image.shape == (
            viewport[3],
            viewport[2],
            3,
        ), f"Shape of `target_image` array does not match {viewport[3]} x {viewport[2]} x 3"
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._frame_pbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
        try:
            # read screen texture into PBO
            gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        except gl.GLException:
            # this can happen if the window is closed/being moved to a different display
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
            return False
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

        pbo_buffer = wp.RegisteredGLBuffer(
            int(self._frame_pbo.value), self._device, wp.RegisteredGLBuffer.WRITE_DISCARD
        )
        screen_size = self.screen_height * self.screen_width
        img = pbo_buffer.map(dtype=wp.uint8, shape=(screen_size * 3))
        img = img.to(target_image.device)
        wp.launch(
            copy_frame_tiles,
            dim=(self.num_tiles, self._tile_width, self._tile_height),
            inputs=[img, viewport[0], viewport[1], self.screen_width, self.screen_height, self._tile_height],
            outputs=[target_image],
            device=target_image.device,
        )
        pbo_buffer.unmap()
        return True

    # def create_image_texture(self, file_path):
    #     from PIL import Image
    #     img = Image.open(file_path)
    #     img_data = np.array(list(img.getdata()), np.uint8)
    #     texture = glGenTextures(1)
    #     glBindTexture(GL_TEXTURE_2D, texture)
    #     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    #     return texture

    # def create_check_texture(self, color1=(0, 0.5, 1.0), color2=None, width=default_texture_size, height=default_texture_size):
    #     if width == 1 and height == 1:
    #         pixels = np.array([np.array(color1)*255], dtype=np.uint8)
    #     else:
    #         pixels = np.zeros((width, height, 3), dtype=np.uint8)
    #         half_w = width // 2
    #         half_h = height // 2
    #         color1 = np.array(np.array(color1)*255, dtype=np.uint8)
    #         pixels[0:half_w, 0:half_h] = color1
    #         pixels[half_w:width, half_h:height] = color1
    #         if color2 is None:
    #             color2 = np.array(np.clip(np.array(color1, dtype=np.float32) + 50, 0, 255), dtype=np.uint8)
    #         else:
    #             color2 = np.array(np.array(color2)*255, dtype=np.uint8)
    #         pixels[half_w:width, 0:half_h] = color2
    #         pixels[0:half_w, half_h:height] = color2
    #     texture = glGenTextures(1)
    #     glBindTexture(GL_TEXTURE_2D, texture)
    #     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.flatten())
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    #     return texture

    def render_plane(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        width: float,
        length: float,
        color: tuple = (1.0, 1.0, 1.0),
        color2=None,
        parent_body: str = None,
        is_template: bool = False,
        u_scaling=1.0,
        v_scaling=1.0,
    ):
        """Add a plane for visualization

        Args:
            name: The name of the plane
            pos: The position of the plane
            rot: The rotation of the plane
            width: The width of the plane
            length: The length of the plane
            color: The color of the plane
            texture: The texture of the plane (optional)
        """
        geo_hash = hash(("plane", width, length))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            faces = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            normal = (0.0, 1.0, 0.0)
            width = width if width > 0.0 else 100.0
            length = length if length > 0.0 else 100.0
            aspect = width / length
            u = width * aspect * u_scaling
            v = length * v_scaling
            gfx_vertices = np.array(
                [
                    [-width, 0.0, -length, *normal, 0.0, 0.0],
                    [-width, 0.0, length, *normal, 0.0, v],
                    [width, 0.0, length, *normal, u, v],
                    [width, 0.0, -length, *normal, u, 0.0],
                ],
                dtype=np.float32,
            )
            shape = self.register_shape(geo_hash, gfx_vertices, faces, color1=color, color2=color2)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_ground(self, size: float = 100.0):
        """Add a ground plane for visualization

        Args:
            size: The size of the ground plane
        """
        color1 = (200 / 255, 200 / 255, 200 / 255)
        color2 = (150 / 255, 150 / 255, 150 / 255)
        sqh = np.sqrt(0.5)
        if self._camera_axis == 0:
            q = (0.0, 0.0, -sqh, sqh)
        elif self._camera_axis == 1:
            q = (0.0, 0.0, 0.0, 1.0)
        elif self._camera_axis == 2:
            q = (sqh, 0.0, 0.0, sqh)
        return self.render_plane(
            "ground",
            (0.0, 0.0, 0.0),
            q,
            size,
            size,
            color1,
            color2=color2,
            u_scaling=1.0,
            v_scaling=1.0,
        )

    def render_sphere(
        self, name: str, pos: tuple, rot: tuple, radius: float, parent_body: str = None, is_template: bool = False
    ):
        """Add a sphere for visualization

        Args:
            pos: The position of the sphere
            radius: The radius of the sphere
            name: A name for the USD prim on the stage
        """
        geo_hash = hash(("sphere", radius))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_sphere_mesh(radius)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_capsule(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        half_height: float,
        parent_body: str = None,
        is_template: bool = False,
        up_axis: int = 1,
    ):
        """Add a capsule for visualization

        Args:
            pos: The position of the capsule
            radius: The radius of the capsule
            half_height: The half height of the capsule
            name: A name for the USD prim on the stage
            up_axis: The axis of the capsule that points up (0: x, 1: y, 2: z)
        """
        geo_hash = hash(("capsule", radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_capsule_mesh(radius, half_height, up_axis=up_axis)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_cylinder(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        half_height: float,
        parent_body: str = None,
        is_template: bool = False,
        up_axis: int = 1,
    ):
        """Add a cylinder for visualization

        Args:
            pos: The position of the cylinder
            radius: The radius of the cylinder
            half_height: The half height of the cylinder
            name: A name for the USD prim on the stage
            up_axis: The axis of the cylinder that points up (0: x, 1: y, 2: z)
        """
        geo_hash = hash(("cylinder", radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_cylinder_mesh(radius, half_height, up_axis=up_axis)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_cone(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        half_height: float,
        parent_body: str = None,
        is_template: bool = False,
        up_axis: int = 1,
    ):
        """Add a cone for visualization

        Args:
            pos: The position of the cone
            radius: The radius of the cone
            half_height: The half height of the cone
            name: A name for the USD prim on the stage
            up_axis: The axis of the cone that points up (0: x, 1: y, 2: z)
        """
        geo_hash = hash(("cone", radius, half_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_cone_mesh(radius, half_height, up_axis=up_axis)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_box(
        self, name: str, pos: tuple, rot: tuple, extents: tuple, parent_body: str = None, is_template: bool = False
    ):
        """Add a box for visualization

        Args:
            pos: The position of the box
            extents: The extents of the box
            name: A name for the USD prim on the stage
        """
        geo_hash = hash(("box", tuple(extents)))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_box_mesh(extents)
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_mesh(
        self,
        name: str,
        points,
        indices,
        colors=None,
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
        update_topology=False,
        parent_body: str = None,
        is_template: bool = False,
        smooth_shading: bool = True,
    ):
        """Add a mesh for visualization

        Args:
            points: The points of the mesh
            indices: The indices of the mesh
            colors: The colors of the mesh
            pos: The position of the mesh
            rot: The rotation of the mesh
            scale: The scale of the mesh
            name: A name for the USD prim on the stage
            smooth_shading: Whether to average face normals at each vertex or introduce additional vertices for each face
        """
        if colors is None:
            colors = np.ones((len(points), 3), dtype=np.float32)
        else:
            colors = np.array(colors, dtype=np.float32)
        points = np.array(points, dtype=np.float32) * np.array(scale, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32).reshape((-1, 3))
        if name in self._instances:
            self.update_shape_instance(name, pos, rot)
            shape = self._instances[name][2]
            self.update_shape_vertices(shape, points)
            return
        geo_hash = hash((points.tobytes(), indices.tobytes(), colors.tobytes()))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            if smooth_shading:
                normals = wp.zeros(len(points), dtype=wp.vec3)
                vertices = wp.array(points, dtype=wp.vec3)
                faces_per_vertex = wp.zeros(len(points), dtype=int)
                wp.launch(
                    compute_average_normals,
                    dim=len(indices),
                    inputs=[wp.array(indices, dtype=int), vertices],
                    outputs=[normals, faces_per_vertex],
                )
                gfx_vertices = wp.zeros((len(points), 8), dtype=float)
                wp.launch(
                    assemble_gfx_vertices,
                    dim=len(points),
                    inputs=[vertices, normals, faces_per_vertex],
                    outputs=[gfx_vertices],
                )
                gfx_vertices = gfx_vertices.numpy()
                gfx_indices = indices.flatten()
            else:
                gfx_vertices = wp.zeros((len(indices) * 3, 8), dtype=float)
                wp.launch(
                    compute_gfx_vertices,
                    dim=len(indices),
                    inputs=[wp.array(indices, dtype=int), wp.array(points, dtype=wp.vec3)],
                    outputs=[gfx_vertices],
                )
                gfx_vertices = gfx_vertices.numpy()
                gfx_indices = np.arange(len(indices) * 3)
            shape = self.register_shape(geo_hash, gfx_vertices, gfx_indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_arrow(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        base_radius: float,
        base_height: float,
        cap_radius: float = None,
        cap_height: float = None,
        parent_body: str = None,
        is_template: bool = False,
        up_axis: int = 1,
        color: Tuple[float, float, float] = None,
    ):
        """Add a arrow for visualization

        Args:
            pos: The position of the arrow
            base_radius: The radius of the cylindrical base of the arrow
            base_height: The height of the cylindrical base of the arrow
            cap_radius: The radius of the conical cap of the arrow
            cap_height: The height of the conical cap of the arrow
            name: A name for the USD prim on the stage
            up_axis: The axis of the arrow that points up (0: x, 1: y, 2: z)
        """
        geo_hash = hash(("arrow", base_radius, base_height, cap_radius, cap_height))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            vertices, indices = self._create_arrow_mesh(
                base_radius, base_height, cap_radius, cap_height, up_axis=up_axis
            )
            shape = self.register_shape(geo_hash, vertices, indices)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot, color1=color, color2=color)
        return shape

    def render_ref(self, name: str, path: str, pos: tuple, rot: tuple, scale: tuple):
        """
        Create a reference (instance) with the given name to the given path.
        """

        if path in self._instances:
            _, body, shape, _, original_scale, color1, color2 = self._instances[path]
            self.add_shape_instance(name, shape, body, pos, rot, scale or original_scale, color1, color2)
            return

        raise Exception("Cannot create reference to path: " + path)

    def render_points(self, name: str, points, radius, colors=None):
        """Add a set of points

        Args:
            points: The points to render
            radius: The radius of the points (scalar or list)
            colors: The colors of the points
            name: A name for the USD prim on the stage
        """

        if len(points) == 0:
            return

        if isinstance(points, wp.array):
            wp_points = points
        else:
            wp_points = wp.array(points, dtype=wp.vec3, device=self._device)

        if name not in self._shape_instancers:
            np_points = points.numpy() if isinstance(points, wp.array) else points
            instancer = ShapeInstancer(self._shape_shader, self._device)
            radius_is_scalar = np.isscalar(radius)
            if radius_is_scalar:
                vertices, indices = self._create_sphere_mesh(radius)
            else:
                vertices, indices = self._create_sphere_mesh(1.0)
            if colors is None:
                color = tab10_color_map(len(self._shape_geo_hash))
            else:
                color = colors[0]
            instancer.register_shape(vertices, indices, color, color)
            scalings = None if radius_is_scalar else np.tile(radius, (3, 1)).T
            instancer.allocate_instances(np_points, colors1=colors, colors2=colors, scalings=scalings)
            self._shape_instancers[name] = instancer
        else:
            instancer = self._shape_instancers[name]
            if len(points) != instancer.num_instances:
                np_points = points.numpy() if isinstance(points, wp.array) else points
                instancer.allocate_instances(np_points)

        with instancer:
            wp.launch(
                update_points_positions,
                dim=len(points),
                inputs=[wp_points, instancer.instance_scalings],
                outputs=[instancer.vbo_transforms],
                device=self._device,
            )

    def _render_lines(self, name: str, lines, color: tuple, radius: float = 0.01):
        if len(lines) == 0:
            return

        if name not in self._shape_instancers:
            instancer = ShapeInstancer(self._shape_shader, self._device)
            vertices, indices = self._create_capsule_mesh(radius, 0.5)
            if color is None or isinstance(color, list):
                color = tab10_color_map(len(self._shape_geo_hash))
            instancer.register_shape(vertices, indices, color, color)
            instancer.allocate_instances(np.zeros((len(lines), 3)))
            self._shape_instancers[name] = instancer
        else:
            instancer = self._shape_instancers[name]
            if len(lines) != instancer.num_instances:
                instancer.allocate_instances(np.zeros((len(lines), 3)))

        lines_wp = wp.array(lines, dtype=wp.vec3, ndim=2, device=self._device)
        with instancer:
            wp.launch(
                update_line_transforms,
                dim=len(lines),
                inputs=[lines_wp],
                outputs=[instancer.vbo_transforms],
                device=self._device,
            )

    def render_line_list(self, name, vertices, indices, color, radius):
        """Add a line list as a set of capsules

        Args:
            vertices: The vertices of the line-list
            indices: The indices of the line-list
            color: The color of the line
            radius: The radius of the line
        """
        lines = []
        for i in range(len(indices) // 2):
            lines.append((vertices[indices[2 * i]], vertices[indices[2 * i + 1]]))
        lines = np.array(lines)
        self._render_lines(name, lines, color, radius)

    def render_line_strip(self, name: str, vertices, color: tuple, radius: float = 0.01):
        """Add a line strip as a set of capsules

        Args:
            vertices: The vertices of the line-strip
            color: The color of the line
            radius: The radius of the line
        """
        lines = []
        for i in range(len(vertices) - 1):
            lines.append((vertices[i], vertices[i + 1]))
        lines = np.array(lines)
        self._render_lines(name, lines, color, radius)

    def update_shape_vertices(self, shape, points):
        if isinstance(points, wp.array):
            wp_points = points.to(self._device)
        else:
            wp_points = wp.array(points, dtype=wp.vec3, device=self._device)

        cuda_buffer = self._shape_gl_buffers[shape][4]
        vertices_shape = self._shapes[shape][0].shape
        vbo_vertices = cuda_buffer.map(dtype=wp.float32, shape=vertices_shape)

        wp.launch(
            update_vbo_vertices,
            dim=vertices_shape[0],
            inputs=[wp_points],
            outputs=[vbo_vertices],
            device=self._device,
        )

        cuda_buffer.unmap()

    @staticmethod
    def _create_sphere_mesh(radius=1.0, num_latitudes=default_num_segments, num_longitudes=default_num_segments, reverse_winding=False):
        vertices = []
        indices = []

        for i in range(num_latitudes + 1):
            theta = i * np.pi / num_latitudes
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(num_longitudes + 1):
                phi = j * 2 * np.pi / num_longitudes
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                x = cos_phi * sin_theta
                y = cos_theta
                z = sin_phi * sin_theta

                u = float(j) / num_longitudes
                v = float(i) / num_latitudes

                vertices.append([x * radius, y * radius, z * radius, x, y, z, u, v])

        for i in range(num_latitudes):
            for j in range(num_longitudes):
                first = i * (num_longitudes + 1) + j
                second = first + num_longitudes + 1

                if reverse_winding:
                    indices.extend([first, second, first + 1, second, second + 1, first + 1])
                else:
                    indices.extend([first, first + 1, second, second, first + 1, second + 1])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    @staticmethod
    def _create_capsule_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
        vertices = []
        indices = []

        x_dir, y_dir, z_dir = ((1, 2, 0), (2, 0, 1), (0, 1, 2))[up_axis]
        up_vector = np.zeros(3)
        up_vector[up_axis] = half_height

        for i in range(segments + 1):
            theta = i * np.pi / segments
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(segments + 1):
                phi = j * 2 * np.pi / segments
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                z = cos_phi * sin_theta
                y = cos_theta
                x = sin_phi * sin_theta

                u = cos_theta * 0.5 + 0.5
                v = cos_phi * sin_theta * 0.5 + 0.5

                xyz = x, y, z
                x, y, z = xyz[x_dir], xyz[y_dir], xyz[z_dir]
                xyz = np.array((x, y, z), dtype=np.float32) * radius
                if j < segments // 2:
                    xyz += up_vector
                else:
                    xyz -= up_vector

                vertices.append([*xyz, x, y, z, u, v])

        nv = len(vertices)
        for i in range(segments + 1):
            for j in range(segments + 1):
                first = (i * (segments + 1) + j) % nv
                second = (first + segments + 1) % nv
                indices.extend([first, second, (first + 1) % nv, second, (second + 1) % nv, (first + 1) % nv])

        vertex_data = np.array(vertices, dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)

        return vertex_data, index_data

    @staticmethod
    def _create_cone_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
        # render it as a cylinder with zero top radius so we get correct normals on the sides
        return OpenGLRenderer._create_cylinder_mesh(radius, half_height, up_axis, segments, 0.0)

    @staticmethod
    def _create_cylinder_mesh(radius, half_height, up_axis=1, segments=default_num_segments, top_radius=None):
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be between 0 and 2")

        x_dir, y_dir, z_dir = (
            (1, 2, 0),
            (0, 1, 2),
            (2, 0, 1),
        )[up_axis]

        indices = []

        cap_vertices = []
        side_vertices = []

        # create center cap vertices
        position = np.array([0, -half_height, 0])[[x_dir, y_dir, z_dir]]
        normal = np.array([0, -1, 0])[[x_dir, y_dir, z_dir]]
        cap_vertices.append([*position, *normal, 0.5, 0.5])
        cap_vertices.append([*-position, *-normal, 0.5, 0.5])

        if top_radius is None:
            top_radius = radius
        side_slope = -np.arctan2(top_radius - radius, 2 * half_height)

        # Create the cylinder base and top vertices
        for j in (-1, 1):
            center_index = max(j, 0)
            if j == 1:
                radius = top_radius
            for i in range(segments):
                theta = 2 * np.pi * i / segments

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                x = cos_theta
                y = j * half_height
                z = sin_theta

                position = np.array([radius * x, y, radius * z])

                normal = np.array([x, side_slope, z])
                normal = normal / np.linalg.norm(normal)
                uv = (i / (segments - 1), (j + 1) / 2)
                vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
                side_vertices.append(vertex)

                normal = np.array([0, j, 0])
                uv = (cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5)
                vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
                cap_vertices.append(vertex)

                indices.extend(
                    [center_index, i + center_index * segments + 2,
                        (i + 1) % segments + center_index * segments + 2][::-j]
                )

        # Create the cylinder side indices
        for i in range(segments):
            index1 = len(cap_vertices) + i + segments
            index2 = len(cap_vertices) + ((i + 1) % segments) + segments
            index3 = len(cap_vertices) + i
            index4 = len(cap_vertices) + ((i + 1) % segments)

            indices.extend([index1, index2, index3, index2, index4, index3])

        vertex_data = np.array(np.vstack((cap_vertices, side_vertices)), dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)

        return vertex_data, index_data

    @staticmethod
    def _create_arrow_mesh(
        base_radius, base_height, cap_radius=None, cap_height=None, up_axis=1, segments=default_num_segments
    ):
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be between 0 and 2")
        if cap_radius is None:
            cap_radius = base_radius * 1.8
        if cap_height is None:
            cap_height = base_height * 0.18

        up_vector = np.array([0, 0, 0])
        up_vector[up_axis] = 1

        base_vertices, base_indices = OpenGLRenderer._create_cylinder_mesh(
            base_radius, base_height / 2, up_axis, segments
        )
        cap_vertices, cap_indices = OpenGLRenderer._create_cone_mesh(cap_radius, cap_height / 2, up_axis, segments)

        base_vertices[:, :3] += base_height / 2 * up_vector
        # move cap slightly lower to avoid z-fighting
        cap_vertices[:, :3] += (base_height + cap_height / 2 - 1e-3 * base_height) * up_vector

        vertex_data = np.vstack((base_vertices, cap_vertices))
        index_data = np.hstack((base_indices, cap_indices + len(base_vertices)))

        return vertex_data, index_data

    @staticmethod
    def _create_box_mesh(extents):
        x_extent, y_extent, z_extent = extents

        vertices = [
            # Position                        Normal    UV
            [-x_extent, -y_extent, -z_extent, -1, 0, 0, 0, 0],
            [-x_extent, -y_extent, z_extent, -1, 0, 0, 1, 0],
            [-x_extent, y_extent, z_extent, -1, 0, 0, 1, 1],
            [-x_extent, y_extent, -z_extent, -1, 0, 0, 0, 1],
            [x_extent, -y_extent, -z_extent, 1, 0, 0, 0, 0],
            [x_extent, -y_extent, z_extent, 1, 0, 0, 1, 0],
            [x_extent, y_extent, z_extent, 1, 0, 0, 1, 1],
            [x_extent, y_extent, -z_extent, 1, 0, 0, 0, 1],
            [-x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 0],
            [-x_extent, -y_extent, z_extent, 0, -1, 0, 1, 0],
            [x_extent, -y_extent, z_extent, 0, -1, 0, 1, 1],
            [x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 1],
            [-x_extent, y_extent, -z_extent, 0, 1, 0, 0, 0],
            [-x_extent, y_extent, z_extent, 0, 1, 0, 1, 0],
            [x_extent, y_extent, z_extent, 0, 1, 0, 1, 1],
            [x_extent, y_extent, -z_extent, 0, 1, 0, 0, 1],
            [-x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 0],
            [-x_extent, y_extent, -z_extent, 0, 0, -1, 1, 0],
            [x_extent, y_extent, -z_extent, 0, 0, -1, 1, 1],
            [x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 1],
            [-x_extent, -y_extent, z_extent, 0, 0, 1, 0, 0],
            [-x_extent, y_extent, z_extent, 0, 0, 1, 1, 0],
            [x_extent, y_extent, z_extent, 0, 0, 1, 1, 1],
            [x_extent, -y_extent, z_extent, 0, 0, 1, 0, 1],
        ]

        # fmt: off
        indices = [
            0, 1, 2,
            0, 2, 3,
            4, 6, 5,
            4, 7, 6,
            8, 10, 9,
            8, 11, 10,
            12, 13, 14,
            12, 14, 15,
            16, 17, 18,
            16, 18, 19,
            20, 22, 21,
            20, 23, 22,
        ]
        # fmt: on
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


if __name__ == "__main__":
    wp.init()
    renderer = OpenGLRenderer()
