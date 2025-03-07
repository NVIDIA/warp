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

"""Helper functions for computing rigid body inertia properties."""

import math
from typing import List, Union

import numpy as np

import warp as wp


@wp.func
def triangle_inertia(
    p: wp.vec3,
    q: wp.vec3,
    r: wp.vec3,
    density: float,
    com: wp.vec3,
    # outputs
    mass: wp.array(dtype=float, ndim=1),
    inertia: wp.array(dtype=wp.mat33, ndim=1),
):
    pcom = p - com
    qcom = q - com
    rcom = r - com

    Dm = wp.mat33(pcom[0], qcom[0], rcom[0], pcom[1], qcom[1], rcom[1], pcom[2], qcom[2], rcom[2])

    volume = wp.abs(wp.determinant(Dm) / 6.0)

    # accumulate mass
    wp.atomic_add(mass, 0, 4.0 * density * volume)

    alpha = wp.sqrt(5.0) / 5.0
    mid = (com + p + q + r) / 4.0
    off_mid = mid - com

    # displacement of quadrature point from COM
    d0 = alpha * (p - mid) + off_mid
    d1 = alpha * (q - mid) + off_mid
    d2 = alpha * (r - mid) + off_mid
    d3 = alpha * (com - mid) + off_mid

    # accumulate inertia
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    I = wp.dot(d0, d0) * identity - wp.outer(d0, d0)
    I += wp.dot(d1, d1) * identity - wp.outer(d1, d1)
    I += wp.dot(d2, d2) * identity - wp.outer(d2, d2)
    I += wp.dot(d3, d3) * identity - wp.outer(d3, d3)

    wp.atomic_add(inertia, 0, (density * volume) * I)

    return volume


@wp.kernel
def compute_solid_mesh_inertia(
    # inputs
    com: wp.vec3,
    weight: float,
    indices: wp.array(dtype=int, ndim=1),
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    # outputs
    mass: wp.array(dtype=float, ndim=1),
    inertia: wp.array(dtype=wp.mat33, ndim=1),
    volume: wp.array(dtype=float, ndim=1),
):
    i = wp.tid()

    p = vertices[indices[i * 3 + 0]]
    q = vertices[indices[i * 3 + 1]]
    r = vertices[indices[i * 3 + 2]]

    vol = triangle_inertia(p, q, r, weight, com, mass, inertia)
    wp.atomic_add(volume, 0, vol)


@wp.kernel
def compute_hollow_mesh_inertia(
    # inputs
    com: wp.vec3,
    density: float,
    indices: wp.array(dtype=int, ndim=1),
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    thickness: wp.array(dtype=float, ndim=1),
    # outputs
    mass: wp.array(dtype=float, ndim=1),
    inertia: wp.array(dtype=wp.mat33, ndim=1),
    volume: wp.array(dtype=float, ndim=1),
):
    tid = wp.tid()
    i = indices[tid * 3 + 0]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]

    vi = vertices[i]
    vj = vertices[j]
    vk = vertices[k]

    normal = -wp.normalize(wp.cross(vj - vi, vk - vi))
    ti = normal * thickness[i]
    tj = normal * thickness[j]
    tk = normal * thickness[k]

    # wedge vertices
    vi0 = vi - ti
    vi1 = vi + ti
    vj0 = vj - tj
    vj1 = vj + tj
    vk0 = vk - tk
    vk1 = vk + tk

    triangle_inertia(vi0, vj0, vk0, density, com, mass, inertia)
    triangle_inertia(vj0, vk1, vk0, density, com, mass, inertia)
    triangle_inertia(vj0, vj1, vk1, density, com, mass, inertia)
    triangle_inertia(vj0, vi1, vj1, density, com, mass, inertia)
    triangle_inertia(vj0, vi0, vi1, density, com, mass, inertia)
    triangle_inertia(vj1, vi1, vk1, density, com, mass, inertia)
    triangle_inertia(vi1, vi0, vk0, density, com, mass, inertia)
    triangle_inertia(vi1, vk0, vk1, density, com, mass, inertia)

    # compute volume
    a = wp.length(wp.cross(vj - vi, vk - vi)) * 0.5
    vol = a * (thickness[i] + thickness[j] + thickness[k]) / 3.0
    wp.atomic_add(volume, 0, vol)


def compute_sphere_inertia(density: float, r: float) -> tuple:
    """Helper to compute mass and inertia of a solid sphere

    Args:
        density: The sphere density
        r: The sphere radius

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    v = 4.0 / 3.0 * math.pi * r * r * r

    m = density * v
    Ia = 2.0 / 5.0 * m * r * r

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

    return (m, wp.vec3(), I)


def compute_capsule_inertia(density: float, r: float, h: float) -> tuple:
    """Helper to compute mass and inertia of a solid capsule extending along the y-axis

    Args:
        density: The capsule density
        r: The capsule radius
        h: The capsule height (full height of the interior cylinder)

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    ms = density * (4.0 / 3.0) * math.pi * r * r * r
    mc = density * math.pi * r * r * h

    # total mass
    m = ms + mc

    # adapted from ODE
    Ia = mc * (0.25 * r * r + (1.0 / 12.0) * h * h) + ms * (0.4 * r * r + 0.375 * r * h + 0.25 * h * h)
    Ib = (mc * 0.5 + ms * 0.4) * r * r

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ia]])

    return (m, wp.vec3(), I)


def compute_cylinder_inertia(density: float, r: float, h: float) -> tuple:
    """Helper to compute mass and inertia of a solid cylinder extending along the y-axis

    Args:
        density: The cylinder density
        r: The cylinder radius
        h: The cylinder height (extent along the y-axis)

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    m = density * math.pi * r * r * h

    Ia = 1 / 12 * m * (3 * r * r + h * h)
    Ib = 1 / 2 * m * r * r

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ia]])

    return (m, wp.vec3(), I)


def compute_cone_inertia(density: float, r: float, h: float) -> tuple:
    """Helper to compute mass and inertia of a solid cone extending along the y-axis

    Args:
        density: The cone density
        r: The cone radius
        h: The cone height (extent along the y-axis)

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    m = density * math.pi * r * r * h / 3.0

    Ia = 1 / 20 * m * (3 * r * r + 2 * h * h)
    Ib = 3 / 10 * m * r * r

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ia]])

    return (m, wp.vec3(), I)


def compute_box_inertia(density: float, w: float, h: float, d: float) -> tuple:
    """Helper to compute mass and inertia of a solid box

    Args:
        density: The box density
        w: The box width along the x-axis
        h: The box height along the y-axis
        d: The box depth along the z-axis

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    v = w * h * d
    m = density * v

    Ia = 1.0 / 12.0 * m * (h * h + d * d)
    Ib = 1.0 / 12.0 * m * (w * w + d * d)
    Ic = 1.0 / 12.0 * m * (w * w + h * h)

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ic]])

    return (m, wp.vec3(), I)


def compute_mesh_inertia(
    density: float, vertices: list, indices: list, is_solid: bool = True, thickness: Union[List[float], float] = 0.001
) -> tuple:
    """Computes mass, center of mass, 3x3 inertia matrix, and volume for a mesh."""
    com = wp.vec3(np.mean(vertices, 0))

    indices = np.array(indices).flatten()
    num_tris = len(indices) // 3

    # compute signed inertia for each tetrahedron
    # formed with the interior point, using an order-2
    # quadrature: https://www.sciencedirect.com/science/article/pii/S0377042712001604#br000040

    # Allocating for mass and inertia
    I_warp = wp.zeros(1, dtype=wp.mat33)
    mass_warp = wp.zeros(1, dtype=float)
    vol_warp = wp.zeros(1, dtype=float)

    if is_solid:
        weight = 0.25
        # alpha = math.sqrt(5.0) / 5.0
        wp.launch(
            kernel=compute_solid_mesh_inertia,
            dim=num_tris,
            inputs=[
                com,
                weight,
                wp.array(indices, dtype=int),
                wp.array(vertices, dtype=wp.vec3),
            ],
            outputs=[mass_warp, I_warp, vol_warp],
        )
    else:
        weight = 0.25 * density
        if isinstance(thickness, float):
            thickness = [thickness] * len(vertices)
        wp.launch(
            kernel=compute_hollow_mesh_inertia,
            dim=num_tris,
            inputs=[
                com,
                weight,
                wp.array(indices, dtype=int),
                wp.array(vertices, dtype=wp.vec3),
                wp.array(thickness, dtype=float),
            ],
            outputs=[mass_warp, I_warp, vol_warp],
        )

    # Extract mass and inertia and save to class attributes.
    mass = float(mass_warp.numpy()[0] * density)
    I = wp.mat33(*(I_warp.numpy()[0] * density))
    volume = float(vol_warp.numpy()[0])
    return mass, com, I, volume


def transform_inertia(m, I, p, q):
    R = wp.quat_to_matrix(q)

    # Steiner's theorem
    return R @ I @ wp.transpose(R) + m * (wp.dot(p, p) * wp.mat33(np.eye(3)) - wp.outer(p, p))
