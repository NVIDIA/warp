# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""A module for building simulation models and state.
"""

import warp as wp
import numpy as np

import math
import copy

from typing import List, Optional, Tuple, Union

from warp.types import Volume
Vec3 = List[float]
Vec4 = List[float]
Quat = List[float]
Mat33 = List[float]
Transform = Tuple[Vec3, Quat]

# Shape geometry types
GEO_SPHERE = wp.constant(0)
GEO_BOX = wp.constant(1)
GEO_CAPSULE = wp.constant(2)
GEO_CYLINDER = wp.constant(3)
GEO_CONE = wp.constant(4)
GEO_MESH = wp.constant(5)
GEO_SDF = wp.constant(6)
GEO_PLANE = wp.constant(7)
GEO_NONE = wp.constant(8)

# Shape properties of geometry
@wp.struct
class ModelShapeGeometry:
    type: wp.array(dtype=wp.int32)  # The type of geometry (GEO_SPHERE, GEO_BOX, etc.)
    is_solid: wp.array(dtype=wp.uint8)  # Indicates whether the shape is solid or hollow
    thickness: wp.array(dtype=float)    # The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)
    source: wp.array(dtype=wp.uint64)   # Pointer to the source geometry (in case of a mesh, zero otherwise)
    scale: wp.array(dtype=wp.vec3)      # The 3D scale of the shape


@wp.func
def triangle_inertia(
    p: wp.vec3,
    q: wp.vec3,
    r: wp.vec3,
    density: float,
    com: wp.vec3,
    # outputs
    mass: wp.array(dtype=float, ndim=1),
    inertia: wp.array(dtype=wp.mat33, ndim=1)):

    pcom = p - com
    qcom = q - com
    rcom = r - com

    Dm = wp.mat33(pcom[0], qcom[0], rcom[0],
                  pcom[1], qcom[1], rcom[1],
                  pcom[2], qcom[2], rcom[2])

    volume = wp.determinant(Dm) / 6.0

    # accumulate mass
    wp.atomic_add(mass, 0, 4.0 * density * volume)

    alpha = wp.sqrt(5.0) / 5.0
    mid = (com + p + q + r) / 4.
    off_mid = mid - com

    # displacement of quadrature point from COM
    d0 = alpha * (p - mid) + off_mid
    d1 = alpha * (q - mid) + off_mid
    d2 = alpha * (r - mid) + off_mid
    d3 = alpha * (com - mid) + off_mid

    # accumulate inertia
    identity = wp.mat33(1., 0., 0., 0., 1., 0., 0., 0., 1.)
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
    volume: wp.array(dtype=float, ndim=1)):

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
    volume: wp.array(dtype=float, ndim=1)):
    
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

    I = np.array([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

    return (m, np.zeros(3), I)


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

    I = np.array([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ia]])

    return (m, np.zeros(3), I)


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

    Ia = 1/12 * m * (3 * r * r + h * h)
    Ib = 1/2 * m * r * r

    I = np.array([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ia]])

    return (m, np.zeros(3), I)


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

    Ia = 1/20 * m * (3 * r * r + 2 * h * h)
    Ib = 3/10 * m * r * r

    I = np.array([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ia]])

    return (m, np.zeros(3), I)


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

    I = np.array([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ic]])

    return (m, np.zeros(3), I)


def compute_mesh_inertia(density: float, vertices: list, indices: list, is_solid: bool = True, thickness: Union[List[float], float] = 0.001) -> tuple:
    """Computes mass, center of mass, 3x3 inertia matrix, and volume for a mesh."""
    com = np.mean(vertices, 0)
    com_warp = wp.vec3(com[0], com[1], com[2])

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
        alpha = math.sqrt(5.0) / 5.0
        wp.launch(kernel=compute_solid_mesh_inertia,
                    dim=num_tris,
                    inputs=[
                        com_warp,
                        weight,
                        wp.array(indices, dtype=int),
                        wp.array(vertices, dtype=wp.vec3),
                        ],
                    outputs=[
                        mass_warp,
                        I_warp,
                        vol_warp])
    else:
        weight = 0.25 * density
        if isinstance(thickness, float):
            thickness = [thickness] * len(vertices)
        wp.launch(kernel=compute_hollow_mesh_inertia,
                    dim=num_tris,
                    inputs=[
                        com_warp,
                        weight,
                        wp.array(indices, dtype=int),
                        wp.array(vertices, dtype=wp.vec3),
                        wp.array(thickness, dtype=float),
                        ],
                    outputs=[
                        mass_warp,
                        I_warp,
                        vol_warp])

    # Extract mass and inertia and save to class attributes.
    mass = mass_warp.numpy()[0] * density
    I = I_warp.numpy()[0] * density
    volume = vol_warp.numpy()[0]
    return mass, com, I, volume

def compute_shape_mass(type, scale, src, density, is_solid, thickness):
    
    if density == 0.0 or type == GEO_PLANE:     # zero density means fixed
        return 0.0, np.zeros(3), np.zeros((3, 3))

    if (type == GEO_SPHERE):
        solid = compute_sphere_inertia(density, scale[0])
        if is_solid:
            return solid
        else:
            hollow = compute_sphere_inertia(density, scale[0] - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif (type == GEO_BOX):
        w, h, d = np.array(scale[:3]) * 2.0
        solid = compute_box_inertia(density, w, h, d)
        if is_solid:
            return solid
        else:
            hollow = compute_box_inertia(density, w - thickness, h - thickness, d - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif (type == GEO_CAPSULE):
        r, h = scale[0], scale[1] * 2.0
        solid = compute_capsule_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            hollow = compute_capsule_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif (type == GEO_CYLINDER):
        r, h = scale[0], scale[1] * 2.0
        solid = compute_cylinder_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            hollow = compute_cylinder_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif (type == GEO_CONE):
        r, h = scale[0], scale[1] * 2.0
        solid = compute_cone_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            hollow = compute_cone_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif (type == GEO_MESH):
        if src.mass > 0.0 and src.is_solid == is_solid:
            s = scale[0]
            return (density * src.mass * s * s * s, src.com, density * src.I * s * s * s * s * s)
        else:
            # fall back to computing inertia from mesh geometry
            vertices = np.array(src.vertices) * np.array(scale[:3])
            m, c, I, vol = compute_mesh_inertia(density, vertices, src.indices, is_solid, thickness)
            return m, c, I
    raise ValueError("Unsupported shape type: {}".format(type))


def transform_inertia(m, I, p, q):
    R = np.array(wp.quat_to_matrix(q)).reshape(3,3)

    # Steiner's theorem
    return R @ I @ R.T + m * (np.dot(p, p) * np.eye(3) - np.outer(p, p))