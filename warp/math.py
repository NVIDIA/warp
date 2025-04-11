# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

import warp as wp

"""
Vector norm functions
"""

__all__ = [
    "norm_huber",
    "norm_l1",
    "norm_l2",
    "norm_pseudo_huber",
    "smooth_normalize",
    "transform_compose",
    "transform_decompose",
    "transform_from_matrix",
    "transform_to_matrix",
]


@wp.func
def norm_l1(v: Any):
    """
    Computes the L1 norm of a vector v.

    .. math:: \\|v\\|_1 = \\sum_i |v_i|

    Args:
        v (Vector[Any,Float]): The vector to compute the L1 norm of.

    Returns:
        float: The L1 norm of the vector.
    """
    n = float(0.0)
    for i in range(len(v)):
        n += wp.abs(v[i])
    return n


@wp.func
def norm_l2(v: Any):
    """
    Computes the L2 norm of a vector v.

    .. math:: \\|v\\|_2 = \\sqrt{\\sum_i v_i^2}

    Args:
        v (Vector[Any,Float]): The vector to compute the L2 norm of.

    Returns:
        float: The L2 norm of the vector.
    """
    return wp.length(v)


@wp.func
def norm_huber(v: Any, delta: float = 1.0):
    """
    Computes the Huber norm of a vector v with a given delta.

    .. math::
        H(v) = \\begin{cases} \\frac{1}{2} \\|v\\|^2 & \\text{if } \\|v\\| \\leq \\delta \\\\ \\delta(\\|v\\| - \\frac{1}{2}\\delta) & \\text{otherwise} \\end{cases}

    .. image:: /img/norm_huber.svg
        :align: center

    Args:
        v (Vector[Any,Float]): The vector to compute the Huber norm of.
        delta (float): The threshold value, defaults to 1.0.

    Returns:
        float: The Huber norm of the vector.
    """
    a = wp.dot(v, v)
    if a <= delta * delta:
        return 0.5 * a
    return delta * (wp.sqrt(a) - 0.5 * delta)


@wp.func
def norm_pseudo_huber(v: Any, delta: float = 1.0):
    """
    Computes the "pseudo" Huber norm of a vector v with a given delta.

    .. math::
        H^\\prime(v) = \\delta \\sqrt{1 + \\frac{\\|v\\|^2}{\\delta^2}}

    .. image:: /img/norm_pseudo_huber.svg
        :align: center

    Args:
        v (Vector[Any,Float]): The vector to compute the Huber norm of.
        delta (float): The threshold value, defaults to 1.0.

    Returns:
        float: The Huber norm of the vector.
    """
    a = wp.dot(v, v)
    return delta * wp.sqrt(1.0 + a / (delta * delta))


@wp.func
def smooth_normalize(v: Any, delta: float = 1.0):
    """
    Normalizes a vector using the pseudo-Huber norm.

    See :func:`norm_pseudo_huber`.

    .. math::
        \\frac{v}{H^\\prime(v)}

    Args:
        v (Vector[Any,Float]): The vector to normalize.
        delta (float): The threshold value, defaults to 1.0.

    Returns:
        Vector[Any,Float]: The normalized vector.
    """
    return v / norm_pseudo_huber(v, delta)


def create_transform_from_matrix_func(dtype):
    mat44 = wp.types.matrix((4, 4), dtype)
    vec3 = wp.types.vector(3, dtype)
    transform = wp.types.transformation(dtype)

    def transform_from_matrix(mat: mat44) -> transform:
        """
        Construct a transformation from a 4x4 matrix.

        .. math::
            M = \\begin{bmatrix}
            R_{00} & R_{01} & R_{02} & p_x \\\\
            R_{10} & R_{11} & R_{12} & p_y \\\\
            R_{20} & R_{21} & R_{22} & p_z \\\\
            0 & 0 & 0 & 1
            \\end{bmatrix}

        Where:

        * :math:`R` is the 3x3 rotation matrix created from the orientation quaternion of the input transform.
        * :math:`p` is the 3D position vector :math:`[p_x, p_y, p_z]` of the input transform.

        Args:
            mat (Matrix[4, 4, Float]): Matrix to convert.

        Returns:
            Transformation[Float]: The transformation.
        """
        p = vec3(mat[0][3], mat[1][3], mat[2][3])
        q = wp.quat_from_matrix(mat)
        return transform(p, q)

    return transform_from_matrix


transform_from_matrix = wp.func(
    create_transform_from_matrix_func(wp.float32),
    name="transform_from_matrix",
)
wp.func(
    create_transform_from_matrix_func(wp.float16),
    name="transform_from_matrix",
)
wp.func(
    create_transform_from_matrix_func(wp.float64),
    name="transform_from_matrix",
)


def create_transform_to_matrix_func(dtype):
    mat44 = wp.types.matrix((4, 4), dtype)
    transform = wp.types.transformation(dtype)

    def transform_to_matrix(xform: transform) -> mat44:
        """
        Convert a transformation to a 4x4 matrix.

        .. math::
            M = \\begin{bmatrix}
            R_{00} & R_{01} & R_{02} & p_x \\\\
            R_{10} & R_{11} & R_{12} & p_y \\\\
            R_{20} & R_{21} & R_{22} & p_z \\\\
            0 & 0 & 0 & 1
            \\end{bmatrix}

        Where:

        * :math:`R` is the 3x3 rotation matrix created from the orientation quaternion of the input transform.
        * :math:`p` is the 3D position vector :math:`[p_x, p_y, p_z]` of the input transform.

        Args:
            xform (Transformation[Float]): Transformation to convert.

        Returns:
            Matrix[4, 4, Float]: The matrix.
        """
        p = wp.transform_get_translation(xform)
        q = wp.transform_get_rotation(xform)
        rot = wp.quat_to_matrix(q)
        # fmt: off
        return mat44(
            rot[0][0], rot[0][1], rot[0][2], p[0],
            rot[1][0], rot[1][1], rot[1][2], p[1],
            rot[2][0], rot[2][1], rot[2][2], p[2],
            dtype(0.0), dtype(0.0), dtype(0.0), dtype(1.0),
        )
        # fmt: on

    return transform_to_matrix


transform_to_matrix = wp.func(
    create_transform_to_matrix_func(wp.float32),
    name="transform_to_matrix",
)
wp.func(
    create_transform_to_matrix_func(wp.float16),
    name="transform_to_matrix",
)
wp.func(
    create_transform_to_matrix_func(wp.float64),
    name="transform_to_matrix",
)


def create_transform_compose_func(dtype):
    mat44 = wp.types.matrix((4, 4), dtype)
    quat = wp.types.quaternion(dtype)
    vec3 = wp.types.vector(3, dtype)

    def transform_compose(position: vec3, rotation: quat, scale: vec3):
        """
        Compose a 4x4 transformation matrix from a 3D position, quaternion orientation, and 3D scale.

        .. math::
            M = \\begin{bmatrix}
            s_x R_{00} & s_y R_{01} & s_z R_{02} & p_x \\\\
            s_x R_{10} & s_y R_{11} & s_z R_{12} & p_y \\\\
            s_x R_{20} & s_y R_{21} & s_z R_{22} & p_z \\\\
            0 & 0 & 0 & 1
            \\end{bmatrix}

        Where:

        * :math:`R` is the 3x3 rotation matrix created from the orientation quaternion of the input transform.
        * :math:`p` is the 3D position vector :math:`[p_x, p_y, p_z]` of the input transform.
        * :math:`s` is the 3D scale vector :math:`[s_x, s_y, s_z]` of the input transform.

        Args:
            position (Vector[3, Float]): The 3D position vector.
            rotation (Quaternion[Float]): The quaternion orientation.
            scale (Vector[3, Float]): The 3D scale vector.

        Returns:
            Matrix[4, 4, Float]: The transformation matrix.
        """
        R = wp.quat_to_matrix(rotation)
        # fmt: off
        return mat44(
            scale[0] * R[0,0], scale[1] * R[0,1], scale[2] * R[0,2], position[0],
            scale[0] * R[1,0], scale[1] * R[1,1], scale[2] * R[1,2], position[1],
            scale[0] * R[2,0], scale[1] * R[2,1], scale[2] * R[2,2], position[2],
            dtype(0.0), dtype(0.0), dtype(0.0), dtype(1.0),
        )
        # fmt: on

    return transform_compose


transform_compose = wp.func(
    create_transform_compose_func(wp.float32),
    name="transform_compose",
)
wp.func(
    create_transform_compose_func(wp.float16),
    name="transform_compose",
)
wp.func(
    create_transform_compose_func(wp.float64),
    name="transform_compose",
)


def create_transform_decompose_func(dtype):
    mat44 = wp.types.matrix((4, 4), dtype)
    vec3 = wp.types.vector(3, dtype)
    mat33 = wp.types.matrix((3, 3), dtype)
    zero = dtype(0.0)

    def transform_decompose(m: mat44):
        """
        Decompose a 4x4 transformation matrix into 3D position, quaternion orientation, and 3D scale.

        .. math::
            M = \\begin{bmatrix}
            s_x R_{00} & s_y R_{01} & s_z R_{02} & p_x \\\\
            s_x R_{10} & s_y R_{11} & s_z R_{12} & p_y \\\\
            s_x R_{20} & s_y R_{21} & s_z R_{22} & p_z \\\\
            0 & 0 & 0 & 1
            \\end{bmatrix}

        Where:

        * :math:`R` is the 3x3 rotation matrix created from the orientation quaternion of the input transform.
        * :math:`p` is the 3D position vector :math:`[p_x, p_y, p_z]` of the input transform.
        * :math:`s` is the 3D scale vector :math:`[s_x, s_y, s_z]` of the input transform.

        Args:
            m (Matrix[4, 4, Float]): The matrix to decompose.

        Returns:
            Tuple[Vector[3, Float], Quaternion[Float], Vector[3, Float]]: A tuple containing the position vector, quaternion orientation, and scale vector.
        """
        # extract position
        position = vec3(m[0, 3], m[1, 3], m[2, 3])
        # extract rotation matrix components
        r00, r01, r02 = m[0, 0], m[0, 1], m[0, 2]
        r10, r11, r12 = m[1, 0], m[1, 1], m[1, 2]
        r20, r21, r22 = m[2, 0], m[2, 1], m[2, 2]
        # get scale magnitudes
        sx = wp.sqrt(r00 * r00 + r10 * r10 + r20 * r20)
        sy = wp.sqrt(r01 * r01 + r11 * r11 + r21 * r21)
        sz = wp.sqrt(r02 * r02 + r12 * r12 + r22 * r22)
        # normalize rotation matrix components
        if sx != zero:
            r00 /= sx
            r10 /= sx
            r20 /= sx
        if sy != zero:
            r01 /= sy
            r11 /= sy
            r21 /= sy
        if sz != zero:
            r02 /= sz
            r12 /= sz
            r22 /= sz
        # extract rotation (quaternion)
        rotation = wp.quat_from_matrix(mat33(r00, r01, r02, r10, r11, r12, r20, r21, r22))
        # extract scale
        scale = vec3(sx, sy, sz)
        return position, rotation, scale

    return transform_decompose


transform_decompose = wp.func(
    create_transform_decompose_func(wp.float32),
    name="transform_decompose",
)
wp.func(
    create_transform_decompose_func(wp.float16),
    name="transform_decompose",
)
wp.func(
    create_transform_decompose_func(wp.float64),
    name="transform_decompose",
)


# register API functions so they appear in the documentation

wp.context.register_api_function(
    norm_l1,
    group="Vector Math",
)
wp.context.register_api_function(
    norm_l2,
    group="Vector Math",
)
wp.context.register_api_function(
    norm_huber,
    group="Vector Math",
)
wp.context.register_api_function(
    norm_pseudo_huber,
    group="Vector Math",
)
wp.context.register_api_function(
    smooth_normalize,
    group="Vector Math",
)
wp.context.register_api_function(
    transform_from_matrix,
    group="Transformations",
)
wp.context.register_api_function(
    transform_to_matrix,
    group="Transformations",
)
wp.context.register_api_function(
    transform_compose,
    group="Transformations",
)
wp.context.register_api_function(
    transform_decompose,
    group="Transformations",
)
