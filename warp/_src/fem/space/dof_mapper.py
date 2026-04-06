# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from enum import Enum
from typing import Any

import warp as wp
from warp._src.types import type_scalar_type, type_size, types_equal

_wp_module_name_ = "warp.fem.space.dof_mapper"

vec6 = wp.types.vector(length=6, dtype=wp.float32)
vec6d = wp.types.vector(length=6, dtype=wp.float64)

_SQRT_2 = wp.constant(math.sqrt(2.0))
_SQRT_3 = wp.constant(math.sqrt(3.0))
_SQRT_1_2 = wp.constant(math.sqrt(1.0 / 2.0))
_SQRT_1_3 = wp.constant(math.sqrt(1.0 / 3.0))


class DofMapper:
    """Base class for mapping node degrees of freedom to function values."""

    value_dtype: type
    """Value type produced by this mapper."""
    dof_dtype: type
    """Degree-of-freedom type for each node."""
    DOF_SIZE: int
    """Number of scalar components per node degree of freedom."""

    @wp.func
    def dof_to_value(dof: Any):
        """Convert a degree-of-freedom value to a function value."""
        raise NotImplementedError

    @wp.func
    def value_to_dof(val: Any):
        """Convert a function value to a degree-of-freedom value."""
        raise NotImplementedError

    def __str__(self):
        """Return a short string identifier for the mapper."""
        return f"{self.value_dtype.__name__}_{self.DOF_SIZE}"


class IdentityMapper(DofMapper):
    """Identity mapper."""

    def __init__(self, dtype: type):
        if dtype == float:
            dtype = wp.float32

        self.value_dtype = dtype
        self.dof_dtype = dtype

        size = type_size(dtype)
        self.DOF_SIZE = wp.constant(size)

    @wp.func
    def dof_to_value(dof: Any):
        return dof

    @wp.func
    def value_to_dof(val: Any):
        return val


class SymmetricTensorMapper(DofMapper):
    """Orthonormal isomorphism from R^{n (n+1)} to nxn symmetric tensors,
    using usual L2 norm for vectors and half Frobenius norm, (tau : tau)/2 for tensors.
    """

    class Mapping(Enum):
        VOIGT = 0
        """Voigt ordering of vector coefficients:
            first the three diagonal terms, then off-diagonal coefficients"""
        DB16 = 1
        """Ordering that also separates normal from tangential coefficients:
           first trace, then other diagonal terms, then off-diagonal coefficients.
           See [Daviet and Bertails-Descoubes 2016]"""

    mapping: Mapping
    """Mapping convention for coefficient ordering."""

    def __init__(self, dtype: type, mapping: Mapping = Mapping.VOIGT):
        self.value_dtype = dtype
        self.mapping = mapping

        # Accept both fp32 and fp64 matrix types; the @wp.func overloads auto-dispatch
        scalar = type_scalar_type(dtype)
        if types_equal(dtype, wp.mat22) or types_equal(dtype, wp.mat22d):
            self.dof_dtype = wp.vec3 if scalar == wp.float32 else wp.vec3d
            self.DOF_SIZE = wp.constant(3)
            if mapping == SymmetricTensorMapper.Mapping.VOIGT:
                self.dof_to_value = SymmetricTensorMapper.dof_to_value_2d_voigt
                self.value_to_dof = SymmetricTensorMapper.value_to_dof_2d_voigt
            else:
                self.dof_to_value = SymmetricTensorMapper.dof_to_value_2d
                self.value_to_dof = SymmetricTensorMapper.value_to_dof_2d
        elif types_equal(dtype, wp.mat33) or types_equal(dtype, wp.mat33d):
            self.dof_dtype = vec6 if scalar == wp.float32 else vec6d
            self.DOF_SIZE = wp.constant(6)
            if mapping == SymmetricTensorMapper.Mapping.VOIGT:
                self.dof_to_value = SymmetricTensorMapper.dof_to_value_3d_voigt
                self.value_to_dof = SymmetricTensorMapper.value_to_dof_3d_voigt
            else:
                self.dof_to_value = SymmetricTensorMapper.dof_to_value_3d
                self.value_to_dof = SymmetricTensorMapper.value_to_dof_3d
        else:
            raise ValueError("Unsupported value dtype: ", dtype)

    def __str__(self):
        return f"{self.mapping}_{self.DOF_SIZE}"

    @wp.func
    def dof_to_value_2d(dof: wp.vec3):
        """Convert 2D symmetric DOFs to a matrix value."""
        a = dof[0]
        b = dof[1]
        c = dof[2]
        return wp.mat22(a + b, c, c, a - b)

    @wp.func
    def dof_to_value_2d(dof: wp.vec3d):
        a = dof[0]
        b = dof[1]
        c = dof[2]
        return wp.mat22d(a + b, c, c, a - b)

    @wp.func
    def value_to_dof_2d(val: wp.mat22):
        """Convert a 2D symmetric matrix to DOFs."""
        a = 0.5 * (val[0, 0] + val[1, 1])
        b = 0.5 * (val[0, 0] - val[1, 1])
        c = 0.5 * (val[0, 1] + val[1, 0])
        return wp.vec3(a, b, c)

    @wp.func
    def value_to_dof_2d(val: wp.mat22d):
        h = wp.float64(0.5)
        a = h * (val[0, 0] + val[1, 1])
        b = h * (val[0, 0] - val[1, 1])
        c = h * (val[0, 1] + val[1, 0])
        return wp.vec3d(a, b, c)

    @wp.func
    def dof_to_value_2d_voigt(dof: wp.vec3):
        """Convert 2D symmetric DOFs (Voigt) to a matrix value."""
        a = _SQRT_2 * dof[0]
        b = _SQRT_2 * dof[1]
        c = dof[2]
        return wp.mat22(a, c, c, b)

    @wp.func
    def dof_to_value_2d_voigt(dof: wp.vec3d):
        a = wp.float64(_SQRT_2) * dof[0]
        b = wp.float64(_SQRT_2) * dof[1]
        c = dof[2]
        return wp.mat22d(a, c, c, b)

    @wp.func
    def value_to_dof_2d_voigt(val: wp.mat22):
        """Convert a 2D symmetric matrix to DOFs (Voigt)."""
        a = _SQRT_1_2 * val[0, 0]
        b = _SQRT_1_2 * val[1, 1]
        c = 0.5 * (val[0, 1] + val[1, 0])
        return wp.vec3(a, b, c)

    @wp.func
    def value_to_dof_2d_voigt(val: wp.mat22d):
        h = wp.float64(0.5)
        a = wp.float64(_SQRT_1_2) * val[0, 0]
        b = wp.float64(_SQRT_1_2) * val[1, 1]
        c = h * (val[0, 1] + val[1, 0])
        return wp.vec3d(a, b, c)

    @wp.func
    def dof_to_value_3d(dof: vec6):
        """Convert 3D symmetric DOFs to a matrix value."""
        a = dof[0] * _SQRT_2 * _SQRT_1_3
        b = dof[1]
        c = dof[2] * _SQRT_1_3
        d = dof[3]
        e = dof[4]
        f = dof[5]
        return wp.mat33(
            a + b - c,
            f,
            e,
            f,
            a - b - c,
            d,
            e,
            d,
            a + 2.0 * c,
        )

    @wp.func
    def dof_to_value_3d(dof: vec6d):
        a = dof[0] * wp.float64(_SQRT_2) * wp.float64(_SQRT_1_3)
        b = dof[1]
        c = dof[2] * wp.float64(_SQRT_1_3)
        d = dof[3]
        e = dof[4]
        f = dof[5]
        return wp.mat33d(
            a + b - c,
            f,
            e,
            f,
            a - b - c,
            d,
            e,
            d,
            a + wp.float64(2.0) * c,
        )

    @wp.func
    def value_to_dof_3d(val: wp.mat33):
        """Convert a 3D symmetric matrix to DOFs."""
        a = (val[0, 0] + val[1, 1] + val[2, 2]) * _SQRT_1_3 * _SQRT_1_2
        b = 0.5 * (val[0, 0] - val[1, 1])
        c = 0.5 * (val[2, 2] - (val[0, 0] + val[1, 1] + val[2, 2]) / 3.0) * _SQRT_3

        d = 0.5 * (val[2, 1] + val[1, 2])
        e = 0.5 * (val[0, 2] + val[2, 0])
        f = 0.5 * (val[1, 0] + val[0, 1])

        return vec6(a, b, c, d, e, f)

    @wp.func
    def value_to_dof_3d(val: wp.mat33d):
        h = wp.float64(0.5)
        t = wp.float64(3.0)
        a = (val[0, 0] + val[1, 1] + val[2, 2]) * wp.float64(_SQRT_1_3) * wp.float64(_SQRT_1_2)
        b = h * (val[0, 0] - val[1, 1])
        c = h * (val[2, 2] - (val[0, 0] + val[1, 1] + val[2, 2]) / t) * wp.float64(_SQRT_3)

        d = h * (val[2, 1] + val[1, 2])
        e = h * (val[0, 2] + val[2, 0])
        f = h * (val[1, 0] + val[0, 1])

        return vec6d(a, b, c, d, e, f)

    @wp.func
    def dof_to_value_3d_voigt(dof: vec6):
        """Convert 3D symmetric DOFs (Voigt) to a matrix value."""
        a = _SQRT_2 * dof[0]
        b = _SQRT_2 * dof[1]
        c = _SQRT_2 * dof[2]
        d = dof[3]
        e = dof[4]
        f = dof[5]
        return wp.mat33(
            a,
            f,
            e,
            f,
            b,
            d,
            e,
            d,
            c,
        )

    @wp.func
    def dof_to_value_3d_voigt(dof: vec6d):
        a = wp.float64(_SQRT_2) * dof[0]
        b = wp.float64(_SQRT_2) * dof[1]
        c = wp.float64(_SQRT_2) * dof[2]
        d = dof[3]
        e = dof[4]
        f = dof[5]
        return wp.mat33d(
            a,
            f,
            e,
            f,
            b,
            d,
            e,
            d,
            c,
        )

    @wp.func
    def value_to_dof_3d_voigt(val: wp.mat33):
        """Convert a 3D symmetric matrix to DOFs (Voigt)."""
        a = _SQRT_1_2 * val[0, 0]
        b = _SQRT_1_2 * val[1, 1]
        c = _SQRT_1_2 * val[2, 2]

        d = 0.5 * (val[2, 1] + val[1, 2])
        e = 0.5 * (val[0, 2] + val[2, 0])
        f = 0.5 * (val[1, 0] + val[0, 1])

        return vec6(a, b, c, d, e, f)

    @wp.func
    def value_to_dof_3d_voigt(val: wp.mat33d):
        h = wp.float64(0.5)
        a = wp.float64(_SQRT_1_2) * val[0, 0]
        b = wp.float64(_SQRT_1_2) * val[1, 1]
        c = wp.float64(_SQRT_1_2) * val[2, 2]

        d = h * (val[2, 1] + val[1, 2])
        e = h * (val[0, 2] + val[2, 0])
        f = h * (val[1, 0] + val[0, 1])

        return vec6d(a, b, c, d, e, f)


class SkewSymmetricTensorMapper(DofMapper):
    """Orthonormal isomorphism from R^{n (n-1)} to nxn skew-symmetric tensors,
    using usual L2 norm for vectors and half Frobenius norm, (tau : tau)/2 for tensors.
    """

    def __init__(self, dtype: type):
        self.value_dtype = dtype

        # Accept both fp32 and fp64 matrix types; the @wp.func overloads auto-dispatch
        scalar = type_scalar_type(dtype)
        if types_equal(dtype, wp.mat22) or types_equal(dtype, wp.mat22d):
            self.dof_dtype = float if scalar == wp.float32 else wp.float64
            self.DOF_SIZE = wp.constant(1)
            self.dof_to_value = SkewSymmetricTensorMapper.dof_to_value_2d
            self.value_to_dof = SkewSymmetricTensorMapper.value_to_dof_2d
        elif types_equal(dtype, wp.mat33) or types_equal(dtype, wp.mat33d):
            self.dof_dtype = wp.vec3 if scalar == wp.float32 else wp.vec3d
            self.DOF_SIZE = wp.constant(3)
            self.dof_to_value = SkewSymmetricTensorMapper.dof_to_value_3d
            self.value_to_dof = SkewSymmetricTensorMapper.value_to_dof_3d
        else:
            raise ValueError("Unsupported value dtype: ", dtype)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.DOF_SIZE}"

    @wp.func
    def dof_to_value_2d(dof: float):
        """Convert 2D skew-symmetric DOFs to a matrix value."""
        return wp.mat22(0.0, -dof, dof, 0.0)

    @wp.func
    def dof_to_value_2d(dof: wp.float64):
        z = wp.float64(0.0)
        return wp.mat22d(z, -dof, dof, z)

    @wp.func
    def value_to_dof_2d(val: wp.mat22):
        """Convert a 2D skew-symmetric matrix to DOFs."""
        return 0.5 * (val[1, 0] - val[0, 1])

    @wp.func
    def value_to_dof_2d(val: wp.mat22d):
        return wp.float64(0.5) * (val[1, 0] - val[0, 1])

    @wp.func
    def dof_to_value_3d(dof: wp.vec3):
        """Convert 3D skew-symmetric DOFs to a matrix value."""
        a = dof[0]
        b = dof[1]
        c = dof[2]
        return wp.mat33(0.0, -c, b, c, 0.0, -a, -b, a, 0.0)

    @wp.func
    def dof_to_value_3d(dof: wp.vec3d):
        a = dof[0]
        b = dof[1]
        c = dof[2]
        z = wp.float64(0.0)
        return wp.mat33d(z, -c, b, c, z, -a, -b, a, z)

    @wp.func
    def value_to_dof_3d(val: wp.mat33):
        """Convert a 3D skew-symmetric matrix to DOFs."""
        a = 0.5 * (val[2, 1] - val[1, 2])
        b = 0.5 * (val[0, 2] - val[2, 0])
        c = 0.5 * (val[1, 0] - val[0, 1])
        return wp.vec3(a, b, c)

    @wp.func
    def value_to_dof_3d(val: wp.mat33d):
        h = wp.float64(0.5)
        a = h * (val[2, 1] - val[1, 2])
        b = h * (val[0, 2] - val[2, 0])
        c = h * (val[1, 0] - val[0, 1])
        return wp.vec3d(a, b, c)
