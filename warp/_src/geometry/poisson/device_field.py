# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device field handles and single-query evaluation for application kernels."""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from .field import evaluation_function, query_vjp_function, reference_evaluation_function

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def _precision_functions(dtype):
    vector = wp.types.vector(length=3, dtype=dtype)
    matrix = wp.types.matrix(shape=(3, 3), dtype=dtype)
    transform = wp.types.matrix(shape=(4, 4), dtype=dtype)
    direct_query = evaluation_function(dtype)
    reference_query = reference_evaluation_function(dtype)

    class DeviceField:
        """Hold a zero-copy Poisson field for device queries.

        Obtain a handle from ``PoissonReconstruction.device_field()``. Its arrays
        belong to the reconstruction; keep that result alive and immutable through
        kernel execution and backward. The layout is an implementation detail.
        """

        keys: wp.array[wp.int64]
        coefficients: wp.array[dtype]
        cumulative: wp.array[dtype]
        linear: matrix
        translation: vector
        depth: int
        reference: bool
        transforms: wp.array[transform]
        has_transform: bool

    DeviceField.__name__ = "PoissonField64" if dtype == wp.float64 else "PoissonField"
    DeviceField.__qualname__ = DeviceField.__name__
    DeviceField = wp.struct(DeviceField)
    DeviceField.__doc__ = DeviceField.cls.__doc__

    @wp.func(inline=True)
    def query(field: DeviceField, point: vector):
        if field.has_transform:
            t = field.transforms[1]
            linear = matrix(t[0, 0], t[0, 1], t[0, 2], t[1, 0], t[1, 1], t[1, 2], t[2, 0], t[2, 1], t[2, 2])
            translation = vector(t[0, 3], t[1, 3], t[2, 3])
            return reference_query(
                field.keys, field.coefficients, field.cumulative, point, linear, translation, field.depth
            )
        if field.reference:
            return reference_query(
                field.keys, field.coefficients, field.cumulative, point, field.linear, field.translation, field.depth
            )
        return direct_query(field.keys, field.coefficients, point, field.linear, field.translation, field.depth)

    reference_vjp = query_vjp_function(dtype)
    direct_vjp = query_vjp_function(dtype, reference=False)

    @wp.func_grad(query)
    def query_grad(field: DeviceField, point: vector, value_seed: dtype, gradient_seed: vector, valid_seed: int):
        if value_seed == dtype(0) and gradient_seed == vector(dtype(0)):
            return
        linear = field.linear
        translation = field.translation
        if field.has_transform:
            t = field.transforms[1]
            linear = matrix(t[0, 0], t[0, 1], t[0, 2], t[1, 0], t[1, 1], t[1, 2], t[2, 0], t[2, 1], t[2, 2])
            translation = vector(t[0, 3], t[1, 3], t[2, 3])
        p = linear * point + translation
        if not (
            wp.isfinite(p[0])
            and wp.isfinite(p[1])
            and wp.isfinite(p[2])
            and wp.min(p) >= dtype(0)
            and wp.max(p) <= dtype(1)
        ):
            return
        local_seed = linear * gradient_seed
        query_seed = vector(dtype(0))
        if field.reference:
            query_seed = reference_vjp(
                field.keys,
                field.coefficients,
                field.cumulative,
                p,
                field.depth,
                value_seed,
                local_seed,
                wp.adjoint[field.coefficients],
                wp.adjoint[field.cumulative],
            )
        else:
            query_seed = direct_vjp(
                field.keys,
                field.coefficients,
                field.cumulative,
                p,
                field.depth,
                value_seed,
                local_seed,
                wp.adjoint[field.coefficients],
                wp.adjoint[field.cumulative],
            )
        wp.adjoint[point] += wp.transpose(linear) * query_seed
        grad_transforms = wp.adjoint[field.transforms]
        if grad_transforms.shape[0]:
            # Only differentiable reconstruction supplies device transforms.
            identity = matrix(dtype(1), dtype(0), dtype(0), dtype(0), dtype(1), dtype(0), dtype(0), dtype(0), dtype(1))
            _value, spatial, _valid = reference_query(
                field.keys, field.coefficients, field.cumulative, p, identity, vector(dtype(0)), field.depth
            )
            gradient = transform(dtype(0))
            for a in range(3):
                for b in range(3):
                    gradient[a, b] = query_seed[a] * point[b] + spatial[a] * gradient_seed[b]
                gradient[a, 3] = query_seed[a]
            wp.atomic_add(grad_transforms, 1, gradient)

    return SimpleNamespace(field_type=DeviceField, query=query)


PoissonField = _precision_functions(wp.float32).field_type
PoissonField64 = _precision_functions(wp.float64).field_type
_query_float = _precision_functions(wp.float32).query
_query_double = _precision_functions(wp.float64).query


@wp.func(inline=True)
def poisson_evaluate(field: PoissonField, point: wp.vec3):
    """Evaluate a float32 field at one point in input coordinates.

    Args:
        field: Handle returned by ``result.device_field()``.
        point: Query position in the reconstruction's input coordinate system.

    Returns:
        The raw implicit value, its spatial gradient in input coordinates, and
        an integer validity flag. Invalid queries return NaN value/gradient and
        zero validity. The surface is ``value == result.iso_value``. First-order
        differentiation follows fixed lookup decisions; differentiating through
        reconstruction requires a result created with ``poisson_reconstruct(..., tape=tape)``.
    """
    return _query_float(field, point)


@wp.func(inline=True)
def poisson_evaluate(field: PoissonField64, point: wp.vec3d):
    """Evaluate a float64 field with the same contract as the float32 overload."""
    return _query_double(field, point)


wp._src.context.register_api_function(poisson_evaluate, module="warp.geometry", differentiable=True)


def _device_field(field, *, transforms=None):
    """Construct a handle sharing field arrays without device work."""
    dtype, device = field.coefficients.dtype, field.coefficients.device
    for name in ("keys", "coefficients", "cumulative"):
        array = getattr(field, name)
        if array is not None and (array.ndim != 1 or array.device != device):
            raise ValueError(f"{name} must be one-dimensional and share the field device")
    if field.cumulative is not None:
        if field.cumulative.dtype != dtype or field.cumulative.shape != field.coefficients.shape:
            raise ValueError("Cumulative coefficients must match coefficient precision and shape")
    transform_type = wp.types.matrix(shape=(4, 4), dtype=dtype)
    if transforms is not None:
        if (
            not wp.types.types_equal(transforms.dtype, transform_type)
            or transforms.shape != (2,)
            or transforms.device != device
        ):
            raise ValueError("Transforms must contain two matching-precision matrices on the field device")
        if field.cumulative is None:
            raise ValueError("Device transforms require a cumulative reconstruction field")
    handle = _precision_functions(dtype).field_type()
    handle.keys = field.keys
    handle.coefficients = field.coefficients
    handle.cumulative = field.cumulative if field.cumulative is not None else wp.empty(0, dtype=dtype, device=device)
    handle.linear = wp.types.matrix(shape=(3, 3), dtype=dtype)(*field.linear)
    handle.translation = wp.types.vector(length=3, dtype=dtype)(*field.translation)
    handle.depth = field.depth
    handle.reference = field.cumulative is not None
    handle.transforms = transforms if transforms is not None else wp.empty(0, dtype=transform_type, device=device)
    handle.has_transform = transforms is not None
    return handle
