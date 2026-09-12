# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static type-checking fixture for the generated stubs.

This kernel is type-checked by pyright and mypy in CI but never launched. It
pins that registered defaults are optional at the call site, that supported
Python literals match Warp's runtime behavior, and that omitted type arguments
produce the same result types as code generation.
"""

from typing import Literal, assert_type, cast

import warp as wp


@wp.kernel
def check_builtin_defaults(mesh_id: wp.uint64, p: wp.vec3f, x: wp.float32):
    # Built-ins whose trailing parameters have registered defaults.
    wp.mesh_query_point_sign_winding_number(mesh_id, p, 1.0e6)
    wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6)

    # The same call passing every documented default explicitly, as Python floats.
    wp.mesh_query_point_sign_winding_number(mesh_id, p, 1.0e6, 2.0, 0.5)

    # Value constructors cast Python literals to their concrete scalar type.
    v = wp.vec3f(1.0, 2.0, 3.0)
    wp.mat33f(0.0)

    # A uint32 parameter requires an explicit Warp value.
    wp.curlnoise(wp.uint32(42), v)
    wp.curlnoise(wp.uint32(42), v, wp.uint32(2))

    # A built-in whose trailing ``tolerance`` default is omitted, called both
    # with Warp scalars and with Python literals.
    wp.expect_near(x, x)
    wp.expect_near(1.0, 1.0)
    wp.expect_near(x, x, 1.0e-3)

    # A `dtype` argument names a type: omitting it yields the documented result,
    # and passing one parameterizes that result.
    assert_type(wp.tile_arange(4), wp.Tile[wp.float32, tuple[int]])
    assert_type(wp.tile_arange(4, dtype=wp.uint32), wp.Tile[wp.uint32, tuple[int]])
    assert_type(wp.quat_identity(), wp.quatf)
    assert_type(wp.quat_identity(dtype=wp.float64), wp.Quaternion[wp.float64])
    assert_type(wp.transform_identity(dtype=wp.float64), wp.Transformation[wp.float64])
    assert_type(wp.quaternion(dtype=wp.float64), wp.Quaternion[wp.float64])

    # A defaulted argument before dtype must not make dtype keyword-only.
    p64 = cast(wp.Vector[wp.float64, Literal[3]], wp.vec3d())
    q64 = cast(wp.Quaternion[wp.float64], wp.quatd())
    assert_type(wp.transformation(p64, q64, wp.float64), wp.Transformation[wp.float64])
    assert_type(wp.transformation(p64, q64, dtype=wp.float64), wp.Transformation[wp.float64])
    assert_type(wp.transformation(p64, dtype=wp.float64), wp.Transformation[wp.float64])

    # The same argument is accepted wherever a built-in declares one.
    wp.vector(1.0, 2.0, length=2, dtype=wp.float64)
    wp.identity(n=3, dtype=wp.float64)
