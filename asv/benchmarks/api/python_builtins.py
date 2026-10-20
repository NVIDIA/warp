# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from warp._src.context import get_builtin_call_desc
from warp._src.types import _binary_builtin_cache, _rbinary_builtin_cache, _unary_builtin_cache

mat = wp.mat44(*range(16))
xform = wp.transform(*range(7))


class PythonBuiltins:
    repeat = 1000  # Number of samples to run
    number = 1  # Number of timed calls per sample

    def setup(self):
        wp.init()

    def time_call_builtin_mul_op(self):
        mat * mat

    def time_call_builtin_transform_identity_fn(self):
        wp.transform_identity()

    def time_call_builtin_synthetic_workload(self):
        value = wp.min(wp.max(1.0, 2.0), 3.0)
        point = wp.vec3(*range(3))

        mat1 = wp.transpose(mat)
        mat2 = mat * mat * value

        xform1 = wp.transform_identity()
        xform2 = wp.transform_inverse(xform1)
        xform3 = xform * xform * value

        dot = wp.ddot(mat1, mat2)
        point = wp.transform_point(wp.inverse(mat1) * mat2, point * dot)
        pos = wp.transform_get_translation(xform2)
        rot = wp.transform_get_rotation(xform3)
        coeff = wp.exp(-wp.length(point))
        rot *= wp.quat_from_axis_angle(
            wp.vec3(1.0, 0.0, 0.0),
            wp.radians(coeff * value),
        )
        point = wp.quat_rotate(rot, point + pos)


class PythonBuiltinsDescriptorCold:
    """Measure built-in latency when call descriptors are not cached.

    ASV warmup populates the descriptor caches before ordinary timing samples,
    so the warm-path benchmarks above cannot detect regressions in descriptor
    construction. Clearing the caches in ``setup()`` leaves cache invalidation
    outside the timed region while making every measured call exercise the cold
    path. Operators keep their own descriptor caches in front of
    ``get_builtin_call_desc()``, so those are cleared too.
    """

    repeat = 64  # Calibrated sample count for detecting 5% changes.
    number = 1  # Keep each cache-cold call as an independent raw sample.

    def setup(self):
        wp.init()
        get_builtin_call_desc.cache_clear()
        _unary_builtin_cache.clear()
        _binary_builtin_cache.clear()
        _rbinary_builtin_cache.clear()

    def time_call_builtin_transform_identity_fn(self):
        wp.transform_identity()

    def time_call_builtin_mul_op(self):
        # Unlike the single transform_identity() overload, matrix multiplication
        # rejects many wp.mul overloads before matching, so it stresses the
        # per-overload cost of building descriptors.
        mat * mat

    time_call_builtin_mul_op.repeat = 32  # Calibrated separately for detecting 5% changes.
