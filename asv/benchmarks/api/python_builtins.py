# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

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
