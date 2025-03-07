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

import math
import unittest

import warp as wp
from warp.tests.unittest_utils import *

kernel_cache = {}


def getkernel(func, suffix=""):
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key)
    return kernel_cache[key]


def test_infinity_scalar(test, device, dtype, register_kernels=False):
    def check_infinity(outputs: wp.array(dtype=dtype), bool_outputs: wp.array(dtype=wp.bool)):
        outputs[0] = dtype(wp.inf)
        outputs[1] = dtype(-wp.inf)
        outputs[2] = dtype(2.0 * wp.inf)
        outputs[3] = dtype(-2.0 * wp.inf)
        outputs[4] = dtype(2.0 / 0.0)
        outputs[5] = dtype(-2.0 / 0.0)
        outputs[6] = wp.log(dtype(0))
        outputs[7] = wp.exp(dtype(800))

        # Fill out bool outputs
        bool_outputs[0] = wp.isinf(dtype(wp.inf))
        bool_outputs[1] = wp.isfinite(dtype(wp.inf))
        bool_outputs[2] = wp.isinf(dtype(-wp.inf))
        bool_outputs[3] = wp.isfinite(dtype(-wp.inf))
        bool_outputs[4] = wp.isinf(dtype(0))
        bool_outputs[5] = wp.isinf(wp.exp(dtype(800)))

    kernel = getkernel(check_infinity, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs = wp.empty(8, dtype=dtype, device=device)
    outputs_bool = wp.empty(6, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[], outputs=[outputs, outputs_bool], device=device)

    outputs_cpu = outputs.to("cpu").list()

    test.assertEqual(outputs_cpu[0], math.inf)
    test.assertEqual(outputs_cpu[1], -math.inf)
    test.assertEqual(outputs_cpu[2], math.inf)
    test.assertEqual(outputs_cpu[3], -math.inf)
    test.assertEqual(outputs_cpu[4], math.inf)
    test.assertEqual(outputs_cpu[5], -math.inf)
    test.assertEqual(outputs_cpu[6], -math.inf)
    test.assertEqual(outputs_cpu[7], math.inf)

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertTrue(outputs_bool_cpu[0], "wp.isinf(wp.inf) is not True")
    test.assertFalse(outputs_bool_cpu[1], "wp.isfinite(wp.inf) is not False")
    test.assertTrue(outputs_bool_cpu[2], "wp.isinf(-wp.inf) is not True")
    test.assertFalse(outputs_bool_cpu[3], "wp.isfinite(-wp.inf) is not False")
    test.assertFalse(outputs_bool_cpu[4], "wp.isinf(0) is not False")
    test.assertTrue(outputs_bool_cpu[5], "wp.isinf(wp.exp(800)) is not True")


def test_nan_scalar(test, device, dtype, register_kernels=False):
    def check_nan(outputs: wp.array(dtype=dtype), bool_outputs: wp.array(dtype=wp.bool)):
        outputs[0] = dtype(wp.nan)
        outputs[1] = dtype(-wp.nan)
        outputs[2] = dtype(2.0 * wp.nan)
        outputs[3] = dtype(2.0 + wp.nan)
        outputs[4] = dtype(0.0 / 0.0)
        outputs[5] = wp.sqrt(dtype(-1))
        outputs[6] = wp.log(dtype(-1))
        outputs[7] = dtype(wp.inf) - dtype(wp.inf)

        # Fill out bool outputs
        bool_outputs[0] = dtype(wp.nan) == dtype(wp.nan)
        bool_outputs[1] = dtype(wp.nan) != dtype(wp.nan)
        bool_outputs[2] = dtype(wp.nan) == dtype(1)
        bool_outputs[3] = dtype(wp.nan) != dtype(1)
        bool_outputs[4] = wp.isnan(wp.nan)
        bool_outputs[5] = wp.isnan(dtype(0.0))
        bool_outputs[6] = wp.isnan(dtype(wp.inf))
        bool_outputs[7] = dtype(wp.nan) > dtype(1)
        bool_outputs[8] = dtype(wp.nan) >= dtype(1)
        bool_outputs[9] = dtype(wp.nan) < dtype(1)
        bool_outputs[10] = dtype(wp.nan) <= dtype(1)
        bool_outputs[11] = dtype(wp.nan) > dtype(wp.nan)
        bool_outputs[12] = dtype(wp.nan) >= dtype(wp.nan)
        bool_outputs[13] = dtype(wp.nan) < dtype(wp.nan)
        bool_outputs[14] = dtype(wp.nan) <= dtype(wp.nan)
        bool_outputs[15] = wp.isfinite(dtype(wp.nan))
        bool_outputs[16] = wp.isinf(dtype(wp.nan))

    kernel = getkernel(check_nan, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs = wp.empty(8, dtype=dtype, device=device)
    outputs_bool = wp.empty(17, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[], outputs=[outputs, outputs_bool], device=device)

    outputs_cpu = outputs.to("cpu").list()

    test.assertTrue(math.isnan(outputs_cpu[0]), "wp.nan is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[1]), "-wp.nan is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[2]), "2.0*wp.nan is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[3]), "2.0+wp.nan is not NaN ")
    test.assertTrue(math.isnan(outputs_cpu[4]), "0.0/0.0 is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[5]), "Sqrt of a negative number is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[6]), "Log of a negative number is not NaN")
    test.assertTrue(math.isnan(outputs_cpu[7]), "Subtracting infinity from infinity is not NaN")

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertFalse(outputs_bool_cpu[0], "wp.nan == wp.nan is not False")
    test.assertTrue(outputs_bool_cpu[1], "wp.nan != wp.nan is not True")
    test.assertFalse(outputs_bool_cpu[2], "wp.nan == 1 is not False")
    test.assertTrue(outputs_bool_cpu[3], "wp.nan != 1 is not True")
    test.assertTrue(outputs_bool_cpu[4], "isnan(wp.nan) is not True")
    test.assertFalse(outputs_bool_cpu[5], "isnan(0.0) is not False")
    test.assertFalse(outputs_bool_cpu[6], "isnan(wp.inf) is not False")
    test.assertFalse(outputs_bool_cpu[7], "wp.nan > 1 is not False")
    test.assertFalse(outputs_bool_cpu[8], "wp.nan >= 1 is not False")
    test.assertFalse(outputs_bool_cpu[9], "wp.nan < 1 is not False")
    test.assertFalse(outputs_bool_cpu[10], "wp.nan <= 1 is not False")
    test.assertFalse(outputs_bool_cpu[11], "wp.nan > wp.nan is not False")
    test.assertFalse(outputs_bool_cpu[12], "wp.nan >= wp.nan is not False")
    test.assertFalse(outputs_bool_cpu[13], "wp.nan < wp.nan is not False")
    test.assertFalse(outputs_bool_cpu[14], "wp.nan <= wp.nan is not False")
    test.assertFalse(outputs_bool_cpu[15], "wp.isfinite(wp.nan) is not False")
    test.assertFalse(outputs_bool_cpu[16], "wp.isinf(wp.nan) is not False")


def test_is_special_vec(test, device, dtype, register_kernels=False):
    vector_type = wp.types.vector(5, dtype)

    def check_special_vec(bool_outputs: wp.array(dtype=wp.bool)):
        zeros_vector = vector_type()
        bool_outputs[0] = wp.isfinite(zeros_vector)
        bool_outputs[1] = wp.isinf(zeros_vector)
        bool_outputs[2] = wp.isnan(zeros_vector)

        nan_vector = vector_type()
        nan_vector[0] = dtype(wp.NAN)
        bool_outputs[3] = wp.isfinite(nan_vector)
        bool_outputs[4] = wp.isinf(nan_vector)
        bool_outputs[5] = wp.isnan(nan_vector)

        inf_vector = vector_type()
        inf_vector[0] = dtype(wp.inf)
        bool_outputs[6] = wp.isfinite(inf_vector)
        bool_outputs[7] = wp.isinf(inf_vector)
        bool_outputs[8] = wp.isnan(inf_vector)

    kernel = getkernel(check_special_vec, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs_bool = wp.empty(9, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[outputs_bool], device=device)

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertTrue(outputs_bool_cpu[0], "wp.isfinite(zeros_vector) is not True")
    test.assertFalse(outputs_bool_cpu[1], "wp.isinf(zeros_vector) is not False")
    test.assertFalse(outputs_bool_cpu[2], "wp.isnan(zeros_vector) is not False")

    test.assertFalse(outputs_bool_cpu[3], "wp.isfinite(nan_vector) is not False")
    test.assertFalse(outputs_bool_cpu[4], "wp.isinf(nan_vector) is not False")
    test.assertTrue(outputs_bool_cpu[5], "wp.isnan(nan_vector) is not True")

    test.assertFalse(outputs_bool_cpu[6], "wp.isfinite(inf_vector) is not False")
    test.assertTrue(outputs_bool_cpu[7], "wp.isinf(inf_vector) is not True")
    test.assertFalse(outputs_bool_cpu[8], "wp.isnan(inf_vector) is not False")


def test_is_special_mat(test, device, dtype, register_kernels=False):
    mat_type = wp.types.matrix((5, 5), dtype)

    def check_special_mat(bool_outputs: wp.array(dtype=wp.bool)):
        zeros_mat = mat_type()
        bool_outputs[0] = wp.isfinite(zeros_mat)
        bool_outputs[1] = wp.isinf(zeros_mat)
        bool_outputs[2] = wp.isnan(zeros_mat)

        nan_mat = mat_type()
        nan_mat[0, 0] = dtype(wp.NAN)
        bool_outputs[3] = wp.isfinite(nan_mat)
        bool_outputs[4] = wp.isinf(nan_mat)
        bool_outputs[5] = wp.isnan(nan_mat)

        inf_mat = mat_type()
        inf_mat[0, 0] = dtype(wp.inf)
        bool_outputs[6] = wp.isfinite(inf_mat)
        bool_outputs[7] = wp.isinf(inf_mat)
        bool_outputs[8] = wp.isnan(inf_mat)

    kernel = getkernel(check_special_mat, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs_bool = wp.empty(9, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[outputs_bool], device=device)

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertTrue(outputs_bool_cpu[0], "wp.isfinite(zeros_mat) is not True")
    test.assertFalse(outputs_bool_cpu[1], "wp.isinf(zeros_mat) is not False")
    test.assertFalse(outputs_bool_cpu[2], "wp.isnan(zeros_mat) is not False")

    test.assertFalse(outputs_bool_cpu[3], "wp.isfinite(nan_mat) is not False")
    test.assertFalse(outputs_bool_cpu[4], "wp.isinf(nan_mat) is not False")
    test.assertTrue(outputs_bool_cpu[5], "wp.isnan(nan_mat) is not True")

    test.assertFalse(outputs_bool_cpu[6], "wp.isfinite(inf_mat) is not False")
    test.assertTrue(outputs_bool_cpu[7], "wp.isinf(inf_mat) is not True")
    test.assertFalse(outputs_bool_cpu[8], "wp.isnan(inf_mat) is not False")


def test_is_special_quat(test, device, dtype, register_kernels=False):
    quat_type = wp.types.quaternion(dtype)

    def check_special_quat(bool_outputs: wp.array(dtype=wp.bool)):
        zeros_quat = quat_type()
        bool_outputs[0] = wp.isfinite(zeros_quat)
        bool_outputs[1] = wp.isinf(zeros_quat)
        bool_outputs[2] = wp.isnan(zeros_quat)

        nan_quat = quat_type(dtype(wp.NAN), dtype(0), dtype(0), dtype(0))
        bool_outputs[3] = wp.isfinite(nan_quat)
        bool_outputs[4] = wp.isinf(nan_quat)
        bool_outputs[5] = wp.isnan(nan_quat)

        inf_quat = quat_type(dtype(wp.INF), dtype(0), dtype(0), dtype(0))
        bool_outputs[6] = wp.isfinite(inf_quat)
        bool_outputs[7] = wp.isinf(inf_quat)
        bool_outputs[8] = wp.isnan(inf_quat)

    kernel = getkernel(check_special_quat, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs_bool = wp.empty(9, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[outputs_bool], device=device)

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertTrue(outputs_bool_cpu[0], "wp.isfinite(zeros_quat) is not True")
    test.assertFalse(outputs_bool_cpu[1], "wp.isinf(zeros_quat) is not False")
    test.assertFalse(outputs_bool_cpu[2], "wp.isnan(zeros_quat) is not False")

    test.assertFalse(outputs_bool_cpu[3], "wp.isfinite(nan_quat) is not False")
    test.assertFalse(outputs_bool_cpu[4], "wp.isinf(nan_quat) is not False")
    test.assertTrue(outputs_bool_cpu[5], "wp.isnan(nan_quat) is not True")

    test.assertFalse(outputs_bool_cpu[6], "wp.isfinite(inf_quat) is not False")
    test.assertTrue(outputs_bool_cpu[7], "wp.isinf(inf_quat) is not True")
    test.assertFalse(outputs_bool_cpu[8], "wp.isnan(inf_quat) is not False")


def test_is_special_int(test, device, dtype, register_kernels=False):
    vector_type = wp.types.vector(5, dtype)
    matrix_type = wp.types.matrix((5, 5), dtype)

    def check_is_special_int(bool_outputs: wp.array(dtype=wp.bool)):
        bool_outputs[0] = wp.isfinite(dtype(0))
        bool_outputs[1] = wp.isnan(dtype(0))
        bool_outputs[2] = wp.isinf(dtype(0))

        bool_outputs[3] = wp.isfinite(vector_type())
        bool_outputs[4] = wp.isnan(vector_type())
        bool_outputs[5] = wp.isinf(vector_type())

        bool_outputs[6] = wp.isfinite(matrix_type())
        bool_outputs[7] = wp.isnan(matrix_type())
        bool_outputs[8] = wp.isinf(matrix_type())

    kernel = getkernel(check_is_special_int, suffix=dtype.__name__)

    if register_kernels:
        return

    outputs_bool = wp.empty(9, dtype=wp.bool, device=device)

    wp.launch(kernel, dim=1, inputs=[outputs_bool], device=device)

    outputs_bool_cpu = outputs_bool.to("cpu").list()

    test.assertTrue(outputs_bool_cpu[0], "wp.isfinite(0) is not True")
    test.assertFalse(outputs_bool_cpu[1], "wp.isinf(0) is not False")
    test.assertFalse(outputs_bool_cpu[2], "wp.isnan(0) is not False")

    test.assertTrue(outputs_bool_cpu[3], "wp.isfinite(vec) is not True")
    test.assertFalse(outputs_bool_cpu[4], "wp.isinf(vec) is not False")
    test.assertFalse(outputs_bool_cpu[5], "wp.isnan(vec) is not False")

    test.assertTrue(outputs_bool_cpu[6], "wp.isfinite(matrix) is not True")
    test.assertFalse(outputs_bool_cpu[7], "wp.isinf(matrix) is not False")
    test.assertFalse(outputs_bool_cpu[8], "wp.isnan(matrix) is not False")


devices = get_test_devices()


class TestSpecialValues(unittest.TestCase):
    pass


for dtype in [wp.float16, wp.float32, wp.float64]:
    add_function_test_register_kernel(
        TestSpecialValues, f"test_infinity_{dtype.__name__}", test_infinity_scalar, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpecialValues, f"test_nan_{dtype.__name__}", test_nan_scalar, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpecialValues, f"test_is_special_vec_{dtype.__name__}", test_is_special_vec, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpecialValues, f"test_is_special_mat_{dtype.__name__}", test_is_special_mat, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpecialValues, f"test_is_special_quat_{dtype.__name__}", test_is_special_quat, devices=devices, dtype=dtype
    )

# Ensure functions like wp.isfinite work on integer types
for dtype in wp.types.int_types:
    add_function_test_register_kernel(
        TestSpecialValues, f"test_is_special_int_{dtype.__name__}", test_is_special_int, devices=devices, dtype=dtype
    )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
