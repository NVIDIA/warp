# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the API that does type conversions."""

from typing import Any

import omni.graph.core as og
import omni.kit
import omni.warp

import warp as wp


def are_array_annotations_equal(a: Any, b: Any) -> bool:
    return isinstance(a, wp.array) and isinstance(b, wp.array) and a.dtype == b.dtype and a.ndim == b.ndim


class TestApiTypeConversions(omni.kit.test.AsyncTestCase):
    async def test_og_to_warp_conversion(self):
        og_type = og.Type(
            og.BaseDataType.BOOL,
            tuple_count=1,
            array_depth=0,
            role=og.AttributeRole.NONE,
        )

        wp_type = omni.warp.nodes.type_convert_og_to_warp(og_type)
        expected = wp.int8
        self.assertEqual(wp_type, expected)

        wp_type = omni.warp.nodes.type_convert_og_to_warp(og_type, dim_count=1)
        expected = wp.array(dtype=wp.int8)
        self.assertTrue(are_array_annotations_equal(wp_type, expected))

        # ----------------------------------------------------------------------

        og_type = og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=3,
            array_depth=1,
            role=og.AttributeRole.COLOR,
        )

        wp_type = omni.warp.nodes.type_convert_og_to_warp(og_type)
        expected = wp.array(dtype=wp.vec3)
        self.assertTrue(are_array_annotations_equal(wp_type, expected))

        wp_type = omni.warp.nodes.type_convert_og_to_warp(og_type, dim_count=0)
        expected = wp.vec3
        self.assertEqual(wp_type, expected)

        # ----------------------------------------------------------------------

        og_type = og.Type(
            og.BaseDataType.DOUBLE,
            tuple_count=9,
            array_depth=0,
            role=og.AttributeRole.MATRIX,
        )

        wp_type = omni.warp.nodes.type_convert_og_to_warp(og_type)
        expected = wp.mat33d
        self.assertEqual(wp_type, expected)

        wp_type = omni.warp.nodes.type_convert_og_to_warp(og_type, dim_count=1)
        expected = wp.array(dtype=wp.mat33d)
        self.assertTrue(are_array_annotations_equal(wp_type, expected))

        # ----------------------------------------------------------------------

        og_type = og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=4,
            array_depth=1,
            role=og.AttributeRole.QUATERNION,
        )

        wp_type = omni.warp.nodes.type_convert_og_to_warp(og_type)
        expected = wp.array(dtype=wp.quat)
        self.assertTrue(are_array_annotations_equal(wp_type, expected))

        wp_type = omni.warp.nodes.type_convert_og_to_warp(og_type, dim_count=0)
        expected = wp.quat
        self.assertEqual(wp_type, expected)

    async def test_sdf_name_to_warp_conversion(self):
        sdf_name = "color4f"

        wp_type = omni.warp.nodes.type_convert_sdf_name_to_warp(sdf_name)
        expected = wp.vec4
        self.assertEqual(wp_type, expected)

        wp_type = omni.warp.nodes.type_convert_sdf_name_to_warp(sdf_name, dim_count=1)
        expected = wp.array(dtype=wp.vec4)
        self.assertTrue(are_array_annotations_equal(wp_type, expected))

        # ----------------------------------------------------------------------

        sdf_name = "point3f[]"

        wp_type = omni.warp.nodes.type_convert_sdf_name_to_warp(sdf_name)
        expected = wp.array(dtype=wp.vec3)
        self.assertTrue(are_array_annotations_equal(wp_type, expected))

        wp_type = omni.warp.nodes.type_convert_sdf_name_to_warp(sdf_name, dim_count=0)
        expected = wp.vec3
        self.assertEqual(wp_type, expected)

        # ----------------------------------------------------------------------

        sdf_name = "timecode"

        wp_type = omni.warp.nodes.type_convert_sdf_name_to_warp(sdf_name)
        expected = wp.float64
        self.assertEqual(wp_type, expected)

        wp_type = omni.warp.nodes.type_convert_sdf_name_to_warp(sdf_name, dim_count=1)
        expected = wp.array(dtype=wp.float64)
        self.assertTrue(are_array_annotations_equal(wp_type, expected))

        # ----------------------------------------------------------------------

        sdf_name = "token[]"

        wp_type = omni.warp.nodes.type_convert_sdf_name_to_warp(sdf_name)
        expected = wp.array(dtype=wp.uint64)
        self.assertTrue(are_array_annotations_equal(wp_type, expected))

        wp_type = omni.warp.nodes.type_convert_sdf_name_to_warp(sdf_name, dim_count=0)
        expected = wp.uint64
        self.assertEqual(wp_type, expected)

    async def test_sdf_name_to_og_conversion(self):
        sdf_name = "float2"

        og_type = omni.warp.nodes.type_convert_sdf_name_to_og(sdf_name)
        expected = og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=2,
            array_depth=0,
            role=og.AttributeRole.NONE,
        )
        self.assertEqual(og_type, expected)

        og_type = omni.warp.nodes.type_convert_sdf_name_to_og(sdf_name, is_array=True)
        expected = og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=2,
            array_depth=1,
            role=og.AttributeRole.NONE,
        )
        self.assertEqual(og_type, expected)

        # ----------------------------------------------------------------------

        sdf_name = "matrix3d[]"

        og_type = omni.warp.nodes.type_convert_sdf_name_to_og(sdf_name)
        expected = og.Type(
            og.BaseDataType.DOUBLE,
            tuple_count=9,
            array_depth=1,
            role=og.AttributeRole.MATRIX,
        )
        self.assertEqual(og_type, expected)

        og_type = omni.warp.nodes.type_convert_sdf_name_to_og(sdf_name, is_array=False)
        expected = og.Type(
            og.BaseDataType.DOUBLE,
            tuple_count=9,
            array_depth=0,
            role=og.AttributeRole.MATRIX,
        )
        self.assertEqual(og_type, expected)

        # ----------------------------------------------------------------------

        sdf_name = "texCoord2f"

        og_type = omni.warp.nodes.type_convert_sdf_name_to_og(sdf_name)
        expected = og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=2,
            array_depth=0,
            role=og.AttributeRole.TEXCOORD,
        )
        self.assertEqual(og_type, expected)

        og_type = omni.warp.nodes.type_convert_sdf_name_to_og(sdf_name, is_array=True)
        expected = og.Type(
            og.BaseDataType.FLOAT,
            tuple_count=2,
            array_depth=1,
            role=og.AttributeRole.TEXCOORD,
        )
        self.assertEqual(og_type, expected)

        # ----------------------------------------------------------------------

        sdf_name = "uchar[]"

        og_type = omni.warp.nodes.type_convert_sdf_name_to_og(sdf_name)
        expected = og.Type(
            og.BaseDataType.UCHAR,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        )
        self.assertEqual(og_type, expected)

        og_type = omni.warp.nodes.type_convert_sdf_name_to_og(sdf_name, is_array=False)
        expected = og.Type(
            og.BaseDataType.UCHAR,
            tuple_count=1,
            array_depth=0,
            role=og.AttributeRole.NONE,
        )
        self.assertEqual(og_type, expected)
