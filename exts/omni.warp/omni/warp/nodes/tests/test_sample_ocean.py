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

"""Tests for the ocean sample scene."""

import numpy as np
import omni.graph.core as og
import omni.kit
import omni.usd
import omni.warp

from ._common import (
    FrameRange,
    array_are_almost_equal,
    attr_disconnect_all,
    open_sample,
    validate_render,
)

TEST_ID = "ocean"


class TestSampleOcean(omni.kit.test.AsyncTestCase):
    async def test_eval(self) -> None:
        await open_sample(f"{TEST_ID}.usda")

        graph = og.Controller.graph("/World/ActionGraph")

        # Force writing to USD so that we can read resulting values from the stage.
        write_node = og.Controller.node("write_prims", graph)
        write_usd_attr = og.Controller.attribute("inputs:usdWriteBack", write_node)
        attr_disconnect_all(write_usd_attr)
        write_usd_attr.set(True)

        stage = omni.usd.get_context().get_stage()
        mesh_prim = stage.GetPrimAtPath("/World/MeshOut/Plane")
        points_attr = mesh_prim.GetAttribute("points")

        prev_points_hash = None
        curr_points_hash = None

        with FrameRange(30) as frames:
            async for _ in frames:
                points = np.array(points_attr.Get())
                assert np.isfinite(points).all()
                array_are_almost_equal(np.min(points, axis=0), (-8196.0, 0.0, -8707.0), atol=20.0)
                array_are_almost_equal(np.max(points, axis=0), (8197.0, 0.0, 7683.0), atol=20.0)

                curr_points_hash = hash(points.tobytes())
                assert curr_points_hash != prev_points_hash
                prev_points_hash = curr_points_hash

    async def test_capture(self) -> None:
        await open_sample(f"{TEST_ID}.usda")

        with FrameRange(30) as frames:
            async for _ in frames:
                pass

        await validate_render(TEST_ID)
