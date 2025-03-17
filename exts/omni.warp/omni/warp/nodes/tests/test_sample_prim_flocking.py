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

"""Tests for the prim flocking sample scene."""

import numpy as np
import omni.kit
import omni.usd
import omni.warp
import usdrt

import warp as wp

from ._common import (
    FrameRange,
    array_are_almost_equal,
    open_sample,
)

TEST_ID = "prim_flocking"
FRAME_COUNT = 30


class TestSamplePrimFlocking(omni.kit.test.AsyncTestCase):
    async def _test_eval(self, enable_fsd: bool) -> None:
        await open_sample(f"{TEST_ID}.usda", enable_fsd=enable_fsd)

        stage_id = omni.usd.get_context().get_stage_id()
        stage = usdrt.Usd.Stage.Attach(stage_id)
        boid_prims = stage.SelectPrims(
            require_applied_schemas=("BoidTag",),
            require_attrs=((usdrt.Sdf.ValueTypeNames.Double3, "_worldPosition", usdrt.Usd.Access.ReadWrite),),
            device="cuda:0",
        )

        prev_points_hash = None
        curr_points_hash = None

        with FrameRange(30) as frames:
            async for _ in frames:
                points = wp.fabricarray(data=boid_prims, attrib="_worldPosition").numpy()
                assert np.isfinite(points).all()
                array_are_almost_equal(np.min(points, axis=0), (-125.0, 10.0, -100.0), atol=50.0)
                array_are_almost_equal(np.max(points, axis=0), (125.0, 100.0, 100.0), atol=50.0)

                curr_points_hash = hash(points.tobytes())
                assert curr_points_hash != prev_points_hash
                prev_points_hash = curr_points_hash

    async def test_eval_fsd_on(self) -> None:
        await self._test_eval(enable_fsd=True)
