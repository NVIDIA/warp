# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for the prim flocking sample scene."""

import numpy as np
import omni.kit
import omni.timeline
import omni.usd
import omni.warp
import usdrt

import warp as wp

from ._common import (
    array_are_almost_equal,
    open_sample,
)

TEST_ID = "prim_flocking"
FRAME_COUNT = 30


class TestSamplePrimFlocking(omni.kit.test.AsyncTestCase):
    async def _test_eval(self, enable_fsd: bool) -> None:
        await open_sample(f"{TEST_ID}.usda", enable_fsd=enable_fsd)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        stage_id = omni.usd.get_context().get_stage_id()
        stage = usdrt.Usd.Stage.Attach(stage_id)
        boid_prims = stage.SelectPrims(
            require_applied_schemas=("BoidTag",),
            require_attrs=((usdrt.Sdf.ValueTypeNames.Double3, "_worldPosition", usdrt.Usd.Access.ReadWrite),),
            device="cuda:0",
        )

        prev_points_hash = None
        curr_points_hash = None

        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()

            points = wp.fabricarray(data=boid_prims, attrib="_worldPosition").numpy()
            assert np.isfinite(points).all()
            array_are_almost_equal(np.min(points, axis=0), (-125.0, 10.0, -100.0), atol=50.0)
            array_are_almost_equal(np.max(points, axis=0), (125.0, 100.0, 100.0), atol=50.0)

            curr_points_hash = hash(points.tobytes())
            assert curr_points_hash != prev_points_hash
            prev_points_hash = curr_points_hash

    async def test_eval_fsd_on(self) -> None:
        await self._test_eval(enable_fsd=True)
