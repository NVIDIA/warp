# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for the cloth simulation sample scene."""

import unittest

import numpy as np
import omni.graph.core as og
import omni.kit
import omni.timeline
import omni.usd
import omni.warp

from ._common import (
    array_are_almost_equal,
    attr_disconnect_all,
    open_sample,
    validate_render,
)

TEST_ID = "cloth_simulation"


class TestSampleClothSimulation(omni.kit.test.AsyncTestCase):
    async def _test_eval(self, enable_fsd: bool) -> None:
        await open_sample(f"{TEST_ID}.usda", enable_fsd=enable_fsd)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        graph = og.Controller.graph("/World/ActionGraph")

        # Force writing to USD so that we can read resulting values from the stage.
        write_node = og.Controller.node("write_prims", graph)
        write_usd_attr = og.Controller.attribute("inputs:usdWriteBack", write_node)
        attr_disconnect_all(write_usd_attr)
        write_usd_attr.set(True)

        stage = omni.usd.get_context().get_stage()
        cloth_prim = stage.GetPrimAtPath("/World/Cloth")
        points_attr = cloth_prim.GetAttribute("points")

        points_first = np.array(points_attr.Get())
        assert np.isfinite(points_first).all()
        array_are_almost_equal(np.min(points_first, axis=0), (-50.0, 0.0, -50.0))
        array_are_almost_equal(np.max(points_first, axis=0), (50.0, 0.0, 50.0))

        prev_points_hash = None
        curr_points_hash = None

        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

            points = np.array(points_attr.Get())
            assert np.isfinite(points).all()

            curr_points_hash = hash(points.tobytes())
            assert curr_points_hash != prev_points_hash
            prev_points_hash = curr_points_hash

        points_last = np.array(points_attr.Get())
        assert np.mean(points_last, axis=0)[1] < np.mean(points_first, axis=0)[1]

    async def test_eval_fsd_off(self) -> None:
        await self._test_eval(enable_fsd=False)

    async def test_eval_fsd_on(self) -> None:
        await self._test_eval(enable_fsd=True)

    async def _test_capture(self, enable_fsd: bool) -> None:
        await open_sample(f"{TEST_ID}.usda", enable_fsd=enable_fsd)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        fsd_str = "fsd_on" if enable_fsd else "fsd_off"
        await validate_render(f"{TEST_ID}_{fsd_str}")

    @unittest.skipIf(omni.kit.test.utils.is_etm_run(), "Regression in Kit")
    async def test_capture_fsd_off(self) -> None:
        await self._test_capture(enable_fsd=False)

    @unittest.skipIf(omni.kit.test.utils.is_etm_run(), "Regression in Kit")
    async def test_capture_fsd_on(self) -> None:
        await self._test_capture(enable_fsd=True)
