# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for the texture mandelbrot sample scene."""

import unittest

import omni.kit
import omni.timeline
import omni.usd
import omni.warp

from ._common import (
    open_sample,
    validate_render,
)

TEST_ID = "texture_mandelbrot"


class TestSampleTextureMandelbrot(omni.kit.test.AsyncTestCase):
    async def _test_capture(self, enable_fsd: bool) -> None:
        await open_sample(f"{TEST_ID}.usda", enable_fsd=enable_fsd)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()

        fsd_str = "fsd_on" if enable_fsd else "fsd_off"
        await validate_render(f"{TEST_ID}_{fsd_str}")

    @unittest.skipIf(omni.kit.test.utils.is_etm_run(), "Regression in Kit")
    async def test_capture_fsd_off(self) -> None:
        await self._test_capture(enable_fsd=False)

    @unittest.skipIf(omni.kit.test.utils.is_etm_run(), "Regression in Kit")
    async def test_capture_fsd_on(self) -> None:
        await self._test_capture(enable_fsd=True)
