# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for the particle simulation drop sample scene."""

import numpy as np
import omni.graph.core as og
import omni.kit
import omni.timeline
import omni.usd
import omni.warp
from omni.warp.nodes.tests._common import (
    open_sample,
    validate_render,
)

TEST_ID = "particles_simulation_drop"


class TestSampleParticlesSimulationDrop(omni.kit.test.AsyncTestCase):
    async def _test_capture(self, enable_fsd: bool) -> None:
        await open_sample(f"{TEST_ID}.usda", enable_fsd=enable_fsd)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        fsd_str = "fsd_on" if enable_fsd else "fsd_off"
        await validate_render(f"{TEST_ID}_{fsd_str}")

    async def test_capture_fsd_off(self) -> None:
        await self._test_capture(enable_fsd=False)

    async def test_capture_fsd_on(self) -> None:
        await self._test_capture(enable_fsd=True)
