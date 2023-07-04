# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Helpers for Pixar's OpenUSD."""

from typing import Optional

import numpy as np
import omni.usd
from pxr import (
    Usd,
    UsdGeom,
)


def prim_get_world_xform(prim_path: Optional[str]) -> np.ndarray:
    """Retrieves the world transformation matrix from a USD primitive."""
    if prim_path is not None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.IsA(UsdGeom.Xformable):
            prim = UsdGeom.Xformable(prim)
            xform = prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return np.array(xform)

    return np.identity(4)
