# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Node writing a dynamic texture."""

import ctypes
import traceback

import omni.graph.core as og
import omni.ui as ui
import warp as wp

from omni.warp.nodes.ogn.OgnTextureWriteDatabase import OgnTextureWriteDatabase


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnTextureWriteDatabase) -> None:
    """Evaluates the node."""
    if not db.inputs.data.memory or db.inputs.data.shape[0] == 0:
        return

    uri = db.inputs.uri
    if not uri.startswith("dynamic://"):
        return

    dim_count = min(max(db.inputs.dimCount, 0), wp.types.ARRAY_MAX_DIMS)
    resolution = tuple(max(getattr(db.inputs, "dim{}".format(i + 1)), 0) for i in range(dim_count))

    # We need to dereference OG's attribute pointer to get the actual pointer
    # to the data.
    data_ptr = ctypes.cast(db.inputs.data.memory, ctypes.POINTER(ctypes.c_size_t)).contents.value

    # Write the texture to the provider.
    provider = ui.DynamicTextureProvider(uri[10:])
    provider.set_bytes_data_from_gpu(
        data_ptr,
        resolution,
        format=ui.TextureFormat.RGBA32_SFLOAT,
    )


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnTextureWrite:
    """Dynamic texture write node."""

    @staticmethod
    def compute(db: OgnTextureWriteDatabase) -> None:
        try:
            compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            return

        # Trigger the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
