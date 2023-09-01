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
import omni.warp.nodes
import warp as wp

from omni.warp.nodes.ogn.OgnTextureWriteDatabase import OgnTextureWriteDatabase

try:
    import omni.ui as ui
except ImportError:
    ui = None


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Internal state for the node."""

    def __init__(self) -> None:
        self.texture_provider = None

        self.is_valid = False

        self.attr_tracking = omni.warp.nodes.AttrTracking(
            (
                "uri",
            ),
        )

    def needs_initialization(self, db: OgnTextureWriteDatabase) -> bool:
        """Checks if the internal state needs to be (re)initialized."""
        if not self.is_valid:
            return True

        if self.attr_tracking.have_attrs_changed(db):
            return True

        return False

    def initialize(
        self,
        db: OgnTextureWriteDatabase,
    ) -> bool:
        """Initializes the internal state."""
        uri = db.inputs.uri
        if not uri.startswith("dynamic://"):
            return False

        texture_provider = ui.DynamicTextureProvider(uri[10:])

        # Store the class members.
        self.texture_provider = texture_provider

        self.attr_tracking.update_state(db)

        return True


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnTextureWriteDatabase) -> None:
    """Evaluates the node."""
    if ui is None:
        db.log_warning("Cannot write dynamic textures in headless mode.")
        return

    if not db.inputs.data.memory or db.inputs.data.shape[0] == 0:
        return

    state = db.internal_state

    if state.needs_initialization(db):
        # Initialize the internal state if it hasn't been already.
        if not state.initialize(db):
            return

    dim_count = min(max(db.inputs.dimCount, 0), wp.types.ARRAY_MAX_DIMS)
    resolution = tuple(max(getattr(db.inputs, "dim{}".format(i + 1)), 0) for i in range(dim_count))

    # We need to dereference OG's attribute pointer to get the actual pointer
    # to the data.
    data_ptr = ctypes.cast(db.inputs.data.memory, ctypes.POINTER(ctypes.c_size_t)).contents.value

    # Write the texture to the provider.
    state.texture_provider.set_bytes_data_from_gpu(
        data_ptr,
        resolution,
        format=ui.TextureFormat.RGBA32_SFLOAT,
    )


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnTextureWrite:
    """Dynamic texture write node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnTextureWriteDatabase) -> None:
        try:
            compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            db.internal_state.is_valid = False
            return

        db.internal_state.is_valid = True

        # Trigger the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
