# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Node retrieving a time with a fixed time step."""

import omni.timeline


class OgnFixedTimeState:
    def __init__(self):
        self.time = 0.0
        self.initialized = False


class OgnFixedTime:
    """Node."""

    @staticmethod
    def internal_state():
        return OgnFixedTimeState()

    @staticmethod
    def compute(db) -> bool:
        """Compute the outputs from the current input"""

        timeline = omni.timeline.get_timeline_interface()
        context = db.internal_state

        if context.initialized == False:
            context.time = db.inputs.start
            context.initialized = True

        db.outputs.time = context.time

        if timeline.is_playing():
            context.time += 1.0 / db.inputs.fps

        return True
