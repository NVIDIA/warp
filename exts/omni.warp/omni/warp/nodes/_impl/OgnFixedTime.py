# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        context = db.per_instance_state

        if not context.initialized:
            context.time = db.inputs.start
            context.initialized = True

        db.outputs.time = context.time

        if timeline.is_playing():
            context.time += 1.0 / db.inputs.fps

        return True
