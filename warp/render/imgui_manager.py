# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations


class ImGuiManager:
    """Base class for managing an ImGui UI."""

    def __init__(self, renderer):
        try:
            import imgui
            from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer

            self.imgui = imgui
            self.is_available = True
        except ImportError:
            self.is_available = False
            print("Warning: imgui not found. To use the UI, please install it with: pip install imgui[pyglet]")
            return

        self.imgui.create_context()
        self.renderer = renderer
        self.impl = PygletProgrammablePipelineRenderer(self.renderer.window)

    def render_frame(self):
        """Renders a single frame of the UI. This should be called from the main render loop."""
        if not self.is_available:
            return

        io = self.imgui.get_io()
        self.renderer.enable_mouse_interaction = not io.want_capture_mouse
        self.renderer.enable_keyboard_interaction = not io.want_capture_keyboard
        io.display_size = self.renderer.screen_width, self.renderer.screen_height

        self.imgui.new_frame()

        self.draw_ui()

        self.imgui.render()
        self.imgui.end_frame()
        self.impl.render(self.imgui.get_draw_data())

    def draw_ui(self):
        """Draws the UI. To be implemented by subclasses."""
        pass

    def shutdown(self):
        if not self.is_available:
            return
        self.impl.shutdown()
