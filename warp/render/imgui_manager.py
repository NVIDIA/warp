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

import warp as wp


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
            print('Warning: imgui not found. To use the UI, please install it with: pip install "imgui[pyglet]"')
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

    def open_save_file_dialog(
        self,
        title: str = "Save File",
        defaultextension: str = "",
        filetypes: list[tuple[str, str]] | None = None,
    ) -> str | None:
        """Opens a file dialog for saving a file and returns the selected path."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            print("Warning: tkinter not found. To use the file dialog, please install it.")
            return None

        try:
            root = tk.Tk()
        except tk.TclError:
            print("Warning: no display found - cannot open file dialog.")
            return None

        root.withdraw()  # Hide the main window
        file_path = filedialog.asksaveasfilename(
            defaultextension=defaultextension,
            filetypes=filetypes or [("All Files", "*.*")],
            title=title,
        )
        root.destroy()
        return file_path

    def open_load_file_dialog(
        self, title: str = "Open File", filetypes: list[tuple[str, str]] | None = None
    ) -> str | None:
        """Opens a file dialog for loading a file and returns the selected path."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            print("Warning: tkinter not found. To use the file dialog, please install it.")
            return None

        try:
            root = tk.Tk()
        except tk.TclError:
            print("Warning: no display found - cannot open file dialog.")
            return None

        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(
            filetypes=filetypes or [("All Files", "*.*")],
            title=title,
        )
        root.destroy()
        return file_path

    def drag_vec3(self, label, vec, speed=0.1, min_val=0.0, max_val=100.0):
        """Helper method to create a drag widget for a wp.vec3"""
        changed, *values = self.imgui.drag_float3(label, *vec, speed, min_val, max_val)
        if changed:
            vec = wp.vec3(*values)
        return changed, vec

    def drag_vec3_list(
        self,
        label_prefix: str,
        vec_list: list[wp.vec3] | wp.array,
        speed: float = 0.1,
        min_val: float = 0.0,
        max_val: float = 100.0,
        max_num_elements: int = 10,
    ) -> tuple[bool, list[wp.vec3] | wp.array]:
        """Helper method to create drag widgets for a list of wp.vec3.

        Note: The `label_prefix` is used to create unique IDs for each widget.
        If you are displaying multiple lists, ensure that `label_prefix` is unique
        for each list to prevent UI elements from interfering with each other.
        """
        self.imgui.text(f"List: {label_prefix} with {len(vec_list)} elements")
        changed = False

        # Convert to numpy array if it's a warp array
        is_warp_array = isinstance(vec_list, wp.array)
        working_array = vec_list.numpy() if is_warp_array else vec_list

        for i, vec in enumerate(working_array[:max_num_elements]):
            vec_changed, new_vec = self.drag_vec3(f"{label_prefix} {i}", vec, speed, min_val, max_val)
            if vec_changed:
                working_array[i] = new_vec
                changed = True

        if is_warp_array:
            if changed:
                return changed, wp.array(working_array, dtype=wp.vec3)
            return changed, vec_list
        return changed, working_array

    def drag_vec2(
        self, label: str, vec: wp.vec2, speed: float = 0.1, min_val: float = 0.0, max_val: float = 100.0
    ) -> tuple[bool, wp.vec2]:
        """Helper method to create a drag widget for a wp.vec2"""
        changed, *values = self.imgui.drag_float2(label, *vec, speed, min_val, max_val)
        if changed:
            vec = wp.vec2(*values)
        return changed, vec

    def drag_vec2_list(
        self,
        label_prefix: str,
        vec_list: list[wp.vec2] | wp.array,
        speed: float = 0.1,
        min_val: float = 0.0,
        max_val: float = 100.0,
        max_num_elements: int = 10,
    ) -> tuple[bool, list[wp.vec2] | wp.array]:
        """Helper method to create drag widgets for a list of wp.vec2.

        Note: The `label_prefix` is used to create unique IDs for each widget.
        If you are displaying multiple lists, ensure that `label_prefix` is unique
        for each list to prevent UI elements from interfering with each other.
        """
        self.imgui.text(f"List: {label_prefix} with {len(vec_list)} elements")
        changed = False

        # Convert to numpy array if it's a warp array
        is_warp_array = isinstance(vec_list, wp.array)
        working_array = vec_list.numpy() if is_warp_array else vec_list

        for i, vec in enumerate(working_array[:max_num_elements]):
            vec_changed, new_vec = self.drag_vec2(f"{label_prefix} {i}", vec, speed, min_val, max_val)
            if vec_changed:
                working_array[i] = new_vec
                changed = True

        if is_warp_array:
            if changed:
                return changed, wp.array(working_array, dtype=wp.vec2)
            return changed, vec_list
        return changed, working_array

    def drag_vec4(
        self, label: str, vec: wp.vec4, speed: float = 0.1, min_val: float = 0.0, max_val: float = 100.0
    ) -> tuple[bool, wp.vec4]:
        """Helper method to create a drag widget for a wp.vec4"""
        changed, *values = self.imgui.drag_float4(label, *vec, speed, min_val, max_val)
        if changed:
            vec = wp.vec4(*values)
        return changed, vec

    def drag_vec4_list(
        self,
        label_prefix: str,
        vec_list: list[wp.vec4] | wp.array,
        speed: float = 0.1,
        min_val: float = 0.0,
        max_val: float = 100.0,
        max_num_elements: int = 10,
    ) -> tuple[bool, list[wp.vec4] | wp.array]:
        """Helper method to create drag widgets for a list of wp.vec4.

        Note: The `label_prefix` is used to create unique IDs for each widget.
        If you are displaying multiple lists, ensure that `label_prefix` is unique
        for each list to prevent UI elements from interfering with each other.
        """
        self.imgui.text(f"List: {label_prefix} with {len(vec_list)} elements")
        changed = False

        # Convert to numpy array if it's a warp array
        is_warp_array = isinstance(vec_list, wp.array)
        working_array = vec_list.numpy() if is_warp_array else vec_list

        for i, vec in enumerate(working_array[:max_num_elements]):
            vec_changed, new_vec = self.drag_vec4(f"{label_prefix} {i}", vec, speed, min_val, max_val)
            if vec_changed:
                working_array[i] = new_vec
                changed = True

        if is_warp_array:
            if changed:
                return changed, wp.array(working_array, dtype=wp.vec4)
            return changed, vec_list
        return changed, working_array

    def drag_float(
        self, label: str, val: float, speed: float = 0.1, min_val: float = 0.0, max_val: float = 100.0
    ) -> tuple[bool, float]:
        """Helper method to create a drag widget for a float"""
        changed, value = self.imgui.drag_float(label, val, speed, min_val, max_val)
        if changed:
            val = value
        return changed, val

    def drag_float_list(
        self,
        label_prefix: str,
        float_list: list[float] | wp.array,
        speed: float = 0.1,
        min_val: float = 0.0,
        max_val: float = 100.0,
        max_num_elements: int = 10,
    ) -> tuple[bool, list[float] | wp.array]:
        """Helper method to create drag widgets for a list of floats.

        Note: The `label_prefix` is used to create unique IDs for each widget.
        If you are displaying multiple lists, ensure that `label_prefix` is unique
        for each list to prevent UI elements from interfering with each other.
        """
        self.imgui.text(f"List: {label_prefix} with {len(float_list)} elements")
        changed = False

        # Convert to numpy array if it's a warp array
        is_warp_array = isinstance(float_list, wp.array)
        working_array = float_list.numpy() if is_warp_array else float_list

        for i, val in enumerate(working_array[:max_num_elements]):
            val_changed, new_val = self.drag_float(f"{label_prefix} {i}", val, speed, min_val, max_val)
            if val_changed:
                working_array[i] = new_val
                changed = True

        if is_warp_array:
            if changed:
                return changed, wp.array(working_array, dtype=float_list.dtype)
            return changed, float_list
        return changed, working_array

    def shutdown(self):
        if not self.is_available:
            return
        self.impl.shutdown()
