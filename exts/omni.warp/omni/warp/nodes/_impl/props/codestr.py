# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Property to edit the kernel's source code embedded to the node."""

from typing import (
    Any,
    Callable,
)

import omni.graph.core as og
from omni.kit.property.usd.usd_property_widget import UsdPropertyUiEntry
from omni.kit.property.usd.usd_property_widget_builder import UsdPropertiesWidgetBuilder
from omni.kit.widget.text_editor import TextEditor
from omni.kit.window.property.templates import HORIZONTAL_SPACING
import omni.ui as ui


_DIALOG_TITLE = "Kernel Editor"
_DIALOG_WIDTH = 800
_DIALOG_HEIGHT = 600
_BUTTON_WIDTH = 100


class _State:
    """State object shared across the various handlers."""

    def __init__(self, layout: Any):
        self.layout = layout
        self.dialog = None
        self.editor = None

        self.code_str_attr = og.Controller.attribute(
            "inputs:codeStr",
            layout.node,
        )

    def is_read_only(self) -> bool:
        return self.code_str_attr.get_upstream_connection_count() > 0


def _get_save_btn_clicked_handler(state: _State) -> Callable:
    def fn():
        # Trim the trailing new line character inserted by
        # the text editor.
        code = state.editor.text[:-1]

        # Save the content into the corresponding string attribute.
        assert not state.is_read_only()
        og.Controller.set(state.code_str_attr, code, update_usd=True)

        # Restore the title to its default state.
        state.dialog.title = _DIALOG_TITLE

    return fn


def _get_edit_btn_clicked_handler(state: _State) -> Callable:
    def fn():
        read_only = state.is_read_only()

        dialog = ui.Window(
            _DIALOG_TITLE,
            width=_DIALOG_WIDTH,
            height=_DIALOG_HEIGHT,
            dockPreference=ui.DockPreference.MAIN,
        )

        with dialog.frame:
            with ui.VStack():
                state.editor = TextEditor(
                    syntax=TextEditor.Syntax.PYTHON,
                    text=og.Controller.get(state.code_str_attr),
                    read_only=read_only,
                )

                if not read_only:
                    with ui.HStack(height=0):
                        ui.Spacer()
                        ui.Button(
                            "Save",
                            width=_BUTTON_WIDTH,
                            clicked_fn=_get_save_btn_clicked_handler(state),
                            tooltip="Save the changes to the code",
                        )

        # Store the dialog widget into the state to avoid having it
        # not showing up due to being garbage collected.
        state.dialog = dialog

    return fn


def get_code_str_prop_builder(layout: Any) -> Callable:
    """Builds the function used to create the property."""

    def fn(ui_prop: UsdPropertyUiEntry, *args):
        state = _State(layout)

        with ui.HStack(spacing=HORIZONTAL_SPACING):
            UsdPropertiesWidgetBuilder._create_label(
                ui_prop.prop_name,
                ui_prop.metadata,
                {
                    "style": {
                        "alignment": ui.Alignment.RIGHT,
                    },
                },
            )

            ui.Button(
                "View" if state.is_read_only() else "Edit",
                width=_BUTTON_WIDTH,
                clicked_fn=_get_edit_btn_clicked_handler(state),
                tooltip="View/edit the embedded kernel code",
            )

    return fn
