# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from functools import partial

import omni.graph.core as og
from omni.kit.widget.searchfield import SearchField
import omni.ui as ui

_DIALOG_TITLE = "Attribute Creator"
_DIALOG_WIDTH = 400
_DIALOG_HEIGHT = 0
_DIALOG_PADDING = 15
_BUTTON_WIDTH = 100
_WIDGET_PADDING = 5

_FIELD_WIDTH = ui.Percent(60)

class AttributeEditor:
    """Editor to add/remove node attributes."""

    def __init__(self, data_types, create_attr_callback):
        self.supported_data_types = data_types
        self.filtered_data_types = data_types
        self.create_attr_callback = create_attr_callback
        self.dialog = None
        self.data_type_frame = None
        self.name_field = None
        self.input_port_btn = None
        self.output_port_btn = None
        self.selected_data_type_btn = None
        self.error_msg_label = None
        self._build()

    def _handle_data_type_clicked(self, btn):
        if self.selected_data_type_btn is not None:
            self.selected_data_type_btn.checked = False

        self.selected_data_type_btn = btn
        self.selected_data_type_btn.checked = True

    def _handle_search(self, text):
        if text is None:
            self.filtered_data_types = self.supported_data_types
        else:
            text = text[0]
            self.filtered_data_types = tuple(
                x for x in self.supported_data_types
                if text in x
            )

        self._build_data_type_frame()
        self.selected_data_type_btn = None

    def _handle_ok_btn_clicked(self):
        name = self.name_field.model.get_value_as_string()

        if not name:
            self.error_msg_label.text = "Error: Attribute name cannot be empty!"
            return

        if not name[0].isalpha():
            self.error_msg_label.text = (
                "Error: The first character of attribute name must be a letter!"
            )
            return

        if self.selected_data_type_btn is None:
            self.error_msg_label.text = (
                "Error: You must select a type for the new attribute!"
            )
            return

        if self.input_port_btn.checked:
            port_type = og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT
        elif self.output_port_btn.checked:
            port_type = og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT
        else:
            self.error_msg_label.text = "Error: You must select a port type!"
            return

        data_type_name = self.selected_data_type_btn.text
        if (
            port_type == og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT
            and not data_type_name.endswith("[]")
        ):
            self.error_msg_label.text = (
                "Error: Output attributes are required to be arrays!"
            )
            return

        self.create_attr_callback(name, port_type, data_type_name)

        self.dialog.visible = False

    def _handle_cancel_btn_clicked(self):
        self.dialog.visible = False

    def _build_data_type_frame(self):
        self.data_type_frame.clear()
        with self.data_type_frame:
            with ui.VStack():
                for data_type in self.filtered_data_types:
                    btn = ui.Button(data_type)
                    btn.set_clicked_fn(
                        partial(
                            self._handle_data_type_clicked,
                            btn,
                        ),
                    )

    def _build(self):
        self.dialog = ui.Window(
            _DIALOG_TITLE,
            width=_DIALOG_WIDTH,
            height=_DIALOG_HEIGHT,
            padding_x=_DIALOG_PADDING,
            padding_y=_DIALOG_PADDING,
            flags=ui.WINDOW_FLAGS_NO_RESIZE,
        )

        with self.dialog.frame:
            with ui.VStack(spacing=10):

                # Name.
                with ui.HStack(height=0):
                    ui.Label("Attribute Name: ")
                    self.name_field = ui.StringField(width=_FIELD_WIDTH)

                # Port type.
                with ui.HStack(height=0):
                    ui.Label("Attribute Port Type: ")
                    radio_collection = ui.RadioCollection()
                    with ui.HStack(width=_FIELD_WIDTH):
                        self.input_port_btn = ui.RadioButton(
                            text="input",
                            radio_collection=radio_collection
                        )
                        self.output_port_btn = ui.RadioButton(
                            text="output",
                            radio_collection=radio_collection
                        )

                # Data type.
                with ui.HStack(height=0):
                    ui.Label("Attribute Type: ", alignment=ui.Alignment.LEFT)
                    with ui.VStack(width=_FIELD_WIDTH):
                        SearchField(
                            show_tokens=False,
                            on_search_fn=self._handle_search,
                            subscribe_edit_changed=True,
                        )

                        self.data_type_frame = ui.ScrollingFrame(
                            height=150,
                            horizontal_scrollbar_policy=(
                                ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF
                            ),
                            style_type_name_override="TreeView"
                        )
                        self._build_data_type_frame()

                # Dialog buttons
                with ui.HStack(height=0):
                    ui.Spacer()
                    with ui.HStack(width=_FIELD_WIDTH, height=20):
                        ui.Button(
                            "OK",
                            clicked_fn=self._handle_ok_btn_clicked,
                        )
                        ui.Button(
                            "Cancel",
                            clicked_fn=self._handle_cancel_btn_clicked,
                        )

                # Placeholder to display any error message.
                self.error_msg_label = ui.Label(
                    "",
                    height=20,
                    alignment=ui.Alignment.H_CENTER,
                    style={
                        "color": 0xFF0000FF,
                    },
                )
