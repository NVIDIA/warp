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

"""Widget for a dialog window to author node attributes."""

from functools import partial

import omni.graph.core as og
import omni.ui as ui
from omni.kit.widget.searchfield import SearchField

from ..kernel import (
    ArrayAttributeFormat,
    OutputArrayShapeSource,
    UserAttributeDesc,
)

_DIALOG_TITLE = "Attribute Creator"
_DIALOG_PADDING = 15
_DATA_TYPE_FRAME_HEIGHT = 300
_LABEL_WIDTH = 100
_FIELD_WIDTH = 275
_DIALOG_BTNS_FRAME_HEIGHT = 20


def _add_label(title):
    ui.Label(title, alignment=ui.Alignment.RIGHT, width=_LABEL_WIDTH)
    ui.Spacer(width=_DIALOG_PADDING)


class AttributeEditor:
    """Editor to add/remove node attributes."""

    def __init__(self, data_type_names, create_attr_callback):
        self.supported_data_type_names = data_type_names
        self.filtered_data_type_names = data_type_names
        self.create_attr_callback = create_attr_callback
        self.dialog = None
        self.input_port_btn = None
        self.output_port_btn = None
        self.name_field = None
        self.data_type_frame = None
        self.selected_data_type_btn = None
        self.is_array_frame = None
        self.is_array_checkbox = None
        self.array_format_frame = None
        self.array_format_combobox = None
        self.output_array_shape_source_frame = None
        self.output_array_shape_source_combobox = None
        self.optional_frame = None
        self.optional_checkbox = None
        self.error_msg_label = None
        self._build()

    @property
    def port_type(self):
        if self.input_port_btn.checked:
            return og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT

        if self.output_port_btn.checked:
            return og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT

        return None

    @property
    def name(self):
        return self.name_field.model.get_value_as_string()

    @property
    def data_type(self):
        if self.selected_data_type_btn is None:
            return None

        return self.selected_data_type_btn.text

    @property
    def is_array(self):
        return self.is_array_checkbox.model.get_value_as_bool()

    @property
    def array_format(self):
        return ArrayAttributeFormat(self.array_format_combobox.model.get_item_value_model().get_value_as_int())

    @property
    def array_shape_source(self):
        if self.port_type == og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT:
            enum_type = OutputArrayShapeSource
            widget = self.output_array_shape_source_combobox
        else:
            return None

        value = widget.model.get_item_value_model().get_value_as_int()
        return enum_type(value)

    @property
    def optional(self):
        return self.optional_checkbox.model.get_value_as_bool()

    def _update_array_shape_source_visibility(self):
        self.output_array_shape_source_frame.visible = (
            self.port_type == og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT and self.is_array
        )

    def _handle_data_type_clicked(self, btn):
        if self.selected_data_type_btn is not None:
            self.selected_data_type_btn.checked = False

        self.selected_data_type_btn = btn
        self.selected_data_type_btn.checked = True

    def _handle_input_port_btn_clicked(self):
        self.is_array_frame.enabled = True
        self.optional_frame.visible = True
        self._update_array_shape_source_visibility()

    def _handle_output_port_btn_clicked(self):
        self.is_array_checkbox.model.set_value(True)

        self.is_array_frame.enabled = False
        self.optional_frame.visible = False
        self._update_array_shape_source_visibility()

    def _handle_data_type_search(self, text):
        if text is None:
            self.filtered_data_type_names = self.supported_data_type_names
        else:
            text = text[0]
            self.filtered_data_type_names = tuple(x for x in self.supported_data_type_names if text in x)

        self._build_data_type_frame()
        self.selected_data_type_btn = None

    def _handle_is_array_value_changed(self, model):
        # TODO: uncomment when support for dynamic bundle attributes is added
        #       in OmniGraph.
        # self.array_format_frame.visible = model.get_value_as_bool()
        self._update_array_shape_source_visibility()

    def _handle_array_format_item_changed(self, model, item):
        self._update_array_shape_source_visibility()

    def _handle_ok_btn_clicked(self):
        port_type = self.port_type
        if port_type is None:
            self._set_error_msg("A port type must be selected.")
            return

        name = self.name
        if not name:
            self._set_error_msg("The attribute's name cannot be empty.")
            return
        if not name[0].isalpha():
            self._set_error_msg("The first character of the attribute's name must be a letter.")
            return

        if self.data_type is None:
            self._set_error_msg("A data type for the new attribute must be selected.")
            return

        attr_desc = UserAttributeDesc(
            port_type=port_type,
            base_name=name,
            data_type_name=self.data_type,
            is_array=self.is_array,
            array_format=self.array_format,
            array_shape_source=self.array_shape_source,
            optional=(self.port_type == og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT and self.optional),
        )

        try:
            self.create_attr_callback(attr_desc)
        except Exception as e:
            self._set_error_msg(str(e))
            return

        self.dialog.visible = False

    def _handle_cancel_btn_clicked(self):
        self.dialog.visible = False

    def _set_error_msg(self, msg):
        self.error_msg_label.text = msg
        self.error_msg_label.visible = True

    def _build_data_type_frame(self):
        self.data_type_frame.clear()
        with self.data_type_frame:
            with ui.VStack():
                for data_type_name in self.filtered_data_type_names:
                    btn = ui.Button(data_type_name)
                    btn.set_clicked_fn(
                        partial(
                            self._handle_data_type_clicked,
                            btn,
                        ),
                    )

    def _build(self):
        self.dialog = ui.Window(
            _DIALOG_TITLE,
            padding_x=_DIALOG_PADDING,
            padding_y=_DIALOG_PADDING,
            auto_resize=True,
        )

        with self.dialog.frame:
            with ui.VStack(spacing=10):
                # Placeholder to display any error message.
                self.error_msg_label = ui.Label(
                    "",
                    alignment=ui.Alignment.H_CENTER,
                    word_wrap=True,
                    visible=False,
                    style={
                        "color": 0xFF0000FF,
                    },
                )

                # Port type.
                with ui.HStack(height=0):
                    _add_label("Port Type")
                    radio_collection = ui.RadioCollection()
                    with ui.HStack(width=_FIELD_WIDTH):
                        self.input_port_btn = ui.RadioButton(
                            text="input",
                            radio_collection=radio_collection,
                            clicked_fn=self._handle_input_port_btn_clicked,
                        )
                        self.output_port_btn = ui.RadioButton(
                            text="output",
                            radio_collection=radio_collection,
                            clicked_fn=self._handle_output_port_btn_clicked,
                        )

                # Name.
                with ui.HStack(height=0):
                    _add_label("Name")
                    self.name_field = ui.StringField(width=_FIELD_WIDTH)

                # Data type.
                with ui.HStack(height=0):
                    _add_label("Data Type")
                    with ui.VStack(width=_FIELD_WIDTH):
                        SearchField(
                            show_tokens=False,
                            on_search_fn=self._handle_data_type_search,
                            subscribe_edit_changed=True,
                        )

                        self.data_type_frame = ui.ScrollingFrame(
                            height=_DATA_TYPE_FRAME_HEIGHT,
                            horizontal_scrollbar_policy=(ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF),
                            style_type_name_override="TreeView",
                        )
                        self._build_data_type_frame()

                # Is array flag.
                self.is_array_frame = ui.HStack(height=0)
                with self.is_array_frame:
                    _add_label("Is Array")
                    self.is_array_checkbox = ui.CheckBox(width=_FIELD_WIDTH)
                    self.is_array_checkbox.model.add_value_changed_fn(self._handle_is_array_value_changed)

                # Array's format.
                self.array_format_frame = ui.HStack(height=0, visible=False)
                with self.array_format_frame:
                    _add_label("Array Format")
                    self.array_format_combobox = ui.ComboBox(
                        0,
                        *(x.label for x in ArrayAttributeFormat),
                        width=_FIELD_WIDTH,
                    )
                    self.array_format_combobox.model.add_item_changed_fn(self._handle_array_format_item_changed)

                # Output array's shape.
                self.output_array_shape_source_frame = ui.HStack(
                    height=0,
                    visible=False,
                )
                with self.output_array_shape_source_frame:
                    _add_label("Array Shape")
                    self.output_array_shape_source_combobox = ui.ComboBox(
                        0,
                        *(x.label for x in OutputArrayShapeSource),
                        width=_FIELD_WIDTH,
                    )

                # Optional flag.
                self.optional_frame = ui.HStack(height=0)
                with self.optional_frame:
                    _add_label("Optional")
                    self.optional_checkbox = ui.CheckBox(width=_FIELD_WIDTH)

                # Dialog buttons.
                with ui.HStack(height=0):
                    ui.Spacer()
                    with ui.HStack(
                        width=_FIELD_WIDTH,
                        height=_DIALOG_BTNS_FRAME_HEIGHT,
                    ):
                        ui.Button(
                            "OK",
                            clicked_fn=self._handle_ok_btn_clicked,
                        )
                        ui.Button(
                            "Cancel",
                            clicked_fn=self._handle_cancel_btn_clicked,
                        )
