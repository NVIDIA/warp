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

"""Property to edit the node attributes based on the kernel's parameters."""

from functools import partial
from typing import (
    Any,
    Callable,
    Sequence,
)

import omni.graph.core as og
import omni.ui as ui

from ..attributes import (
    ATTR_BUNDLE_TYPE,
    attr_get_name,
    attr_join_name,
)
from ..kernel import (
    ArrayAttributeFormat,
    UserAttributeDesc,
    UserAttributesEvent,
    deserialize_user_attribute_descs,
    serialize_user_attribute_descs,
)
from ..widgets.attributeeditor import AttributeEditor

_BUTTON_WIDTH = 100


class _State:
    """State object shared across the various handlers."""

    def __init__(self, layout):
        self.layout = layout
        self.dialog = None
        self.remove_attr_menu = None


def _add_user_attribute_desc(state: _State, desc: UserAttributeDesc) -> None:
    data = og.Controller.get(state.layout.user_attr_descs_attr)
    descs = deserialize_user_attribute_descs(data)

    descs[desc.name] = desc

    data = serialize_user_attribute_descs(descs)
    og.Controller.set(state.layout.user_attr_descs_attr, data, update_usd=True)


def _remove_user_attribute_desc(
    state: _State,
    port_type: og.AttributePortType,
    base_name: str,
) -> None:
    data = og.Controller.get(state.layout.user_attr_descs_attr)
    descs = deserialize_user_attribute_descs(data)

    name = attr_join_name(port_type, base_name)
    descs.pop(name, None)

    data = serialize_user_attribute_descs(descs)
    og.Controller.set(state.layout.user_attr_descs_attr, data, update_usd=True)


def _get_attribute_creation_handler(state: _State) -> Callable:
    def fn(attr_desc: UserAttributeDesc):
        if any(attr_get_name(x) == attr_desc.name for x in state.layout.node.get_attributes()):
            raise RuntimeError(f"The attribute '{attr_desc.name}' already exists on the node.")

        if attr_desc.array_format == ArrayAttributeFormat.RAW:
            attr_type = attr_desc.type
        elif attr_desc.array_format == ArrayAttributeFormat.BUNDLE:
            attr_type = ATTR_BUNDLE_TYPE
        else:
            raise AssertionError(f"Unexpected array attribute format '{attr_desc.array_format}'.")

        attr = og.Controller.create_attribute(
            state.layout.node,
            attr_desc.base_name,
            attr_type,
            attr_desc.port_type,
        )
        if attr is None:
            raise RuntimeError(f"Failed to create the attribute '{attr_desc.name}'.")

        attr.is_optional_for_compute = attr_desc.optional

        # Store the new attribute's description within the node's state.
        _add_user_attribute_desc(state, attr_desc)

        # Inform the node that a new attribute was created.
        og.Controller.set(
            state.layout.user_attrs_event_attr,
            UserAttributesEvent.CREATED,
        )

        state.layout.refresh()

    return fn


def _get_attribute_removal_handler(state: _State) -> Callable:
    def fn(attr):
        port_type = attr.get_port_type()
        name = attr_get_name(attr)

        if not og.Controller.remove_attribute(attr):
            return

        # Remove that attribute's description from the node's state.
        _remove_user_attribute_desc(state, port_type, name)

        # Inform the node that an existing attribute was removed.
        og.Controller.set(
            state.layout.user_attrs_event_attr,
            UserAttributesEvent.REMOVED,
        )

        state.layout.refresh()

    return fn


def _get_add_btn_clicked_handler(
    state: _State,
    supported_types: Sequence[str],
) -> Callable:
    def fn():
        dialog = AttributeEditor(
            supported_types,
            _get_attribute_creation_handler(state),
        )

        # Store the dialog widget into the state to avoid having it
        # not showing up due to being garbage collected.
        state.dialog = dialog

    return fn


def _get_remove_btn_clicked_handler(state: _State) -> Callable:
    def fn():
        attrs = tuple(x for x in state.layout.node.get_attributes() if x.is_dynamic())
        if not attrs:
            return

        menu = ui.Menu()
        with menu:
            for attr in attrs:
                ui.MenuItem(
                    attr_get_name(attr),
                    triggered_fn=partial(
                        _get_attribute_removal_handler(state),
                        attr,
                    ),
                )

        menu.show()

        # Store the menu widget into the state to avoid having it
        # not showing up due to being garbage collected.
        state.remove_attr_menu = menu

    return fn


def get_edit_attrs_prop_builder(
    layout: Any,
    supported_types: Sequence[str],
) -> Callable:
    """Builds the function used to create the property."""

    def fn(*args):
        state = _State(layout)

        with ui.HStack():
            ui.Button(
                "Add +",
                width=_BUTTON_WIDTH,
                clicked_fn=_get_add_btn_clicked_handler(state, supported_types),
                tooltip="Opens an UI to add a new attribute",
            )

            ui.Spacer(width=8)

            ui.Button(
                "Remove -",
                width=_BUTTON_WIDTH,
                clicked_fn=_get_remove_btn_clicked_handler(state),
                tooltip="Opens a menu to remove an existing node attribute",
            )

    return fn
