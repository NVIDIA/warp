# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Property to edit the node attributes based on the kernel's parameters."""

from functools import partial
from typing import (
    Any,
    Callable,
    Sequence,
)

import omni.graph.core as og
import omni.ui as ui

from omni.warp.scripts.widgets.attributeeditor import AttributeEditor

_BUTTON_WIDTH = 100

class _State:
    """State object shared across the various handlers."""

    def __init__(self, layout):
        self.layout = layout
        self.dialog = None
        self.remove_attr_menu = None

def _get_attribute_creation_handler(state: _State) -> Callable:

    def fn(name, port_type, data_type_name):
        type_name = og.AttributeType.type_from_sdf_type_name(data_type_name)
        if og.Controller.create_attribute(
            state.layout.node,
            "{}:{}".format(og.get_port_type_namespace(port_type), name),
            type_name,
            port_type,
        ) is None:
            return

        state.layout.refresh()

    return fn

def _get_attribute_removal_handler(state: _State) -> Callable:

    def fn(attr):
        if not og.Controller.remove_attribute(attr):
            return

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
        attrs = tuple(
            x for x in state.layout.node.get_attributes()
            if x.is_dynamic()
        )
        if not attrs:
            return

        menu = ui.Menu()
        with menu:
            for attr in attrs:
                ui.MenuItem(
                    attr.get_name(),
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
                tooltip="Opens an UI to edit the node's attributes",
            )

    return fn
