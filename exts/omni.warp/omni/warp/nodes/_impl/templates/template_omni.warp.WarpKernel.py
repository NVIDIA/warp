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

from functools import partial
from typing import (
    Optional,
    Sequence,
)

import omni.graph.core as og
from omni.kit.property.usd.custom_layout_helper import (
    CustomLayoutFrame,
    CustomLayoutGroup,
    CustomLayoutProperty,
)
from omni.kit.property.usd.usd_property_widget import UsdPropertyUiEntry
from omni.warp.nodes._impl.attributes import (
    attr_get_base_name,
    attr_get_name,
)
from omni.warp.nodes._impl.common import SUPPORTED_SDF_DATA_TYPE_NAMES
from omni.warp.nodes._impl.kernel import EXPLICIT_SOURCE
from omni.warp.nodes._impl.props.codefile import get_code_file_prop_builder
from omni.warp.nodes._impl.props.codestr import get_code_str_prop_builder
from omni.warp.nodes._impl.props.editattrs import get_edit_attrs_prop_builder
from omni.warp.nodes._impl.props.sourcepicker import get_source_picker_prop_builder

import warp as wp


def find_prop(
    props: Sequence[UsdPropertyUiEntry],
    name: str,
) -> Optional[UsdPropertyUiEntry]:
    """Finds a prop by its name."""
    try:
        return next(p for p in props if p.prop_name == name)
    except StopIteration:
        return None


class CustomLayout:
    """Custom UI for the kernel node."""

    def __init__(self, compute_node_widget):
        self.enable = True
        self.compute_node_widget = compute_node_widget
        self.node_prim_path = self.compute_node_widget._payload[-1]
        self.node = og.Controller.node(self.node_prim_path)

        self.dim_source_attr = og.Controller.attribute(
            "inputs:dimSource",
            self.node,
        )
        self.dim_count_attr = og.Controller.attribute(
            "inputs:dimCount",
            self.node,
        )
        self.code_provider_attr = og.Controller.attribute(
            "inputs:codeProvider",
            self.node,
        )
        self.user_attr_descs_attr = og.Controller.attribute(
            "state:userAttrDescs",
            self.node,
        )
        self.user_attrs_event_attr = og.Controller.attribute(
            "state:userAttrsEvent",
            self.node,
        )

        self.node.register_on_connected_callback(self._handle_node_attr_connected)
        self.node.register_on_disconnected_callback(self._handle_node_attr_disconnected)
        self.dim_source_attr.register_value_changed_callback(self._handle_dim_source_value_changed)
        self.dim_count_attr.register_value_changed_callback(self._handle_dim_count_value_changed)
        self.code_provider_attr.register_value_changed_callback(self._handle_code_provider_value_changed)

    def _handle_node_attr_connected(
        self,
        attr_from: og.Attribute,
        attr_to: og.Attribute,
    ) -> None:
        """Callback for a node attribute having been disconnected."""
        if attr_get_name(attr_to) == "inputs:codeStr":
            # Redraw the UI to update the view/edit code button label.
            self.refresh()

    def _handle_node_attr_disconnected(
        self,
        attr_from: og.Attribute,
        attr_to: og.Attribute,
    ) -> None:
        """Callback for a node attribute having been disconnected."""
        if attr_get_name(attr_to) == "inputs:codeStr":
            # Redraw the UI to update the view/edit code button label.
            self.refresh()

    def _handle_dim_source_value_changed(self, attr: og.Attribute) -> None:
        """Callback for the dimension source attribute value having changed."""
        # Redraw the UI to display a different set of attributes depending on
        # the dimension source value.
        self.refresh()

    def _handle_dim_count_value_changed(self, attr: og.Attribute) -> None:
        """Callback for the dimension count attribute value having changed."""
        # Redraw the UI to display a different set of attributes depending on
        # the dimension count value.
        self.refresh()

    def _handle_code_provider_value_changed(self, attr: og.Attribute) -> None:
        """Callback for the code provider attribute value having changed."""
        # Redraw the UI to display a different set of attributes depending on
        # the code provider value.
        self.refresh()

    def refresh(self) -> None:
        """Redraws the UI."""
        self.compute_node_widget.rebuild_window()

    def apply(self, props) -> Sequence[UsdPropertyUiEntry]:
        """Builds the UI."""
        input_array_attrs = tuple(
            x
            for x in self.node.get_attributes()
            if (
                x.is_dynamic()
                and x.get_port_type() == og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT
                and x.get_resolved_type().array_depth > 0
            )
        )
        dim_sources = (EXPLICIT_SOURCE,) + tuple(attr_get_base_name(x) for x in input_array_attrs)

        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("Add and Remove Attributes"):
                CustomLayoutProperty(
                    None,
                    display_name=None,
                    build_fn=get_edit_attrs_prop_builder(
                        self,
                        SUPPORTED_SDF_DATA_TYPE_NAMES,
                    ),
                )

            with CustomLayoutGroup("Inputs"):
                prop = find_prop(props, "inputs:device")
                if prop is not None:
                    CustomLayoutProperty(
                        prop.prop_name,
                        display_name=prop.metadata["customData"]["uiName"],
                    )

                prop = find_prop(props, "inputs:dimSource")
                if prop is not None:
                    CustomLayoutProperty(
                        prop.prop_name,
                        display_name=prop.metadata["customData"]["uiName"],
                        build_fn=partial(
                            get_source_picker_prop_builder(
                                self,
                                dim_sources,
                            ),
                            prop,
                        ),
                    )

                dim_source = og.Controller.get(self.dim_source_attr)
                if dim_source == EXPLICIT_SOURCE:
                    prop = find_prop(props, "inputs:dimCount")
                    if prop is not None:
                        CustomLayoutProperty(
                            prop.prop_name,
                            display_name=prop.metadata["customData"]["uiName"],
                        )

                    dim_count = min(
                        max(
                            og.Controller.get(self.dim_count_attr),
                            0,
                        ),
                        wp.types.ARRAY_MAX_DIMS,
                    )
                    for i in range(dim_count):
                        prop = find_prop(props, f"inputs:dim{i + 1}")
                        if prop is not None:
                            CustomLayoutProperty(
                                prop.prop_name,
                                display_name=prop.metadata["customData"]["uiName"],
                            )

                prop = find_prop(props, "inputs:codeProvider")
                if prop is not None:
                    CustomLayoutProperty(
                        prop.prop_name,
                        display_name=prop.metadata["customData"]["uiName"],
                    )

                code_provider = og.Controller.get(self.code_provider_attr)
                if code_provider == "embedded":
                    prop = find_prop(props, "inputs:codeStr")
                    if prop is not None:
                        CustomLayoutProperty(
                            prop.prop_name,
                            display_name=prop.metadata["customData"]["uiName"],
                            build_fn=partial(
                                get_code_str_prop_builder(self),
                                prop,
                            ),
                        )
                elif code_provider == "file":
                    prop = find_prop(props, "inputs:codeFile")
                    if prop is not None:
                        CustomLayoutProperty(
                            prop.prop_name,
                            display_name=prop.metadata["customData"]["uiName"],
                            build_fn=partial(
                                get_code_file_prop_builder(self),
                                prop,
                            ),
                        )
                else:
                    raise RuntimeError(f"Unexpected code provider '{code_provider}'")

        return frame.apply(props)
