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

        self.dim_count_attr = og.Controller.attribute(
            "inputs:dimCount",
            self.node,
        )

        self.dim_count_attr.register_value_changed_callback(self._handle_dim_count_value_changed)

    def _handle_dim_count_value_changed(self, attr: og.Attribute) -> None:
        """Callback for the dimension count attribute value having changed."""
        # Redraw the UI to display a different set of attributes depending on
        # the dimension count value.
        self.refresh()

    def refresh(self) -> None:
        """Redraws the UI."""
        self.compute_node_widget.rebuild_window()

    def apply(self, props) -> Sequence[UsdPropertyUiEntry]:
        """Builds the UI."""
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("Inputs"):
                prop = find_prop(props, "inputs:uri")
                if prop is not None:
                    CustomLayoutProperty(
                        prop.prop_name,
                        display_name=prop.metadata["customData"]["uiName"],
                    )

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
                    prop = find_prop(
                        props,
                        f"inputs:dim{i + 1}",
                    )
                    if prop is not None:
                        CustomLayoutProperty(
                            prop.prop_name,
                            display_name=(prop.metadata["customData"]["uiName"]),
                        )

                prop = find_prop(props, "inputs:pixelFormat")
                if prop is not None:
                    CustomLayoutProperty(
                        prop.prop_name,
                        display_name=prop.metadata["customData"]["uiName"],
                    )

        return frame.apply(props)
