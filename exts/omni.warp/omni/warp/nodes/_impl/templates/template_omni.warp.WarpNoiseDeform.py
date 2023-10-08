# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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


PARTIAL_PROPS = (
    "inputs:upAxis",
    "inputs:base",
    "inputs:falloff",
)

SHARED_PROPS = (
    "inputs:func",
    "inputs:cellSize",
    "inputs:speed",
    "inputs:amplitude",
    "inputs:axisAmplitude",
    "inputs:seed",
    "inputs:time",
)


def find_prop(
    props: Sequence[UsdPropertyUiEntry],
    name: str,
) -> Optional[UsdPropertyUiEntry]:
    try:
        return next(x for x in props if x.prop_name == name)
    except StopIteration:
        return None


class CustomLayout:
    def __init__(self, compute_node_widget):
        self.enable = True
        self.compute_node_widget = compute_node_widget
        self.node_prim_path = self.compute_node_widget._payload[-1]
        self.node = og.Controller.node(self.node_prim_path)

        self.mode_attr = og.Controller.attribute(
            "inputs:mode",
            self.node,
        )

        self.mode_attr.register_value_changed_callback(self._handle_mode_value_changed)

    def _handle_mode_value_changed(self, attr: og.Attribute) -> None:
        """Callback for the mode attribute value having changed."""
        # Redraw the UI to display a different set of attributes depending on
        # the mode value.
        self.refresh()

    def refresh(self) -> None:
        """Redraws the UI."""
        self.compute_node_widget.rebuild_window()

    def apply(self, props) -> Sequence[UsdPropertyUiEntry]:
        frame = CustomLayoutFrame(hide_extra=True)
        with frame:
            with CustomLayoutGroup("Inputs"):
                prop = find_prop(props, "inputs:mode")
                if prop is not None:
                    CustomLayoutProperty(
                        prop.prop_name,
                        display_name=prop.metadata["customData"]["uiName"],
                    )

                mode = og.Controller.get(self.mode_attr)
                if mode == "partial":
                    partial_props = tuple(find_prop(props, x) for x in PARTIAL_PROPS)
                    for prop in partial_props:
                        if prop is not None:
                            CustomLayoutProperty(
                                prop.prop_name,
                                display_name=prop.metadata["customData"]["uiName"],
                            )

                shared_props = tuple(find_prop(props, x) for x in SHARED_PROPS)
                for prop in shared_props:
                    if prop is not None:
                        CustomLayoutProperty(
                            prop.prop_name,
                            display_name=prop.metadata["customData"]["uiName"],
                        )

        return frame.apply(props)
