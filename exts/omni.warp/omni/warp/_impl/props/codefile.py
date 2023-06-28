# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Property to edit the file path pointing to the kernel's code."""

from typing import (
    Any,
    Callable,
)

from omni.kit.property.usd.usd_property_widget import UsdPropertyUiEntry
from omni.kit.property.usd.usd_property_widget_builder import UsdPropertiesWidgetBuilder
from omni.kit.window.property.templates import HORIZONTAL_SPACING
import omni.ui as ui


def get_code_file_prop_builder(layout: Any) -> Callable:
    """Builds the function used to create the property."""

    def fn(ui_prop: UsdPropertyUiEntry, *args):
        with ui.HStack(spacing=HORIZONTAL_SPACING):
            UsdPropertiesWidgetBuilder._string_builder(
                layout.compute_node_widget.stage,
                ui_prop.prop_name,
                ui_prop.property_type,
                ui_prop.metadata,
                prim_paths=(layout.node_prim_path,),
                additional_label_kwargs={
                    "style": {
                        "alignment": ui.Alignment.RIGHT,
                    },
                },
            )

    return fn
