# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Property to pick the source used to infer an attribute's value."""

from typing import (
    Any,
    Callable,
    Sequence,
    Tuple,
)

from omni.kit.property.usd.usd_attribute_model import TfTokenAttributeModel
from omni.kit.property.usd.usd_property_widget import UsdPropertyUiEntry
from omni.kit.property.usd.usd_property_widget_builder import UsdPropertiesWidgetBuilder
from omni.kit.window.property.templates import HORIZONTAL_SPACING
import omni.ui as ui


def get_source_picker_prop_builder(
    layout: Any,
    sources: Sequence[str],
) -> Callable:
    """Builds the function used to create the property."""

    def fn(ui_prop: UsdPropertyUiEntry, *args):
        class _Model(TfTokenAttributeModel):
            def _get_allowed_tokens(self, _) -> Tuple[str]:
                return tuple(sources)

        with ui.HStack(spacing=HORIZONTAL_SPACING):
            # It is necessary to define a dummy allowed token otherwise
            # the code in `UsdPropertiesWidgetBuilder._tftoken_builder()` exits
            # before attempting to call our custom model.
            metadata = ui_prop.metadata.copy()
            metadata.update(
                {
                    "allowedTokens": ("dummy",),
                }
            )

            UsdPropertiesWidgetBuilder.build(
                layout.compute_node_widget.stage,
                ui_prop.prop_name,
                metadata,
                ui_prop.property_type,
                prim_paths=(layout.node_prim_path,),
                additional_label_kwargs={
                    "style": {
                        "alignment": ui.Alignment.RIGHT,
                    },
                },
                additional_widget_kwargs={
                    "model_cls": _Model,
                },
            )

    return fn
