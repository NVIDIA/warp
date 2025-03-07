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

"""Property to pick the source used to infer an attribute's value."""

from typing import (
    Any,
    Callable,
    Sequence,
    Tuple,
)

import omni.ui as ui
from omni.kit.property.usd.usd_attribute_model import TfTokenAttributeModel
from omni.kit.property.usd.usd_property_widget import UsdPropertyUiEntry
from omni.kit.property.usd.usd_property_widget_builder import UsdPropertiesWidgetBuilder
from omni.kit.window.property.templates import HORIZONTAL_SPACING


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
