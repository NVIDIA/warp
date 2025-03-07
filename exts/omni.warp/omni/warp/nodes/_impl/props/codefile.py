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

"""Property to edit the file path pointing to the kernel's code."""

from typing import (
    Any,
    Callable,
)

import omni.ui as ui
from omni.kit.property.usd.usd_property_widget import UsdPropertyUiEntry
from omni.kit.property.usd.usd_property_widget_builder import UsdPropertiesWidgetBuilder
from omni.kit.window.property.templates import HORIZONTAL_SPACING


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
