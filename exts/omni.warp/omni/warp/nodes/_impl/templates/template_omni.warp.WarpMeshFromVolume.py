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

PROPS = (
    "inputs:size",
    "inputs:dim1",
    "inputs:dim2",
    "inputs:dim3",
    "inputs:maxPoints",
    "inputs:maxTriangles",
    "inputs:threshold",
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

    def apply(self, props) -> Sequence[UsdPropertyUiEntry]:
        user_props = tuple(find_prop(props, x) for x in PROPS)

        frame = CustomLayoutFrame(hide_extra=True)
        with frame:
            with CustomLayoutGroup("Inputs"):
                for prop in user_props:
                    if prop is not None:
                        CustomLayoutProperty(
                            prop.prop_name,
                            display_name=prop.metadata["customData"]["uiName"],
                        )

        return frame.apply(props)
