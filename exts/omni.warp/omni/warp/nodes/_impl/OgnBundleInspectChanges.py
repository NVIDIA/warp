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

"""Node inspecting the changes made to an attribute."""

import traceback
from typing import Union

import numpy as np
import omni.graph.core as og
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnBundleInspectChangesDatabase import OgnBundleInspectChangesDatabase

import warp as wp

_ATTR_NAMES_PER_PRIM_TYPE = {
    "Mesh": (
        "points",
        "faceVertexCounts",
        "faceVertexIndices",
        "primvars:normals",
        "primvars:st",
    ),
    "Points": (
        "points",
        "widths",
    ),
}


#   Helpers
# ------------------------------------------------------------------------------


def get_cpu_array(
    attr: og.AttributeData,
    read_only: bool = True,
) -> Union[np.ndarray, str]:
    """Retrieves the value of an array attribute living on the CPU."""
    return attr.get_array(
        on_gpu=False,
        get_for_write=not read_only,
        reserved_element_count=0 if read_only else attr.size(),
    )


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnBundleInspectChangesDatabase) -> None:
    """Evaluates the node."""
    if not db.inputs.bundle.valid or not db.outputs.bundle.valid:
        return

    db.outputs.bundle = db.inputs.bundle

    attrs_changed = []
    with db.inputs.bundle.changes() as bundle_changes:
        for bundle in db.inputs.bundle.bundle.get_child_bundles():
            attr = bundle.get_attribute_by_name("sourcePrimType")
            prim_type = get_cpu_array(attr)
            attr_names = _ATTR_NAMES_PER_PRIM_TYPE.get(prim_type)
            for attr_name in attr_names:
                if attr_name in attrs_changed:
                    continue

                attr = bundle.get_attribute_by_name(attr_name)
                changes = bundle_changes.get_change(attr)
                if changes == og.BundleChangeType.NONE:
                    continue

                attrs_changed.append(attr_name)

    db.outputs.attrsChanged = " ".join(attrs_changed)
    db.outputs.topologyChanged = db.outputs.attrsChanged != "points"


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnBundleInspectChanges:
    """Node."""

    @staticmethod
    def compute(db: OgnBundleInspectChangesDatabase) -> None:
        device = omni.warp.nodes.device_get_cuda_compute()

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            return

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
