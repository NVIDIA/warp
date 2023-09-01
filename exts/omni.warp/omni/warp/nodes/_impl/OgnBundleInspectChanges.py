# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Node inspecting the changes made to an attribute."""

import traceback
from typing import Union

import numpy as np
import omni.graph.core as og
import warp as wp

from omni.warp.nodes.ogn.OgnBundleInspectChangesDatabase import OgnBundleInspectChangesDatabase


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
        device = wp.get_device("cuda:0")

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            return

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
