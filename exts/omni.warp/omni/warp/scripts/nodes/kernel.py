# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from enum import IntFlag
import json
from typing import (
    Any,
    Mapping,
    NamedTuple,
)

import omni.graph.core as og

from omni.warp.scripts.attributes import join_attr_name

MAX_DIMENSIONS = 4

#   User Attribute Events
# ------------------------------------------------------------------------------

class UserAttributesEvent(IntFlag):
    """User attributes event."""

    NONE    = 0
    CREATED = 1 << 0
    REMOVED = 1 << 1

#   User Attributes Description
# ------------------------------------------------------------------------------

class UserAttributeDesc(NamedTuple):
    """Description of an attribute added dynamically by users through the UI.

    This struct is what the Attribute Editor UI passes to the node in order to
    communicate any attribute metadata.
    """

    port_type: og.AttributePortType
    base_name: str
    type_name: str
    optional: bool

    @classmethod
    def deserialize(cls, data: Mapping[str: Any]) -> UserAttributeDesc:
        """Creates a new instance based on a serialized representation."""
        inst = cls(**data)
        return inst._replace(
            port_type=og.AttributePortType(inst.port_type),
        )

    @property
    def name(self) -> str:
        """Retrieves the attribute's name prefixed with its port type."""
        return join_attr_name(self.port_type, self.base_name)

    @property
    def type(self) -> og.Attribute:
        """Retrieves OmniGraph's attribute type."""
        return og.AttributeType.type_from_sdf_type_name(self.type_name)

    def serialize(self) -> Mapping[str: Any]:
        """Converts this instance into a serialized representation."""
        return self._replace(
            port_type=int(self.port_type),
        )._asdict()

def deserialize_user_attribute_descs(
    data: str,
) -> Mapping[str, UserAttributeDesc]:
    """Deserializes a string into a mapping of (name, desc)."""
    return {
        join_attr_name(x["port_type"], x["base_name"]):
            UserAttributeDesc.deserialize(x)
        for x in json.loads(data)
    }

def serialize_user_attribute_descs(
    descs: Mapping[str, UserAttributeDesc],
) -> str:
    """Serializes a mapping of (name, desc) into a string."""
    return json.dumps(tuple(x.serialize() for x in descs.values()))
