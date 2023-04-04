# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import operator
from typing import (
    NamedTuple,
    Sequence,
    Tuple,
)

import omni.graph.core as og
from omni.warp.scripts.attributes import (
    convert_og_type_to_warp,
    insert_bundle_attr,
)
from omni.warp.scripts.common import (
    BUNDLE_TYPE_ATTR_NAME,
    BUNDLE_TYPE_OG_TYPE,
    get_annotations,
)
import warp as wp

#   Warp Type
# ------------------------------------------------------------------------------

def get_warp_type(data_dtype: type, data_ndim: int) -> type:
    """Retrieves the Warp type matching the array bundle type."""
    return wp.array(dtype=data_dtype, ndim=data_ndim)

#   Bundle Accessor
# ------------------------------------------------------------------------------

_ACCESSOR_ATTR_NAME_FMT = "{}_attr"

class Accessor(NamedTuple):
    """Accessor for an existing bundle."""

    BUNDLE_TYPE = "omni.warp:Array"

    bundle: og.BundleContents
    data_attr: og.RuntimeAttribute
    shape_attr: og.RuntimeAttribute

    @property
    def data(self) -> wp.array:
        return wp.from_ptr(
            self.data_attr.gpu_value.memory,
            self.data_attr.size,
            dtype=convert_og_type_to_warp(self.data_attr.type),
            shape=self.shape,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.shape_attr.cpu_value.tolist())

    def get_warp_type(self) -> type:
        """Retrieves the Warp type corresponding to this bundle."""
        return get_warp_type(
            convert_og_type_to_warp(self.data_attr.type),
            len(self.shape),
        )

    def get_warp_instance(self) -> wp.array:
        """Retrieves the Warp instance corresponding to this bundle."""
        inst = self.get_warp_type()()
        for field in get_annotations(inst):
            setattr(inst, field, getattr(self, field))

        return inst

def create_bundle(
    bundle: og.BundleContents,
    og_data_type: og.Type,
    shape: Sequence[int],
) -> Accessor:
    """Creates a bundle describing an array."""
    # Create the attribute describing this bundle's type.
    bundle_type_attr = insert_bundle_attr(
        bundle,
        BUNDLE_TYPE_ATTR_NAME,
        BUNDLE_TYPE_OG_TYPE,
    )
    bundle_type_attr.cpu_value = Accessor.BUNDLE_TYPE

    # Create the bundle attributes corresponding to the Warp type fields.
    data_attr = insert_bundle_attr(
        bundle,
        "data",
        og.Type(
            og_data_type.base_type,
            tuple_count=og_data_type.tuple_count,
            array_depth=len(shape),
            role=og_data_type.role,
        ),
    )
    data_attr.size = functools.reduce(operator.mul, shape)

    shape_attr = insert_bundle_attr(
        bundle,
        "shape",
        og.Type(
            og.BaseDataType.INT,
            tuple_count=1,
            array_depth=1,
            role=og.AttributeRole.NONE,
        ),
    )
    shape_attr.size = len(shape)
    shape_attr.cpu_value = shape

    # Initialize the accessor instance from the bundle attributes.
    accessor = Accessor(
        bundle,
        data_attr=data_attr,
        shape_attr=shape_attr,
    )

    return accessor

def read_bundle(bundle: og.BundleContents) -> Accessor:
    """Reads a bundle describing an array."""
    # Retrieve the attribute describing this bundle's type.
    bundle_type_attr = bundle.attribute_by_name(BUNDLE_TYPE_ATTR_NAME)

    # Validate the data.
    if (
        bundle_type_attr is not None
        and bundle_type_attr.cpu_value != Accessor.BUNDLE_TYPE
    ):
        raise RuntimeError(
            "Trying to read a bundle of type '{}' "
            "when expecting one of type '{}'."
            .format(bundle_type_attr.cpu_value, Accessor.BUNDLE_TYPE)
        )

    # Initialize the accessor instance from the bundle attributes.
    return Accessor(
        bundle,
        data_attr=bundle.attribute_by_name("data"),
        shape_attr=bundle.attribute_by_name("shape"),
    )
