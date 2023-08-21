"""Shared helpers for the extension's tests."""

from typing import Optional
import importlib
import os
import tempfile

import numpy as np
import omni.graph.core as og
import omni.graph.tools.ogn as ogn
import warp as wp


def register_node(
    cls: type,
    ogn_definition: str,
    extension: str,
) -> None:
    """Registers a new node type based on its class object and OGN code."""
    # Parse the OGN definition.
    interface_wrapper = ogn.NodeInterfaceWrapper(ogn_definition, extension)

    # Generate the node's `og.Database` class and load it as a module.
    db_definition = ogn.generate_python(
        ogn.GeneratorConfiguration(
            node_file_path=None,
            node_interface=interface_wrapper.node_interface,
            extension=extension,
            module=extension,
            base_name=cls.__name__,
            destination_directory=None,
        ),
    )
    module_name = "{}.{}".format(extension, cls.__name__)
    file, file_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(file, "w") as f:
            f.write(db_definition)

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.remove(file_path)

    # Register the node.
    db_cls = getattr(module, "{}Database".format(cls.__name__))
    db_cls.register(cls)


def bundle_create_attr(
    bundle: og.BundleContents,
    name: str,
    og_type: og.Type,
    size: int = 0,
) -> og.AttributeData:
    """Creates a bundle attribute if it doesn't already exist."""
    if bundle.bundle.get_child_bundle_count() > 0:
        prim_bundle = bundle.bundle.get_child_bundle(0)
    else:
        prim_bundle = bundle.bundle.create_child_bundle("prim0")

    attr = prim_bundle.get_attribute_by_name(name)
    if attr.is_valid() and attr.get_type() == og_type and attr.size() == size:
        return attr

    return prim_bundle.create_attribute(name, og_type, element_count=size)


def bundle_get_attr(
    bundle: og.BundleContents,
    name: str,
) -> Optional[og.AttributeData]:
    """Retrieves a bundle attribute from its name."""
    if bundle.bundle.get_child_bundle_count():
        attr = bundle.bundle.get_child_bundle(0).get_attribute_by_name(name)
    else:
        attr = bundle.bundle.get_attribute_by_name(name)

    if not attr.is_valid():
        return None

    return attr


def attr_set_array(
    attr: og.AttributeData,
    value: wp.array,
    on_gpu: bool = False,
) -> None:
    """Sets the given array data onto an attribute."""
    if on_gpu:
        attr.gpu_ptr_kind = og.PtrToPtrKind.CPU
        (ptr, _) = attr.get_array(
            on_gpu=True,
            get_for_write=True,
            reserved_element_count=attr.size(),
        )
        data = wp.from_ptr(ptr, attr.size(), dtype=value.dtype)
        wp.copy(data, value)
    else:
        attr.set(value.numpy(), on_gpu=False)


def array_are_equal(
    a: wp.array,
    b: wp.array,
) -> bool:
    """Checks whether two arrays are equal."""
    return a.shape == b.shape and a.dtype == a.dtype and np.array_equal(a.numpy(), b.numpy())
