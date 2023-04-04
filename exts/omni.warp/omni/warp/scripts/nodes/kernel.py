# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from enum import IntFlag
import functools
import hashlib
import importlib.util
import json
import operator
import os
import tempfile
from typing import (
    Any,
    Mapping,
    NamedTuple,
    Sequence,
    Tuple,
)

import omni.graph.core as og
import warp as wp

from omni.warp.scripts.attributes import (
    cast_array_attr_value_to_warp,
    convert_sdf_type_name_to_warp,
    join_attr_name,
)

MAX_DIMENSIONS = 4

_ATTR_PORT_TYPE_INPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT
_ATTR_PORT_TYPE_OUTPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT

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

#   Kernel Code
# ------------------------------------------------------------------------------

_HEADER_CODE_TEMPLATE = """import warp as wp

@wp.struct
class Inputs:
{inputs}
    pass

@wp.struct
class Outputs:
{outputs}
    pass
"""

def _generate_header_code(
    attrs: Sequence[og.Attribute],
    attr_descs: Mapping[str, UserAttributeDesc],
) -> str:
    """Generates the code header based on the node's attributes."""
    # Convert all the inputs/outputs attributes into warp members.
    params = {}
    for attr in attrs:
        attr_type = attr.get_type_name()
        is_array = attr_type.endswith("[]")
        if is_array:
            attr_data_type = attr_type[:-2]
        else:
            attr_data_type = attr_type
        warp_type = convert_sdf_type_name_to_warp(
            attr_data_type,
            dim_count=int(is_array),
            as_str=True,
        )

        if warp_type is None:
            raise RuntimeError(
                "Unsupported node attribute type '{}'.".format(attr_type)
            )

        params.setdefault(attr.get_port_type(), []).append(
            (
                attr.get_name().split(":")[-1],
                warp_type,
            ),
        )

    # Generate the lines of code declaring the members for each port type.
    members = {
        port_type: "\n".join("    {}: {}".format(*x) for x in items)
        for port_type, items in params.items()
    }

    # Return the template code populated with the members.
    return _HEADER_CODE_TEMPLATE.format(
        inputs=members.get(_ATTR_PORT_TYPE_INPUT, ""),
        outputs=members.get(_ATTR_PORT_TYPE_OUTPUT, ""),
    )

def _get_user_code(code_provider: str, code_str: str, code_file: str) -> str:
    """Retrieves the code provided by the user."""
    if code_provider == "embedded":
        return code_str

    if code_provider == "file":
        with open(code_file, "r") as f:
            return f.read()

    assert False, "Unexpected code provider '{}'.".format(code_provider)

#   Kernel Module
# ------------------------------------------------------------------------------

def _load_code_as_module(code: str, name: str) -> Any:
    """Loads a Python module from the given source code."""
    # It's possible to use the `exec()` built-in function to create and
    # populate a Python module with the source code defined in a string,
    # however warp requires access to the source code of the kernel's
    # function, which is only available when the original source file
    # pointed by the function attribute `__code__.co_filename` can
    # be opened to read the lines corresponding to that function.
    # As such, we must write the source code into a temporary file
    # on disk before importing it as a module and having the function
    # turned into a kernel by warp's mechanism.

    # Create a temporary file.
    file, file_path = tempfile.mkstemp(suffix=".py")

    try:
        # Save the embedded code into the temporary file.
        with os.fdopen(file, "w") as f:
            f.write(code)

        # Import the temporary file as a Python module.
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        # The resulting Python module is stored into memory as a bytcode
        # object and the kernel function has already been parsed by warp
        # as long as it was correctly decorated, so it's now safe to
        # clean-up the temporary file.
        os.remove(file_path)

    return module

def initialize_kernel_module(
    attrs: Sequence[og.Attribute],
    attr_descs: Mapping[str, UserAttributeDesc],
    code_provider: str,
    code_str: str,
    code_file: str,
) -> wp.context.Module:
    """Initializes the kernel's module based on the node's attributes."""
    # Retrieve the kernel code to evaluate.
    header_code = _generate_header_code(attrs, attr_descs)
    user_code = _get_user_code(code_provider, code_str, code_file)
    code = "{}\n{}".format(header_code, user_code)

    # Create a Python module made of the kernel code.
    # We try to keep its name unique to ensure that it's not clashing with
    # other kernel modules from the same session.
    uid = hashlib.blake2b(bytes(code, encoding="utf-8"), digest_size=8)
    module_name = "warp-kernelnode-{}".format(uid.hexdigest())
    kernel_module = _load_code_as_module(code, module_name)

    # Validate the module's contents.
    if not hasattr(kernel_module, "compute"):
        raise RuntimeError(
            "The code must define a kernel function named 'compute'."
        )
    if not isinstance(kernel_module.compute, wp.context.Kernel):
        raise RuntimeError(
            "The 'compute' function must be decorated with '@wp.kernel'."
        )

    # Configure warp to only compute the forward pass.
    wp.set_module_options({"enable_backward": False}, module=kernel_module)

    return kernel_module

#   Data I/O
# ------------------------------------------------------------------------------

def _are_array_annotations_equal(
    annotation_1: Any,
    annotation_2: Any,
) -> bool:
    """Checks whether two array annotations are equal."""
    assert isinstance(annotation_1, wp.array)
    assert isinstance(annotation_2, wp.array)
    return (
        annotation_1.dtype == annotation_2.dtype
        and annotation_1.ndim == annotation_2.ndim
    )

def get_kernel_args(
    db_inputs: Any,
    db_outputs: Any,
    module: Any,
    annotations: Mapping[og.AttributePortType, Sequence[Tuple[str, Any]]],
    dims: Sequence[int],
    device: wp.context.Device,
) -> Tuple[Any, Any]:
    """Retrieves the in/out argument values to pass to the kernel."""
    # Initialize the kernel's input data.
    inputs = module.Inputs()
    for name, warp_annotation in annotations[_ATTR_PORT_TYPE_INPUT].items():
        # Retrieve the input attribute value and cast it to the corresponding
        # warp type if is is an array.
        value = getattr(db_inputs, name)
        if isinstance(warp_annotation, wp.array):
            value = cast_array_attr_value_to_warp(
                value,
                warp_annotation.dtype,
                (value.shape[0],),
                device,
            )

        # Store the result in the inputs struct.
        setattr(inputs, name, value)

    # Initialize the kernel's output data.
    outputs = module.Outputs()
    for name, warp_annotation in annotations[_ATTR_PORT_TYPE_OUTPUT].items():
        assert isinstance(warp_annotation, wp.array)

        # Retrieve the size of the array to allocate.
        ref_annotation = annotations[_ATTR_PORT_TYPE_INPUT].get(name)
        if (
            isinstance(ref_annotation, wp.array)
            and _are_array_annotations_equal(warp_annotation, ref_annotation)
        ):
            # If there's an existing input with the same name and type,
            # we allocate a new array matching the input's length.
            size = len(getattr(inputs, name))
        else:
            # Fallback to allocate an array matching the kernel's dimensions.
            size = functools.reduce(operator.mul, dims)

        # Allocate the array.
        setattr(db_outputs, "{}_size".format(name), size)

        # Retrieve the output attribute value and cast it to the corresponding
        # warp type.
        value = getattr(db_outputs, name)
        value = cast_array_attr_value_to_warp(
            value,
            warp_annotation.dtype,
            (size,),
            device,
        )

        # Store the result in the outputs struct.
        setattr(outputs, name, value)

    return (inputs, outputs)

#   Validation
# ------------------------------------------------------------------------------

def validate_input_arrays(
    node: og.Node,
    annotations: Mapping[og.AttributePortType, Sequence[Tuple[str, Any]]],
    kernel_inputs: Any,
) -> None:
    """Validates array input attributes."""
    # Ensure that all array input attributes are not NULL, unless they are set
    # as being optional.
    # Note that adding a new non-optional array attribute might still cause
    # the compute to succeed since the kernel recompilation is delayed until
    # `InternalState.needs_initialization()` requests it, meaning that the new
    # attribute won't show up as a kernel annotation just yet.
    for attr_name in annotations[_ATTR_PORT_TYPE_INPUT]:
        value = getattr(kernel_inputs, attr_name)
        if not isinstance(value, wp.array):
            continue

        attr = og.Controller.attribute("inputs:{}".format(attr_name), node)
        if not attr.is_optional_for_compute and not value.ptr:
            raise RuntimeError(
                "Empty value for non-optional attribute 'inputs:{}'."
                .format(attr_name)
            )

#   Node's Internal State
# ------------------------------------------------------------------------------

class InternalStateBase:
    """Base class for the node's internal state."""

    def __init__(self) -> None:
        self._code_provider = None
        self._code_str = None
        self._code_file = None
        self._code_file_timestamp = None

        self.kernel_module = None
        self.kernel_annotations = None

        self.is_valid = False

    def needs_initialization(
        self,
        db: Any,
        check_file_modified_time: bool,
    ) -> bool:
        """Checks if the internal state needs to be (re)initialized."""
        if self.is_valid:
            # If everything is in order, we only need to recompile the kernel
            # when attributes are removed, since adding new attributes is not
            # a breaking change.
            if (
                self.kernel_module is None
                or self.kernel_annotations is None
                or UserAttributesEvent.REMOVED & db.state.userAttrsEvent
            ):
                return True
        else:
            # If something previously went wrong, we always recompile the kernel
            # when attributes are edited, in case it might fix code that
            # errored out due to referencing a non-existing attribute.
            if db.state.userAttrsEvent != UserAttributesEvent.NONE:
                return True

        if self._code_provider != db.inputs.codeProvider:
            return True

        if self._code_provider == "embedded":
            if self._code_str != db.inputs.codeStr:
                return True
        elif self._code_provider == "file":
            if (
                self._code_file != db.inputs.codeFile
                or (
                    check_file_modified_time
                    and (
                        self._code_file_timestamp
                        != os.path.getmtime(self._code_file)
                    )
                )
            ):
                return True
        else:
            assert False, (
                "Unexpected code provider '{}'.".format(self._code_provider),
            )

        return False

    def initialize(self, db: Any) -> bool:
        """Initialize the internal state and recompile the kernel."""
        # Cache the node attribute values relevant to this internal state.
        # They're the ones used to check whether this state is outdated or not.
        self._code_provider = db.inputs.codeProvider
        self._code_str = db.inputs.codeStr
        self._code_file = db.inputs.codeFile

        return True
