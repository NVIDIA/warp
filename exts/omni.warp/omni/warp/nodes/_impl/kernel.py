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

"""Backend implementation for kernel node(s)."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import operator
import os
import tempfile
from enum import IntFlag
from typing import (
    Any,
    Callable,
    Mapping,
    NamedTuple,
    Sequence,
)

import omni.graph.core as og

import warp as wp

from .attributes import (
    ATTR_BUNDLE_TYPE,
    attr_cast_array_to_warp,
    attr_get_base_name,
    attr_get_name,
    attr_join_name,
)
from .common import (
    IntEnum,
    get_warp_type_from_data_type_name,
    type_convert_og_to_warp,
)

_ATTR_PORT_TYPE_INPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT
_ATTR_PORT_TYPE_OUTPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT

EXPLICIT_SOURCE = "explicit"


#   Enumerators
# ------------------------------------------------------------------------------


class UserAttributesEvent(IntFlag):
    """User attributes event."""

    NONE = 0
    CREATED = 1 << 0
    REMOVED = 1 << 1


class OutputArrayShapeSource(IntEnum):
    """Method to infer the shape of output attribute arrays."""

    AS_INPUT_OR_AS_KERNEL = (0, "as input if any, or as kernel")
    AS_KERNEL = (1, "as kernel")


class OutputBundleTypeSource(IntEnum):
    """Method to infer the type of output attribute bundles."""

    AS_INPUT = (0, "as input if any")
    AS_INPUT_OR_EXPLICIT = (1, "as input if any, or explicit")
    EXPLICIT = (2, "explicit")


class ArrayAttributeFormat(IntEnum):
    """Format describing how attribute arrays are defined on the node."""

    RAW = (0, "raw")
    BUNDLE = (1, "bundle")


#   User Attributes Description
# ------------------------------------------------------------------------------


class UserAttributeDesc(NamedTuple):
    """Description of an attribute added dynamically by users through the UI.

    This struct is what the Attribute Editor UI passes to the node in order to
    communicate any attribute metadata.
    """

    port_type: og.AttributePortType
    base_name: str
    data_type_name: str
    is_array: bool
    array_format: ArrayAttributeFormat
    array_shape_source: OutputArrayShapeSource | None
    optional: bool

    @classmethod
    def deserialize(
        cls,
        data: Mapping[str:Any],
    ) -> UserAttributeDesc | None:
        """Creates a new instance based on a serialized representation."""
        # Retrieve the port type. It's invalid not to have any set.
        port_type = data.get("port_type")
        if port_type is None:
            return None
        port_type = og.AttributePortType(port_type)

        # Define sensible default values.
        # Although this class requires all of its member values to be explicitly
        # defined upon initialization, it's possible that the incoming data was
        # serialized with an older version of this class, in which case we might
        # want to try filling any gap.
        values = {
            "array_format": ArrayAttributeFormat.RAW,
            "array_shape_source": (
                OutputArrayShapeSource.AS_INPUT_OR_AS_KERNEL if port_type == _ATTR_PORT_TYPE_OUTPUT else None
            ),
            "optional": False,
        }

        # Override the default values with the incoming data.
        values.update({k: v for k, v in data.items() if k in cls._fields})

        # Ensure that the member values are set using their rightful types.
        values.update(
            {
                "port_type": port_type,
                "array_format": ArrayAttributeFormat(values["array_format"]),
                "array_shape_source": (
                    None
                    if values["array_shape_source"] is None
                    else OutputArrayShapeSource(values["array_shape_source"])
                ),
            }
        )

        try:
            # This might error in case some members are still missing.
            return cls(**values)
        except TypeError:
            return None

    @property
    def name(self) -> str:
        """Retrieves the attribute's name prefixed with its port type."""
        return attr_join_name(self.port_type, self.base_name)

    @property
    def type(self) -> og.Attribute:
        """Retrieves OmniGraph's attribute type."""
        return og.AttributeType.type_from_sdf_type_name(self.type_name)

    @property
    def type_name(self) -> str:
        """Retrieves OmniGraph's attribute type name."""
        if self.is_array:
            return f"{self.data_type_name}[]"

        return self.data_type_name

    def serialize(self) -> Mapping[str:Any]:
        """Converts this instance into a serialized representation."""
        return self._replace(
            port_type=int(self.port_type),
        )._asdict()


def deserialize_user_attribute_descs(
    data: str,
) -> Mapping[str, UserAttributeDesc]:
    """Deserializes a string into a mapping of (name, desc)."""
    descs = {attr_join_name(x["port_type"], x["base_name"]): UserAttributeDesc.deserialize(x) for x in json.loads(data)}

    # Filter out any invalid description.
    return {k: v for k, v in descs.items() if v is not None}


def serialize_user_attribute_descs(
    descs: Mapping[str, UserAttributeDesc],
) -> str:
    """Serializes a mapping of (name, desc) into a string."""
    return json.dumps(tuple(x.serialize() for x in descs.values()))


#   User Attributes Information
# ------------------------------------------------------------------------------


class OutputAttributeInfo(NamedTuple):
    """Information relating to an output node attribute."""

    array_shape_source: OutputArrayShapeSource | None
    bundle_type_source: OutputBundleTypeSource | None
    bundle_type_explicit: str | None = None


class AttributeInfo(NamedTuple):
    """Information relating to a node attribute.

    This struct contains all the metadata required by the node to initialize
    and evaluate. This includes compiling the kernel and initializing the Inputs
    and Outputs structs that are then passed to the kernel as parameters.

    We don't directly store the array shape, if any, since it is possible that
    it might vary between each evaluation of the node's compute. Instead,
    we store which method to use to infer the array's shape and let the node
    determine the actual shape during each compute step.

    Note
    ----

    The `warp_type` member represents the type of the kernel parameter
    corresdonding to that attribute. If the attribute is a bundle, then it is
    expected to be a `wp.struct` holding the values of the bundle, unless
    the bundle is of type :class:`Array`, in which case `warp_type` should be
    a standard `wp.array`.
    """

    port_type: og.AttributePortType
    base_name: str
    og_type: og.Type
    warp_type: type
    output: OutputAttributeInfo | None = None

    @property
    def name(self) -> str:
        return attr_join_name(self.port_type, self.base_name)

    @property
    def og_data_type(self) -> og.Type:
        return og.Type(
            self.og_type.base_type,
            tuple_count=self.og_type.tuple_count,
            array_depth=0,
            role=self.og_type.role,
        )

    @property
    def is_array(self) -> bool:
        return self.og_type.array_depth > 0

    @property
    def is_bundle(self) -> bool:
        return self.og_type == ATTR_BUNDLE_TYPE

    @property
    def dim_count(self) -> int:
        if self.is_array:
            return self.warp_type.ndim

        return 0

    @property
    def warp_data_type(self) -> type:
        if self.is_array:
            return self.warp_type.dtype

        return self.warp_type

    @property
    def warp_type_name(self) -> str:
        if self.is_bundle:
            return self.warp_type.cls.__name__

        return get_warp_type_from_data_type_name(
            self.warp_data_type.__name__,
            dim_count=self.dim_count,
            as_str=True,
        )

    @property
    def warp_data_type_name(self) -> str:
        if self.is_bundle:
            return self.warp_type.cls.__name__

        return get_warp_type_from_data_type_name(
            self.warp_data_type.__name__,
            dim_count=0,
            as_str=True,
        )


def gather_attribute_infos(
    node: og.Node,
    db_inputs: Any,
    db_outputs: Any,
    attr_descs: Mapping[str, UserAttributeDesc],
    kernel_dim_count: int,
) -> Mapping[og.AttributePortType, tuple[AttributeInfo, ...]]:
    """Gathers the information for each user attribute.

    See also: :class:`AttributeInfo`.
    """

    def extract_partial_info_from_attr(attr: og.Attribute) -> tuple[Any, ...]:
        """Extract a partial information set from an attribute."""
        name = attr_get_name(attr)
        base_name = attr_get_base_name(attr)
        og_type = attr.get_resolved_type()
        is_array = og_type.array_depth > 0
        return (name, base_name, og_type, is_array)

    # Retrieve the user attributes defined on the node.
    attrs = tuple(x for x in node.get_attributes() if x.is_dynamic())

    # Gather the information for the input attributes.
    input_attr_infos = []
    for attr in attrs:
        if attr.get_port_type() != _ATTR_PORT_TYPE_INPUT:
            continue

        (name, base_name, og_type, is_array) = extract_partial_info_from_attr(attr)

        og_data_type = og.Type(
            og_type.base_type,
            tuple_count=og_type.tuple_count,
            array_depth=0,
            role=og_type.role,
        )

        input_attr_infos.append(
            AttributeInfo(
                port_type=_ATTR_PORT_TYPE_INPUT,
                base_name=base_name,
                og_type=og_type,
                warp_type=type_convert_og_to_warp(
                    og_data_type,
                    dim_count=int(is_array),
                ),
            )
        )

    # Gather the information for the output attributes.
    output_attr_infos = []
    for attr in attrs:
        if attr.get_port_type() != _ATTR_PORT_TYPE_OUTPUT:
            continue

        (name, base_name, og_type, is_array) = extract_partial_info_from_attr(attr)

        desc = attr_descs.get(name)
        if desc is None:
            # Fallback for nodes created before the attribute description
            # feature was implemented.
            array_shape_source = OutputArrayShapeSource.AS_INPUT_OR_AS_KERNEL
        else:
            array_shape_source = desc.array_shape_source

        if array_shape_source == OutputArrayShapeSource.AS_INPUT_OR_AS_KERNEL:
            # Check if we have an input attribute with a matching name,
            # in which case we use its array dimension count.
            try:
                dim_count = next(x.dim_count for x in input_attr_infos if x.base_name == base_name)
            except StopIteration:
                # Fallback to using the kernel's dimension count.
                dim_count = kernel_dim_count
        elif array_shape_source == OutputArrayShapeSource.AS_KERNEL:
            dim_count = kernel_dim_count
        else:
            raise AssertionError(f"Unexpected array shape source method '{array_shape_source}'.")

        og_data_type = og.Type(
            og_type.base_type,
            tuple_count=og_type.tuple_count,
            array_depth=0,
            role=og_type.role,
        )

        output_attr_infos.append(
            AttributeInfo(
                port_type=_ATTR_PORT_TYPE_OUTPUT,
                base_name=base_name,
                og_type=og_type,
                warp_type=type_convert_og_to_warp(
                    og_data_type,
                    dim_count=dim_count,
                ),
                output=OutputAttributeInfo(
                    array_shape_source=array_shape_source,
                    bundle_type_source=OutputBundleTypeSource.AS_INPUT,
                ),
            )
        )

    return {
        _ATTR_PORT_TYPE_INPUT: tuple(input_attr_infos),
        _ATTR_PORT_TYPE_OUTPUT: tuple(output_attr_infos),
    }


#   Kernel Code
# ------------------------------------------------------------------------------

_STRUCT_DECLARATION_CODE_TEMPLATE = """@wp.struct
class {name}:
{members}
"""


def _generate_struct_declaration_code(warp_struct: wp.struct) -> str:
    """Generates the code declaring a Warp struct."""
    lines = []
    for label, var in warp_struct.vars.items():
        warp_type = var.type
        if isinstance(warp_type, wp.array):
            warp_data_type = warp_type.dtype
            dim_count = warp_type.ndim
        else:
            warp_data_type = warp_type
            dim_count = 0

        warp_type_name = get_warp_type_from_data_type_name(
            warp_data_type.__name__,
            dim_count=dim_count,
            as_str=True,
        )
        lines.append(f"    {label}: {warp_type_name}")

    return _STRUCT_DECLARATION_CODE_TEMPLATE.format(
        name=warp_struct.cls.__name__,
        members="\n".join(lines),
    )


_HEADER_CODE_TEMPLATE = """import warp as wp
{declarations}
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
    attr_infos: Mapping[og.AttributePortType, tuple[AttributeInfo, ...]],
) -> str:
    """Generates the code header based on the node's attributes."""
    # Retrieve all the Warp struct types corresponding to bundle attributes.
    struct_types = {x.warp_type_name: x.warp_type for _, v in attr_infos.items() for x in v if x.is_bundle}

    # Generate the code that declares the Warp structs found.
    declarations = [""]
    declarations.extend(_generate_struct_declaration_code(x) for _, x in struct_types.items())

    # Generate the lines of code declaring the members for each port type.
    lines = {k: tuple(f"    {x.base_name}: {x.warp_type_name}" for x in v) for k, v in attr_infos.items()}

    # Return the template code populated with the members.
    return _HEADER_CODE_TEMPLATE.format(
        declarations="\n".join(declarations),
        inputs="\n".join(lines.get(_ATTR_PORT_TYPE_INPUT, ())),
        outputs="\n".join(lines.get(_ATTR_PORT_TYPE_OUTPUT, ())),
    )


def _get_user_code(code_provider: str, code_str: str, code_file: str) -> str:
    """Retrieves the code provided by the user."""
    if code_provider == "embedded":
        return code_str

    if code_provider == "file":
        with open(code_file) as f:
            return f.read()

    raise AssertionError(f"Unexpected code provider '{code_provider}'.")


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
    attr_infos: Mapping[og.AttributePortType, tuple[AttributeInfo, ...]],
    code_provider: str,
    code_str: str,
    code_file: str,
) -> wp.context.Module:
    # Ensure that all output parameters are arrays. Writing to non-array
    # types is not supported as per CUDA's design.
    invalid_attrs = tuple(x.name for x in attr_infos[_ATTR_PORT_TYPE_OUTPUT] if not x.is_array and not x.is_bundle)
    if invalid_attrs:
        raise RuntimeError(
            "Output attributes are required to be arrays or bundles but the following attributes are not: {}.".format(
                ", ".join(invalid_attrs)
            )
        )

    # Retrieve the kernel code to evaluate.
    code_header = _generate_header_code(attr_infos)
    user_code = _get_user_code(code_provider, code_str, code_file)
    code = f"{code_header}\n{user_code}"

    # Create a Python module made of the kernel code.
    # We try to keep its name unique to ensure that it's not clashing with
    # other kernel modules from the same session.
    uid = hashlib.blake2b(bytes(code, encoding="utf-8"), digest_size=8)
    module_name = f"warp-kernelnode-{uid.hexdigest()}"
    kernel_module = _load_code_as_module(code, module_name)

    # Validate the module's contents.
    if not hasattr(kernel_module, "compute"):
        raise RuntimeError("The code must define a kernel function named 'compute'.")
    if not isinstance(kernel_module.compute, wp.context.Kernel):
        raise RuntimeError("The 'compute' function must be decorated with '@wp.kernel'.")

    # Configure warp to only compute the forward pass.
    wp.set_module_options({"enable_backward": False}, module=kernel_module)

    return kernel_module


#   Data I/O
# ------------------------------------------------------------------------------


def _infer_output_array_shape(
    attr_info: AttributeInfo,
    input_attr_infos: tuple[AttributeInfo, ...],
    kernel_inputs: Any,
    kernel_shape: Sequence[int],
) -> tuple[int, ...]:
    if attr_info.output.array_shape_source == OutputArrayShapeSource.AS_INPUT_OR_AS_KERNEL:
        # Check if we have an input attribute with a matching name,
        # in which case we use its array shape.
        try:
            ref_attr_base_name = next(
                x.base_name
                for x in input_attr_infos
                if (x.base_name == attr_info.base_name and x.is_array and x.dim_count == attr_info.dim_count)
            )
            return getattr(kernel_inputs, ref_attr_base_name).shape
        except StopIteration:
            # Fallback to using the kernel's shape.
            return tuple(kernel_shape)

    if attr_info.output.array_shape_source == OutputArrayShapeSource.AS_KERNEL:
        return tuple(kernel_shape)

    raise AssertionError(f"Unexpected array shape source method '{attr_info.output.array_shape_source}'.")


class KernelArgsConfig(NamedTuple):
    """Configuration for resolving kernel arguments."""

    input_bundle_handlers: Mapping[str, Callable] | None = None
    output_bundle_handlers: Mapping[str, Callable] | None = None


def get_kernel_args(
    db_inputs: Any,
    db_outputs: Any,
    attr_infos: Mapping[og.AttributePortType, tuple[AttributeInfo, ...]],
    kernel_module: Any,
    kernel_shape: Sequence[int],
    device: wp.context.Device | None = None,
    config: KernelArgsConfig | None = None,
) -> tuple[Any, Any]:
    """Retrieves the in/out argument values to pass to the kernel."""
    if device is None:
        device = wp.get_device()

    if config is None:
        config = KernelArgsConfig()

    # Initialize the kernel's input data.
    inputs = kernel_module.Inputs()
    for info in attr_infos[_ATTR_PORT_TYPE_INPUT]:
        # Retrieve the input attribute value and cast it to
        # the corresponding warp type.
        if info.is_array:
            value = getattr(db_inputs, info.base_name)

            # The array value might define 2 dimensions when tuples such as
            # wp.vec3 are used as data type, so we preserve only the first
            # dimension to retrieve the actual shape since OmniGraph only
            # supports 1D arrays anyways.
            shape = value.shape[:1]

            value = attr_cast_array_to_warp(
                value,
                info.warp_data_type,
                shape,
                device,
            )
        elif info.is_bundle:
            raise NotImplementedError("Bundle attributes are not yet supported.")
        else:
            value = getattr(db_inputs, info.base_name)

        # Store the result in the inputs struct.
        setattr(inputs, info.base_name, value)

    # Initialize the kernel's output data.
    outputs = kernel_module.Outputs()
    for info in attr_infos[_ATTR_PORT_TYPE_OUTPUT]:
        # Retrieve the output attribute value and cast it to the corresponding
        # warp type.
        if info.is_array:
            shape = _infer_output_array_shape(
                info,
                attr_infos[_ATTR_PORT_TYPE_INPUT],
                inputs,
                kernel_shape,
            )

            # Allocate a buffer for the array.
            size = functools.reduce(operator.mul, shape)
            setattr(db_outputs, f"{info.base_name}_size", size)

            value = getattr(db_outputs, info.base_name)
            value = attr_cast_array_to_warp(
                value,
                info.warp_data_type,
                shape,
                device,
            )
        elif info.is_bundle:
            raise NotImplementedError("Bundle attributes are not yet supported.")
        else:
            raise AssertionError("Output attributes are expected to be arrays or bundles.")

        # Store the result in the outputs struct.
        setattr(outputs, info.base_name, value)

    return (inputs, outputs)


def write_output_attrs(
    db_outputs: Any,
    attr_infos: Mapping[og.AttributePortType, tuple[AttributeInfo, ...]],
    kernel_outputs: Any,
    device: wp.context.Device | None = None,
) -> None:
    """Writes the output values to the node's attributes."""
    if device is None:
        device = wp.get_device()

    if device.is_cuda:
        # CUDA attribute arrays are directly being written to by Warp.
        return

    for info in attr_infos[_ATTR_PORT_TYPE_OUTPUT]:
        value = getattr(kernel_outputs, info.base_name)
        setattr(db_outputs, info.base_name, value)


#   Validation
# ------------------------------------------------------------------------------


def validate_input_arrays(
    node: og.Node,
    attr_infos: Mapping[og.AttributePortType, tuple[AttributeInfo, ...]],
    kernel_inputs: Any,
) -> None:
    """Validates array input attributes."""
    for info in attr_infos[_ATTR_PORT_TYPE_INPUT]:
        value = getattr(kernel_inputs, info.base_name)
        if not isinstance(value, wp.array):
            continue

        # Ensure that all array input attributes are not NULL,
        # unless they are set as being optional.
        attr = og.Controller.attribute(info.name, node)
        if not attr.is_optional_for_compute and not value.ptr:
            raise RuntimeError(f"Empty value for non-optional attribute '{info.name}'.")


#   Node's Internal State
# ------------------------------------------------------------------------------


class InternalStateBase:
    """Base class for the node's internal state."""

    def __init__(self) -> None:
        self._code_provider = None
        self._code_str = None
        self._code_file = None
        self._code_file_timestamp = None

        self.attr_infos = None
        self.kernel_module = None

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
            if self.kernel_module is None or UserAttributesEvent.REMOVED & db.state.userAttrsEvent:
                return True
        else:
            # If something previously went wrong, we always recompile the kernel
            # when attributes are edited, in case it might fix code that
            # errored out due to referencing a non-existing attribute.
            if db.state.userAttrsEvent != UserAttributesEvent.NONE:
                return True

        if self.attr_infos is None:
            return True

        if self._code_provider != db.inputs.codeProvider:
            return True

        if self._code_provider == "embedded":
            if self._code_str != db.inputs.codeStr:
                return True
        elif self._code_provider == "file":
            if self._code_file != db.inputs.codeFile or (
                check_file_modified_time and (self._code_file_timestamp != os.path.getmtime(self._code_file))
            ):
                return True
        else:
            raise AssertionError(f"Unexpected code provider '{self._code_provider}'.")

        return False

    def initialize(self, db: Any) -> bool:
        """Initialize the internal state and recompile the kernel."""
        # Cache the node attribute values relevant to this internal state.
        # They're the ones used to check whether this state is outdated or not.
        self._code_provider = db.inputs.codeProvider
        self._code_str = db.inputs.codeStr
        self._code_file = db.inputs.codeFile

        return True
