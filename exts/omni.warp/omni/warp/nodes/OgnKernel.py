"""Warp kernel exposed as an Omni Graph node."""

import hashlib
import importlib.util
import os
import tempfile
from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import warp as wp

import omni.graph.core as og
import omni.timeline

from omni.warp.ogn.OgnKernelDatabase import OgnKernelDatabase
from omni.warp.scripts.kernelnode import ATTR_TO_WARP_TYPE

wp.init()

ATTR_PORT_TYPE_INPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT
ATTR_PORT_TYPE_OUTPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT

CODE_HEADER_TEMPLATE = """import warp as wp

@wp.struct
class Inputs:
{inputs}
    pass

@wp.struct
class Outputs:
{outputs}
    pass
"""

#   Internal State
# ------------------------------------------------------------------------------

def get_annotations(obj: Any) -> Mapping[str, Any]:
    """Alternative to `inspect.get_annotations()` for Python 3.9 and older."""
    # See https://docs.python.org/3/howto/annotations.html#accessing-the-annotations-dict-of-an-object-in-python-3-9-and-older
    if isinstance(obj, type):
        return obj.__dict__.get("__annotations__", {})

    return getattr(obj, "__annotations__", {})

def generate_code_header(attrs: Sequence[og.Attribute]) -> str:
    """Generates the code header based on the node's attributes."""
    # Convert all the inputs/outputs attributes into warp members.
    params = {}
    for attr in attrs:
        attr_type = attr.get_type_name()
        warp_type = ATTR_TO_WARP_TYPE.get(attr_type)

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
    return CODE_HEADER_TEMPLATE.format(
        inputs=members.get(ATTR_PORT_TYPE_INPUT, ""),
        outputs=members.get(ATTR_PORT_TYPE_OUTPUT, ""),
    )

def get_user_code(db: OgnKernelDatabase) -> str:
    """Retrieves the code provided by the user."""
    code_provider = db.inputs.codeProvider

    if code_provider == "embedded":
        return db.inputs.codeStr

    if code_provider == "file":
        with open(db.inputs.codeFile, "r") as f:
            return f.read()

    assert False, "Unexpected code provider '{}'.".format(code_provider)

def load_code_as_module(code: str, name: str) -> Any:
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

class InternalState:
    """Internal state for the node."""

    def __init__(self):
        self._attrs = None
        self._code_provider = None
        self._code_str = None
        self._code_file = None
        self._code_file_timestamp = None

        self.kernel_module = None
        self.kernel_annotations = None

    def _is_outdated(
        self,
        db: OgnKernelDatabase,
        check_file_modified_time: bool,
    ) -> bool:
        """Checks if the internal state is outdated."""
        if self._attrs != db.node.get_attributes():
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

    def initialize(
        self,
        db: OgnKernelDatabase,
        check_file_modified_time: bool = False,
    ) -> None:
        """Initialize the internal state if needed."""
        # Check if this internal state is outdated. If not, we can reuse it.
        if not self._is_outdated(db, check_file_modified_time):
            return

        # Cache the node attribute values relevant to this internal state.
        # They're the ones used to check whether this state is outdated or not.
        self._attrs = db.node.get_attributes()
        self._code_provider = db.inputs.codeProvider
        self._code_str = db.inputs.codeStr
        self._code_file = db.inputs.codeFile

        # Retrieve the dynamic user attributes defined on the node.
        attrs = tuple(x for x in self._attrs if x.is_dynamic())

        # Retrieve the kernel code to evaluate.
        code_header = generate_code_header(attrs)
        user_code = get_user_code(db)
        code = "{}\n{}".format(code_header, user_code)

        # Create a Python module made of the kernel code.
        # We try to keep its name unique to ensure that it's not clashing with
        # other kernel modules from the same session.
        uid = hashlib.blake2b(bytes(code, encoding="utf-8"), digest_size=8)
        module_name = "warp-kernelnode-{}".format(uid.hexdigest())
        kernel_module = load_code_as_module(code, module_name)

        # Validate the module's contents.
        if not hasattr(kernel_module, "compute"):
            raise RuntimeError(
                "The code must define a kernel function named 'compute'."
            )
        if not isinstance(kernel_module.compute, wp.context.Kernel):
            raise RuntimeError(
                "The 'compute' function must be decorated with '@wp.kernel'."
            )

        # Retrieves the type annotations for warp's kernel in/out structures.
        kernel_annotations = {
            ATTR_PORT_TYPE_INPUT: get_annotations(kernel_module.Inputs.cls),
            ATTR_PORT_TYPE_OUTPUT: get_annotations(kernel_module.Outputs.cls),
        }

        # Assert that our code is doing the right thing—each annotation found
        # must map onto a corresponding node attribute.
        assert all(
            (
                sorted(annotations.keys())
                == sorted(
                    x.get_name().split(":")[-1] for x in attrs
                    if x.get_port_type() == port_type
                )
            )
            for port_type, annotations in kernel_annotations.items()
        )

        # Ensure that all output parameters are arrays. Writing to non-array
        # types is not supported as per CUDA's design.
        invalid_attrs = tuple(
            k
            for k, v in kernel_annotations[ATTR_PORT_TYPE_OUTPUT].items()
            if not isinstance(v, wp.array)
        )
        if invalid_attrs:
            raise RuntimeError(
                "Output attributes are required to be arrays but "
                "the following attributes are not: {}."
                .format(", ".join(invalid_attrs))
            )

        # Configure warp to only compute the forward pass.
        wp.set_module_options({"enable_backward": False}, module=kernel_module)

        # Store the public members.
        self.kernel_module = kernel_module
        self.kernel_annotations = kernel_annotations

#   Compute
# ------------------------------------------------------------------------------

def cast_array_to_warp_type(
    value: Union[np.array, og.DataWrapper],
    warp_annotation: Any,
    device: wp.context.Device,
) -> wp.array:
    """Casts an attribute array to its corresponding warp type."""
    if device.is_cpu:
        return wp.array(
            value,
            dtype=warp_annotation.dtype,
            device=device,
            owner=False,
        )

    elif device.is_cuda:
        return omni.warp.from_omni_graph(
            value,
            dtype=warp_annotation.dtype,
        )

    assert False, "Unexpected device '{}'.".format(device.alias)

def get_kernel_args(
    db: OgnKernelDatabase,
    module: Any,
    annotations: Mapping[og.AttributePortType, Sequence[Tuple[str, Any]]],
    device: wp.context.Device,
) -> Tuple[Any, Any]:
    """Retrieves the in/out argument values to pass to the kernel."""
    # Initialize the kernel's input data.
    inputs = module.Inputs()
    for name, warp_annotation in annotations[ATTR_PORT_TYPE_INPUT].items():
        # Retrieve the input attribute value and cast it to the corresponding
        # warp type if is is an array.
        value = getattr(db.inputs, name)
        if isinstance(warp_annotation, wp.array):
            value = cast_array_to_warp_type(value, warp_annotation, device)

        # Store the result in the inputs struct.
        setattr(inputs, name, value)

    # Initialize the kernel's output data.
    outputs = module.Outputs()
    for name, warp_annotation in annotations[ATTR_PORT_TYPE_OUTPUT].items():
        assert isinstance(warp_annotation, wp.array)

        # Retrieve the size of the array to allocate.
        if annotations[ATTR_PORT_TYPE_INPUT].get(name) == warp_annotation:
            # If there's an existing input with the same name and type,
            # we allocate a new array matching the input's length.
            size = len(getattr(inputs, name))
        else:
            # Fallback to allocate an array matching the kernel's dimension.
            size = db.inputs.dim

        # Allocate the array.
        setattr(db.outputs, "{}_size".format(name), size)

        # Retrieve the output attribute value and cast it to the corresponding
        # warp type.
        value = getattr(db.outputs, name)
        value = cast_array_to_warp_type(value, warp_annotation, device)

        # Store the result in the outputs struct.
        setattr(outputs, name, value)

    return (inputs, outputs)

def write_output_attrs(
    db: OgnKernelDatabase,
    annotations: Mapping[og.AttributePortType, Sequence[Tuple[str, Any]]],
    outputs: Any,
    device: wp.context.Device,
) -> None:
    """Writes the output values to the node's attributes."""
    if device.is_cuda:
        # CUDA attribute arrays are directly being written to by Warp.
        return

    for name, warp_annotation in annotations[ATTR_PORT_TYPE_OUTPUT].items():
        assert isinstance(warp_annotation, wp.array)

        value = getattr(outputs, name)
        setattr(db.outputs, name, value)

#   Compute
# ------------------------------------------------------------------------------

def compute(db: OgnKernelDatabase) -> None:
    """Evaluates the kernel."""
    try:
        device = wp.get_device(db.inputs.device)
    except Exception:
        # Fallback to a default device.
        # This can happen due to a scene being authored on a device
        # (e.g.: `cuda:1`) that is not available to another user opening
        # that same scene.
        device = wp.get_device("cuda:0")

    if device.is_cpu:
        on_gpu = False
    elif device.is_cuda:
        on_gpu = True
    else:
        assert False, "Unexpected device '{}'.".format(device.alias)

    db.set_dynamic_attribute_memory_location(
        on_gpu=on_gpu,
        gpu_ptr_kind=og.PtrToPtrKind.CPU,
    )

    # Ensure that our internal state is correctly initialized.
    timeline =  omni.timeline.get_timeline_interface()
    db.internal_state.initialize(
        db,
        check_file_modified_time=timeline.is_stopped(),
    )

    # Exit early if there are no outputs.
    if not db.internal_state.kernel_annotations[ATTR_PORT_TYPE_OUTPUT]:
        return

    # Retrieve the inputs and outputs argument values to pass to the kernel.
    inputs, outputs = get_kernel_args(
        db,
        db.internal_state.kernel_module,
        db.internal_state.kernel_annotations,
        device,
    )

    # Ensure that all array input attributes are not NULL.
    for attr_name in db.internal_state.kernel_annotations[ATTR_PORT_TYPE_INPUT]:
        value = getattr(inputs, attr_name)
        if not isinstance(value, wp.array):
            continue

        if not value.ptr:
            return

    # Launch the kernel.
    with wp.ScopedDevice(device):
        wp.launch(
            db.internal_state.kernel_module.compute,
            dim=db.inputs.dim,
            inputs=[inputs],
            outputs=[outputs],
        )

    # Write the output values to the node's attributes.
    write_output_attrs(
        db,
        db.internal_state.kernel_annotations,
        outputs,
        device,
    )

    # Fire the execution for the downstream nodes.
    db.outputs.execOut = og.ExecutionAttributeState.ENABLED

#   Node Entry Point
# ------------------------------------------------------------------------------

class OgnKernel:
    """Warp's kernel node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def initialize(graph_context, node):
        
        # Populate the devices tokens.
        attr = og.Controller.attribute("inputs:device", node)
        if attr.get_metadata(og.MetadataKeys.ALLOWED_TOKENS) is None:
            attr.set_metadata(
                og.MetadataKeys.ALLOWED_TOKENS,
                #",".join(x.alias for x in wp.get_devices()),
                ",".join(["cpu", "cuda:0"])
            )

    @staticmethod
    def compute(db) -> None:
        compute(db)
