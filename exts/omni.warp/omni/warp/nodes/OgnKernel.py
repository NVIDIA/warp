"""Warp kernel exposed as an Omni Graph node."""

from collections import namedtuple
import enum
import importlib.util
import inspect
import os
import tempfile
from typing import (
    Any,
    Callable,
    Tuple,
)

import warp as wp

import omni.graph.core as og
import omni.timeline

from omni.warp.ogn.OgnKernelDatabase import OgnKernelDatabase

wp.init()

#   Internal State
# ------------------------------------------------------------------------------

class Access(enum.IntFlag):
    """Access flags."""

    INVALID = 1 << 0
    READ    = 1 << 1
    WRITE   = 1 << 2

_DataFlow = namedtuple(
    "DataFlow",
    (
        "name",
        "param_type",
        "param_access",
    ),
)

class DataFlow(_DataFlow):
    """Describes how a single piece of data is moved across node and kernel.

    This data flow class is the layer that takes care of passing data from the
    node's input attributes onto the kernel's parameters and then back onto
    the output attributes, while taking care of casting data when necessary.

    Kernel parameter don't need to have are expected to have corresponding
    input and/or output attributes defined on the node. The attributes found for
    a given parameter determine the type of read/write access associated with
    that parameter.

    As a result, there are 3 types of valid data flows:

    - read-only: data is read from an input node attribute and is passed to
      the kernel.
    - read and write: data is read from an input node attribute, it is passed to
      the kernel, and is then written back onto an output node attribute.
    - write-only: a piece of data is initialized according to the type of
      the parameter, it is passed to the kernel, and is then written onto
      an output node attribute.
    """

    __slots__ = ()

    def read_input_value(self, db: OgnKernelDatabase) -> Any:
        """Reads the value to pass onto the kernel."""
        assert self.param_access != Access.INVALID

        if Access.READ in self.param_access:
            # We have an input attribute, so just return its value after
            # casting it to the corresponding warp type if it is an array.
            value = getattr(db.inputs, self.name)

            if isinstance(self.param_type, wp.array):
                return type(self.param_type)(
                    value,
                    shape=value.shape,
                    dtype=self.param_type.dtype,
                    device=db.inputs.device,
                    requires_grad=self.param_type.requires_grad,
                    copy=Access.WRITE in self.param_access,
                )

            return value

        # If we only have an output attribute at hand, we need to declare
        # a new variable of the same type that the kernel can write to.
        if isinstance(self.param_type, wp.array):
            # In the case of an array, we also need to allocate it with the
            # expected size. An acceptable default is to match the kernel's
            # dimension but, in the case that's not good enough, then the user
            # really should create an explicit input attribute for it and
            # initialize it with an array of the expected size.
            return wp.zeros(
                shape=db.inputs.dim,
                dtype=self.param_type.dtype,
                device=db.inputs.device,
                requires_grad=self.param_type.requires_grad,
            )

        return self.param_type(0)

    def write_output_value(self, db: OgnKernelDatabase, value: Any) -> None:
        """Writes the value onto the corresponding output attribute, if any."""
        assert self.param_access != Access.INVALID

        if not Access.WRITE in self.param_access:
            return

        if isinstance(self.param_type, wp.array):
            setattr(db.outputs, self.name, value.numpy())
            return

        setattr(db.outputs, self.name, value)

def load_module(file_path: str, name: str) -> Any:
    """Loads a Python module from its file path."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_kernel_fn(db: OgnKernelDatabase) -> Callable:
    """Retrieves the kernel function object with its decorator."""
    # Ensure that we create a distinct Python module for each node.
    module_name = "warp-kernelnode-{}".format(db.node.node_id())

    # Create the module.
    code_provider = db.inputs.codeProvider
    if code_provider == "embedded":
        # It's possible to use the `exec()` built-in function to create and
        # populate a Python module with the source code defined in a string,
        # however warp requires access to the source code of the kernel's
        # function, which is only available when the original source file
        # pointed by the function attribute `__code__.co_filename` can
        # be opened to read the lines corresponding to that function.
        # As such, we must write the source code into a temporary file
        # on disk before importing it as a module and having the function
        # turned into a kernel by warp's mechanism.

        code_str = db.inputs.codeStr

        # Create a temporary file.
        file, file_path = tempfile.mkstemp(suffix=".py")

        try:
            # Save the embedded code into the temporary file.
            with os.fdopen(file, "w") as f:
                f.write(code_str)

            # Import the temporary file as a Python module.
            module = load_module(file_path, module_name)
        finally:
            # The resulting Python module is stored into memory as a bytcode
            # object and the kernel function has already been parsed by warp
            # as long as it was correctly decorated, so it's now safe to
            # clean-up the temporary file.
            os.remove(file_path)
    elif code_provider == "file":
        code_file = db.inputs.codeFile
        module = load_module(code_file, module_name)
    else:
        assert False, "Unexpected code provider '{}'.".format(code_provider)

    if not hasattr(module, "compute"):
        raise RuntimeError(
            "The code must define a kernel function named `compute`."
        )

    return module.compute

def get_param_access(
    node: og.Node,
    name: str,
) -> Access:
    """Retrieves the read/write access flag for the given parameter name."""
    has_in_attr = node.get_attribute_exists("inputs:{}".format(name))
    has_out_attr = node.get_attribute_exists("outputs:{}".format(name))

    if not has_in_attr and not has_out_attr:
        return Access.INVALID

    if not has_out_attr:
        return Access.READ

    if not has_in_attr:
        return Access.WRITE

    return Access.READ | Access.WRITE

def gather_data_flows(
    node: og.Node,
    kernel_fn: Callable,
) -> Tuple[DataFlow, ...]:
    """Gathers the objects required to move the data across node and kernel."""
    # Retrieve the parameters defined on the kernel with their corresponding
    # type annotations.
    params = inspect.signature(kernel_fn.func).parameters

    # Initialize the data flow objects.
    flows = tuple(
        DataFlow(
            name=name,
            param_type=param.annotation,
            param_access=get_param_access(node, name),
        )
        for name, param in params.items()
    )

    # Check that no attributes are missing.
    missing_attrs = tuple(
        x.name for x in flows if x.param_access == Access.INVALID
    )
    if missing_attrs:
        raise RuntimeError(
            "The following attributes need to be defined onto the node "
            "to map with the kernel's function parameters: {}."
            .format(", ".join(missing_attrs))
        )

    return flows

class InternalState:
    """Internal state for the node."""

    def __init__(self):
        self._code_provider = None
        self._code_str = None
        self._code_file = None
        self._code_file_timestamp = None

        self._initialized = False

        self.kernel_fn = None
        self.data_flows = None

        # self.reset()

    def _is_outdated(
        self,
        db: OgnKernelDatabase,
        check_file_modified_time: bool,
    ) -> bool:
        """Checks if the internal state is outdated."""
        if self._code_provider != db.inputs.codeProvider:
            return True

        if self._code_provider == "embedded":
            return self._code_str != db.inputs.codeStr

        if self._code_provider == "file":
            return (
                self._code_file != db.inputs.codeFile
                or (
                    check_file_modified_time
                    and self._code_file_timestamp != os.path.getmtime(self._code_file)
                )
            )

        assert False, (
            "Unexpected code provider '{}'.".format(self._code_provider),
        )

    def initialize(
        self,
        db: OgnKernelDatabase,
        check_file_modified_time: bool = False,
    ) -> None:
        """Initialize the internal state if needed."""
        if (
            self._initialized
            and not self._is_outdated(db, check_file_modified_time)
        ):
            return

        self._code_provider = db.inputs.codeProvider
        self._code_str = db.inputs.codeStr
        self._code_file = db.inputs.codeFile

        self.kernel_fn = get_kernel_fn(db)
        self.data_flows = gather_data_flows(db.node, self.kernel_fn)

        self._initialized = True

#   Compute
# ------------------------------------------------------------------------------

def compute(db: OgnKernelDatabase) -> None:
    """Runs the provided kernel."""
    timeline =  omni.timeline.get_timeline_interface()
    db.internal_state.initialize(
        db,
        check_file_modified_time=timeline.is_stopped(),
    )

    values = [x.read_input_value(db) for x in db.internal_state.data_flows]
    with wp.ScopedDevice(db.inputs.device):
        for _ in range(db.inputs.iterations):
            wp.launch(
                kernel=db.internal_state.kernel_fn,
                dim=db.inputs.dim,
                inputs=values,
            )

    for data_flow, value in zip(db.internal_state.data_flows, values):
        data_flow.write_output_value(db, value)

    db.outputs.execOut = og.ExecutionAttributeState.ENABLED

#   Node Entry Point
# ------------------------------------------------------------------------------

class OgnKernel:
    """Warp's kernel node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db) -> None:
        compute(db)
