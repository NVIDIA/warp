# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Warp kernel exposed as an OmniGraph node."""

import traceback
from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
)

import warp as wp

import omni.graph.core as og
import omni.timeline

from omni.warp.ogn.OgnKernelDatabase import OgnKernelDatabase
from omni.warp.scripts.nodes.kernel import (
    MAX_DIMENSIONS,
    InternalStateBase,
    UserAttributesEvent,
    deserialize_user_attribute_descs,
    get_kernel_args,
    initialize_kernel_module,
    validate_input_arrays,
)

QUIET_DEFAULT = wp.config.quiet

ATTR_PORT_TYPE_INPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT
ATTR_PORT_TYPE_OUTPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT

#   Internal State
# ------------------------------------------------------------------------------

def get_annotations(obj: Any) -> Mapping[str, Any]:
    """Alternative to `inspect.get_annotations()` for Python 3.9 and older."""
    # See https://docs.python.org/3/howto/annotations.html#accessing-the-annotations-dict-of-an-object-in-python-3-9-and-older
    if isinstance(obj, type):
        return obj.__dict__.get("__annotations__", {})

    return getattr(obj, "__annotations__", {})

class InternalState(InternalStateBase):
    """Internal state for the node."""

    def __init__(self) -> None:
        super().__init__()
        self._dim_count = None

    def needs_initialization(
        self,
        db: OgnKernelDatabase,
        check_file_modified_time: bool,
    ) -> bool:
        """Checks if the internal state needs to be (re)initialized."""
        if super().needs_initialization(
            db,
            check_file_modified_time=check_file_modified_time,
        ):
            return True

        if self._dim_count != db.inputs.dimCount:
            return True

        return False

    def initialize(self, db: OgnKernelDatabase) -> bool:
        """Initialize the internal state and recompile the kernel."""
        if not super().initialize(db):
            return False

        # Cache the node attribute values relevant to this internal state.
        # They're the ones used to check whether this state is outdated or not.
        self._dim_count = db.inputs.dimCount

        # Retrieve the dynamic user attributes defined on the node.
        attrs = tuple(x for x in db.node.get_attributes() if x.is_dynamic())

        # Retrieve any user attribute descriptions available.
        attr_descs = deserialize_user_attribute_descs(db.state.userAttrDescs)

        try:
            kernel_module = initialize_kernel_module(
                attrs,
                attr_descs,
                db.inputs.codeProvider,
                db.inputs.codeStr,
                db.inputs.codeFile,
            )
        except Exception:
            db.log_error(traceback.format_exc())
            return False

        # Retrieves the type annotations for warp's kernel in/out structures.
        kernel_annotations = {
            ATTR_PORT_TYPE_INPUT: get_annotations(kernel_module.Inputs.cls),
            ATTR_PORT_TYPE_OUTPUT: get_annotations(kernel_module.Outputs.cls),
        }

        # Ensure that all output parameters are arrays. Writing to non-array
        # types is not supported as per CUDA's design.
        invalid_attrs = tuple(
            k
            for k, v in kernel_annotations[ATTR_PORT_TYPE_OUTPUT].items()
            if not isinstance(v, wp.array)
        )
        if invalid_attrs:
            db.log_error(
                "Output attributes are required to be arrays but "
                "the following attributes are not: {}."
                .format(", ".join(invalid_attrs))
            )
            return False

        # Define the base class members.
        self.kernel_module = kernel_module
        self.kernel_annotations = kernel_annotations

        return True

#   Compute
# ------------------------------------------------------------------------------

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

def compute(db: OgnKernelDatabase, device: wp.context.Device) -> None:
    """Evaluates the node."""
    db.set_dynamic_attribute_memory_location(
        on_gpu=device.is_cuda,
        gpu_ptr_kind=og.PtrToPtrKind.CPU,
    )

    # Ensure that our internal state is correctly initialized.
    timeline =  omni.timeline.get_timeline_interface()
    if db.internal_state.needs_initialization(db, timeline.is_stopped()):
        if not db.internal_state.initialize(db):
            return

        db.internal_state.is_valid = True

    # Exit early if there are no outputs.
    if not db.internal_state.kernel_annotations[ATTR_PORT_TYPE_OUTPUT]:
        return

    # Retrieve the number of dimensions.
    dim_count = min(max(db.inputs.dimCount, 1), MAX_DIMENSIONS)

    # Retrieve the shape of the dimensions to launch the kernel with.
    dims = tuple(
        max(getattr(db.inputs, "dim{}".format(i + 1)), 0)
        for i in range(dim_count)
    )

    # Retrieve the inputs and outputs argument values to pass to the kernel.
    inputs, outputs = get_kernel_args(
        db.inputs,
        db.outputs,
        db.internal_state.kernel_module,
        db.internal_state.kernel_annotations,
        dims,
        device,
    )

    # Validates array input attributes.
    validate_input_arrays(db.node, db.internal_state.kernel_annotations, inputs)

    # Launch the kernel.
    wp.launch(
        db.internal_state.kernel_module.compute,
        dim=dims,
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

#   Node Entry Point
# ------------------------------------------------------------------------------

class OgnKernel:
    """Warp's kernel node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def initialize(graph_context: og.GraphContext, node: og.Node) -> None:
        # Populate the devices tokens.
        attr = og.Controller.attribute("inputs:device", node)
        if attr.get_metadata(og.MetadataKeys.ALLOWED_TOKENS) is None:
            attr.set_metadata(
                og.MetadataKeys.ALLOWED_TOKENS,
                ",".join(["cpu", "cuda:0"])
            )

    @staticmethod
    def compute(db: OgnKernelDatabase) -> None:
        try:
            device = wp.get_device(db.inputs.device)
        except Exception:
            # Fallback to a default device.
            # This can happen due to a scene being authored on a device
            # (e.g.: `cuda:1`) that is not available to another user opening
            # that same scene.
            device = wp.get_device("cuda:0")

        try:
            with wp.ScopedDevice(device):
                compute(db, device)
        except Exception:
            db.internal_state.is_valid = False
            db.log_error(traceback.format_exc())
            wp.config.quiet = True
            return
        else:
            wp.config.quiet = QUIET_DEFAULT

        # Reset the user attributes event since it has now been processed.
        db.state.userAttrsEvent = UserAttributesEvent.NONE

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
