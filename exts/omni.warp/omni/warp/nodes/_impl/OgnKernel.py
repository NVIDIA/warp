# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Warp kernel exposed as an OmniGraph node."""

import traceback
from typing import Tuple

import omni.graph.core as og
import omni.graph.tools.ogn as ogn
import omni.timeline
import warp as wp

from omni.warp.nodes._impl.kernel import (
    EXPLICIT_SOURCE,
    InternalStateBase,
    UserAttributesEvent,
    deserialize_user_attribute_descs,
    gather_attribute_infos,
    get_kernel_args,
    initialize_kernel_module,
    validate_input_arrays,
    write_output_attrs,
)
from omni.warp.nodes._impl.attributes import attr_join_name
from omni.warp.nodes.ogn.OgnKernelDatabase import OgnKernelDatabase


QUIET_DEFAULT = wp.config.quiet

ATTR_PORT_TYPE_INPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT
ATTR_PORT_TYPE_OUTPUT = og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT


#   Internal State
# ------------------------------------------------------------------------------


class InternalState(InternalStateBase):
    """Internal state for the node."""

    def __init__(self) -> None:
        super().__init__()

        self.attr_tracking = omni.warp.nodes.AttrTracking(
            (
                "dimCount",
            ),
        )

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

        if self.attr_tracking.have_attrs_changed(db):
            return True

        return False

    def initialize(
        self,
        db: OgnKernelDatabase,
        kernel_dim_count: int,
    ) -> bool:
        """Initializes the internal state and recompile the kernel."""
        if not super().initialize(db):
            return False

        # Retrieve the user attribute descriptions, if any.
        attr_descs = deserialize_user_attribute_descs(db.state.userAttrDescs)

        # Gather the information about each attribute to pass to the kernel.
        attr_infos = gather_attribute_infos(
            db.node,
            db.inputs,
            db.outputs,
            attr_descs,
            kernel_dim_count,
        )

        try:
            kernel_module = initialize_kernel_module(
                attr_infos,
                self._code_provider,
                self._code_str,
                self._code_file,
            )
        except Exception:
            db.log_error(traceback.format_exc())
            return False

        # Define the base class members.
        self.attr_infos = attr_infos
        self.kernel_module = kernel_module

        self.attr_tracking.update_state(db)

        return True


#   Compute
# ------------------------------------------------------------------------------


def infer_kernel_shape(
    db: OgnKernelDatabase,
) -> Tuple[int, ...]:
    """Infers the shape of the kernel."""
    source = db.inputs.dimSource
    if source == EXPLICIT_SOURCE:
        dim_count = min(max(db.inputs.dimCount, 0), wp.types.ARRAY_MAX_DIMS)
        return tuple(max(getattr(db.inputs, "dim{}".format(i + 1)), 0) for i in range(dim_count))

    try:
        value = getattr(db.inputs, source)
    except AttributeError:
        raise RuntimeError(
            "The attribute '{}' used to source the dimension doesn't exist.".format(
                attr_join_name(ATTR_PORT_TYPE_INPUT, source)
            )
        )

    try:
        return (value.shape[0],)
    except AttributeError:
        raise RuntimeError(
            "The attribute '{}' used to source the dimension isn't an array.".format(
                attr_join_name(ATTR_PORT_TYPE_INPUT, source)
            )
        )


def compute(db: OgnKernelDatabase, device: wp.context.Device) -> None:
    """Evaluates the node."""
    db.set_dynamic_attribute_memory_location(
        on_gpu=device.is_cuda,
        gpu_ptr_kind=og.PtrToPtrKind.CPU,
    )

    # Infer the kernels's shape.
    kernel_shape = infer_kernel_shape(db)

    # Ensure that our internal state is correctly initialized.
    timeline = omni.timeline.get_timeline_interface()
    if db.internal_state.needs_initialization(db, timeline.is_stopped()):
        if not db.internal_state.initialize(db, len(kernel_shape)):
            return

        db.internal_state.is_valid = True

    # Exit early if there are no outputs defined.
    if not db.internal_state.attr_infos[ATTR_PORT_TYPE_OUTPUT]:
        return

    # Retrieve the inputs and outputs argument values to pass to the kernel.
    inputs, outputs = get_kernel_args(
        db.inputs,
        db.outputs,
        db.internal_state.attr_infos,
        db.internal_state.kernel_module,
        kernel_shape,
    )

    # Ensure that all array input values are valid.
    validate_input_arrays(db.node, db.internal_state.attr_infos, inputs)

    # Launch the kernel.
    wp.launch(
        db.internal_state.kernel_module.compute,
        dim=kernel_shape,
        inputs=[inputs],
        outputs=[outputs],
    )

    # Write the output values to the node's attributes.
    write_output_attrs(db.outputs, db.internal_state.attr_infos, outputs)


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
        if attr.get_metadata(ogn.MetadataKeys.ALLOWED_TOKENS) is None:
            attr.set_metadata(ogn.MetadataKeys.ALLOWED_TOKENS, ",".join(["cpu", "cuda:0"]))

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
