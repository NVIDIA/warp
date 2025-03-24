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

"""Shared helpers for the extension's tests."""

import importlib
import os
import tempfile
from typing import (
    Any,
    Mapping,
    Optional,
    Union,
)

import carb
import numpy as np
import omni.graph.core as og
import omni.graph.tools.ogn as ogn
import omni.kit
import omni.timeline
import omni.usd
from omni.kit.test_helpers_gfx.compare_utils import (
    ComparisonMetric,
    finalize_capture_and_compare,
)
from omni.kit.viewport.utility import (
    capture_viewport_to_file,
    get_active_viewport,
)
from omni.kit.viewport.utility.tests.capture import capture_viewport_and_wait

import warp as wp

from .._impl.attributes import from_omni_graph_ptr

_APP = omni.kit.app.get_app()
_MANAGER = _APP.get_extension_manager()
_EXT_PATH = _MANAGER.get_extension_path_by_module(__name__)
_SAMPLES_PATH = os.path.join(_EXT_PATH, "data/scenes")
_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
_OUTPUT_PATH = omni.kit.test.get_test_output_path()

_SETTING_UPDATE_GOLDEN_IMAGES = "/exts/omni.warp/update_golden_images"


def set_settings(
    values: Mapping[str, Any],
) -> None:
    """Sets Kit settings."""
    settings = carb.settings.get_settings()
    for k, v in values.items():
        settings.set(k, v)


async def open_sample(
    file_name: str,
    enable_fsd: Optional[bool] = None,
) -> None:
    """Opens a sample scene and waits until it finishes loading."""
    context = omni.usd.get_context()

    # Start with a clean slate or else some lingering settings might cause rendering artifacts.
    await context.new_stage_async()
    await omni.kit.app.get_app().next_update_async()

    if enable_fsd is not None:
        set_settings(
            {
                "/app/useFabricSceneDelegate": enable_fsd,
            }
        )

    # Open the scene.
    file_path = os.path.join(_SAMPLES_PATH, file_name)
    await context.open_stage_async(file_path)
    await omni.kit.app.get_app().next_update_async()

    # Try making the renders more deterministic.
    set_settings(
        {
            "/rtx/ambientOcclusion/enabled": False,
            "/rtx/directLighting/sampledLighting/enabled": False,
            "/rtx/ecoMode/enabled": False,
            "/rtx/indirectDiffuse/enabled": False,
            "/rtx/post/aa/op": 0,
            "/rtx/post/tonemap/op": 1,
            "/rtx/raytracing/lightcache/spatialCache/enabled": False,
            "/rtx/reflections/enabled": False,
            "/rtx/shadows/enabled": False,
        }
    )
    await omni.kit.app.get_app().next_update_async()

    # Wait for everything to be loaded.
    max_wait_update_count = 100
    for _ in range(max_wait_update_count):
        _, files_loaded, total_files = context.get_stage_loading_status()
        if files_loaded == 0 and total_files == 0:
            break

        await omni.kit.app.get_app().next_update_async()

    context.reset_renderer_accumulation()

    # Wait a bit more just to make sure...
    extra_update_count = 10
    for _ in range(extra_update_count):
        await omni.kit.app.get_app().next_update_async()


async def validate_render(
    test_id: str,
    threshold: float = 5e-4,
) -> None:
    """Ensures that a render is valid by comparing it to a golden image."""
    file_name = f"{test_id}.png"
    viewport = get_active_viewport()
    settings = carb.settings.get_settings()

    update_golden_images = settings.get(_SETTING_UPDATE_GOLDEN_IMAGES)
    if update_golden_images:
        file_path = os.path.join(_DATA_PATH, file_name)
        capture_viewport_to_file(viewport, file_path=file_path)
        return

    # Capture the viewport.
    await capture_viewport_and_wait(file_name, _OUTPUT_PATH, viewport=viewport)

    # Run the comparison.
    diff = finalize_capture_and_compare(
        file_name,
        threshold,
        _OUTPUT_PATH,
        _DATA_PATH,
        metric=ComparisonMetric.MEAN_ERROR_SQUARED,
        test_id=test_id,
    )

    if diff is None:
        raise RuntimeError(f"error while comparing the rendered result for the test {test_id}.")

    if diff > threshold:
        raise RuntimeError(f"the rendered image for the test {test_id} differs by {diff}.")


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
        data = from_omni_graph_ptr(ptr, (attr.size(),), dtype=value.dtype)
        wp.copy(data, value)
    else:
        attr.set(value.numpy(), on_gpu=False)


def attr_disconnect_all(
    attr: og.Attribute,
) -> None:
    if attr.get_port_type() == og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT:
        for src_attr in attr.get_upstream_connections():
            og.Controller.disconnect(src_attr, attr)
    elif attr.get_port_type() == og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT:
        for dst_attr in attr.get_upstream_connections():
            og.Controller.disconnect(attr, dst_attr)


def array_are_equal(
    a: Union[np.ndarray, wp.array],
    b: Union[np.ndarray, wp.array],
) -> None:
    """Checks whether two arrays are equal."""
    if isinstance(a, wp.array):
        a = a.numpy()

    if isinstance(b, wp.array):
        b = b.numpy()

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
    else:
        assert len(a) == len(b)

    np.testing.assert_equal(a, b)


def array_are_almost_equal(
    a: Union[np.ndarray, wp.array],
    b: Union[np.ndarray, wp.array],
    rtol=1e-05,
    atol=1e-08,
) -> None:
    """Checks whether two arrays are almost equal."""
    if isinstance(a, wp.array):
        a = a.numpy()

    if isinstance(b, wp.array):
        b = b.numpy()

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
    else:
        assert len(a) == len(b)

    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


class FrameRange:
    def __init__(self, count: int):
        self.count = count
        self.step = 0
        self.initial_auto_update = None

        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.stop()
        self.timeline.set_current_time(0)
        self.timeline.commit()

    def __enter__(self):
        self.initial_auto_update = self.timeline.is_auto_updating()
        self.timeline.set_auto_update(False)
        self.timeline.commit()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.timeline.set_auto_update(self.initial_auto_update)
        self.timeline.commit()

    def __aiter__(self):
        return self

    async def __anext__(self) -> int:
        self.step += 1
        if self.step > self.count:
            raise StopAsyncIteration

        self.timeline.forward_one_frame()
        self.timeline.commit()
        await omni.kit.app.get_app().next_update_async()
        return self.step
