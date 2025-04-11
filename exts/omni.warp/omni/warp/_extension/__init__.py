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

"""Entry point for the extension."""

import asyncio
import os
import subprocess
import sys
import webbrowser
from contextlib import suppress
from typing import Sequence

import carb
import carb.dictionary
import omni.ext
import omni.graph.core as og
import omni.kit.actions.core

import warp as wp

SCENES_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scenes"))
NODES_INIT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../nodes/_impl/__init__.py")
)

WARP_GETTING_STARTED_URL = "https://docs.omniverse.nvidia.com/extensions/latest/ext_warp.html"
WARP_DOCUMENTATION_URL = "https://nvidia.github.io/warp/"

SETTING_ENABLE_BACKWARD = "/exts/omni.warp/enable_backward"
SETTING_ENABLE_MENU = "/exts/omni.warp/enable_menu"

SETTING_KERNEL_NODE_OPT_IN = "/app/omni.warp/kernel_opt_in"
SETTING_KERNEL_NODE_ENABLE_OPT_IN = "/app/omni.warp/kernel_enable_opt_in"
OMNIGRAPH_STAGEUPDATE_ORDER = 100  # We want our attach() to run after OG so that nodes have been instantiated


def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        subprocess.call(["xdg-open", filename])


def set_all_graphs_enabled(enable: bool) -> None:
    """Set the enabled state of all OmniGraphs"""
    graphs = og.get_all_graphs()
    for graph in graphs:
        graph.set_disabled(not enable)


def is_kernel_node_check_enabled() -> bool:
    """Check whether the kernel node opt-in is enabled"""
    settings = carb.settings.get_settings()
    if not settings.is_accessible_as(
        carb.dictionary.ItemType.BOOL,
        SETTING_KERNEL_NODE_ENABLE_OPT_IN,
    ):
        # The enable-setting is not present, we enable the check
        return True

    if not settings.get(SETTING_KERNEL_NODE_ENABLE_OPT_IN):
        # The enable-setting is present and False, disable the check
        return False

    # the enable-setting is present and True, enable the check
    return True


VERIFY_KERNEL_NODE_LOAD_MSG = """This stage contains Warp kernel nodes.

There is currently no limitation on what code can be executed by this node. This means that graphs that contain these nodes should only be used when the author of the graph is trusted.

Do you want to enable the Warp kernel node functionality for this session?
"""


def verify_kernel_node_load(nodes: Sequence[og.Node]):
    """Confirm the user wants to run the nodes for the current session."""
    from omni.kit.window.popup_dialog import MessageDialog

    def on_cancel(dialog: MessageDialog):
        settings = carb.settings.get_settings()
        settings.set(SETTING_KERNEL_NODE_OPT_IN, False)
        dialog.hide()

    def on_ok(dialog: MessageDialog):
        settings = carb.settings.get_settings()
        settings.set(SETTING_KERNEL_NODE_OPT_IN, True)
        dialog.hide()

    dialog = MessageDialog(
        title="Warning",
        width=400,
        message=VERIFY_KERNEL_NODE_LOAD_MSG,
        ok_handler=on_ok,
        ok_label="Yes",
        cancel_handler=on_cancel,
        cancel_label="No",
    )

    async def show_async():
        import omni.kit.app

        # wait a few frames to allow the app ui to finish loading
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        dialog.show()

    asyncio.ensure_future(show_async())  # noqa: RUF006


def check_for_kernel_nodes() -> None:
    """Check for kernel node instances and confirm user wants to run them."""
    # If the check is not enabled then we are good
    if not is_kernel_node_check_enabled():
        return

    # Check is enabled - see if they already opted-in
    settings = carb.settings.get_settings()
    if settings.get(SETTING_KERNEL_NODE_OPT_IN):
        # The check is enabled, and they opted-in
        return

    # The check is enabled but they opted out, or haven't been prompted yet
    try:
        import omni.kit.window.popup_dialog
    except ImportError:
        # Don't prompt in headless mode
        return

    graphs = og.get_all_graphs()
    nodes = tuple(
        n for g in graphs for n in g.get_nodes() if n.get_node_type().get_node_type() == "omni.warp.WarpKernel"
    )

    if not nodes:
        # No nodes means we can leave them enabled
        return

    # Disable them until we get the opt-in via the async dialog
    set_all_graphs_enabled(False)
    verify_kernel_node_load(nodes)


def on_attach(*args, **kwargs) -> None:
    """Called when USD stage is attached"""
    check_for_kernel_nodes()


def on_kernel_opt_in_setting_change(
    item: carb.dictionary.Item,
    change_type: carb.settings.ChangeEventType,
) -> None:
    """Update the local cache of the setting value"""
    if change_type != carb.settings.ChangeEventType.CHANGED:
        return

    settings = carb.settings.get_settings()
    if settings.get(SETTING_KERNEL_NODE_OPT_IN):
        set_all_graphs_enabled(True)


class OmniWarpExtension(omni.ext.IExt):
    def __init__(self, *args, **kwargs):
        import omni.kit.app

        super().__init__(*args, **kwargs)
        self._menu = None
        self._stage_subscription = None
        self._opt_in_setting_sub = None

        with suppress(ImportError):
            app = omni.kit.app.get_app()
            manager = app.get_extension_manager()
            if manager.is_extension_enabled("omni.graph.ui"):
                import omni.graph.ui

                omni.graph.ui.ComputeNodeWidget.get_instance().add_template_path(NODES_INIT_PATH)

    def on_startup(self, ext_id):
        import omni.kit.app

        settings = carb.settings.get_settings()

        wp.config.enable_backward = settings.get(SETTING_ENABLE_BACKWARD)

        self._is_live = True
        self._ext_name = "omni.warp"

        if settings.get(SETTING_ENABLE_MENU):
            with suppress(ImportError):
                import omni.kit.menu.utils
                from omni.warp._extension.menu import WarpMenu

                self._register_actions()
                self._menu = WarpMenu()

        with suppress(ImportError):
            import omni.kit.browser.sample

            omni.kit.browser.sample.register_sample_folder(SCENES_PATH, "Warp")

        stage_update = omni.stageupdate.get_stage_update_interface()

        self._stage_subscription = stage_update.create_stage_update_node(
            "WarpKernelAttach",
            on_attach_fn=on_attach,
        )
        assert self._stage_subscription

        nodes = stage_update.get_stage_update_nodes()
        stage_update.set_stage_update_node_order(
            len(nodes) - 1,
            OMNIGRAPH_STAGEUPDATE_ORDER + 1,
        )
        self._opt_in_setting_sub = omni.kit.app.SettingChangeSubscription(
            SETTING_KERNEL_NODE_OPT_IN,
            on_kernel_opt_in_setting_change,
        )
        assert self._opt_in_setting_sub

        # Expose the `from_omni_graph` function onto the `omni.warp` module for
        # backward compatibility. This cannot be done by using a `__init__.py`
        # file directly under the `omni/warp` folder because it represents
        # a Python namespace package.
        import omni.warp
        import omni.warp.nodes

        omni.warp.from_omni_graph = omni.warp.nodes.from_omni_graph

    def on_shutdown(self):
        if self._menu is not None:
            self._menu.shutdown()
            self._menu = None
            self._deregister_actions()

        with suppress(ImportError):
            import omni.kit.browser.sample

            omni.kit.browser.sample.unregister_sample_folder(SCENES_PATH)

        self._stage_subscription = None
        self._opt_in_setting_sub = None

        # Clean-up the extension's API.
        import omni.warp

        delattr(omni.warp, "from_omni_graph")

    def _register_actions(self):
        action_registry = omni.kit.actions.core.get_action_registry()
        actions_tag = "Warp menu actions"

        # actions
        action_registry.register_action(
            self._ext_name,
            "getting_started",
            lambda: self._on_getting_started_click(),
            display_name="Warp->Getting Started",
            description="",
            tag=actions_tag,
        )

        action_registry.register_action(
            self._ext_name,
            "documentation",
            lambda: self._on_documentation_click(),
            display_name="Warp->Documentation",
            description="",
            tag=actions_tag,
        )

        action_registry.register_action(
            self._ext_name,
            "browse_scenes",
            lambda: self._on_browse_scenes_click(),
            display_name="Warp->Browse Scenes",
            description="",
            tag=actions_tag,
        )

    def _deregister_actions(self):
        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.deregister_all_actions_for_extension(self._ext_name)

    def _on_browse_scenes_click(self):
        open_file(SCENES_PATH)

    def _on_getting_started_click(self, *_):
        webbrowser.open(WARP_GETTING_STARTED_URL)

    def _on_documentation_click(self, *_):
        webbrowser.open(WARP_DOCUMENTATION_URL)
