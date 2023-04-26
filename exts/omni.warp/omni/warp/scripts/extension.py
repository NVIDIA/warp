# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import asyncio
from contextlib import suppress
from typing import Sequence
from .menu import WarpMenu
from .common import log_info
from .common import log_error
from . import menu_common
import warp as wp
import os, sys, subprocess
import webbrowser
import importlib
import carb
import carb.dictionary
import omni.graph.core as og
import omni.ext
import omni.kit.actions.core
import omni.timeline

SCRIPTS_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scripts"))
SCENES_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scenes"))

WARP_GETTING_STARTED_URL = "https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_warp.html"
WARP_DOCUMENTATION_URL = "https://nvidia.github.io/warp/"

KERNEL_NODE_OPT_IN_SETTING = "/app/omni.warp.kernel/opt_in"
KERNEL_NODE_ENABLE_OPT_IN_SETTING = "/app/omni.warp.kernel/enable_opt_in"
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
        KERNEL_NODE_ENABLE_OPT_IN_SETTING,
    ):
        # The enable-setting is not present, we enable the check
        return True

    if not settings.get(KERNEL_NODE_ENABLE_OPT_IN_SETTING):
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
        settings.set(KERNEL_NODE_OPT_IN_SETTING, False)
        dialog.hide()

    def on_ok(dialog: MessageDialog):
        settings = carb.settings.get_settings()
        settings.set(KERNEL_NODE_OPT_IN_SETTING, True)
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
        # wait a few frames to allow the app ui to finish loading
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        dialog.show()

    asyncio.ensure_future(show_async())


def check_for_kernel_nodes() -> None:
    """Check for kernel node instances and confirm user wants to run them."""
    # If the check is not enabled then we are good
    if not is_kernel_node_check_enabled():
        return

    # Check is enabled - see if they already opted-in
    settings = carb.settings.get_settings()
    if settings.get(KERNEL_NODE_OPT_IN_SETTING):
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
    if settings.get(KERNEL_NODE_OPT_IN_SETTING):
        set_all_graphs_enabled(True)


class OmniWarpExtension(omni.ext.IExt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stage_subscription = None
        self._opt_in_setting_sub = None

        with suppress(ImportError):
            import omni.kit.app

            app = omni.kit.app.get_app()
            manager = app.get_extension_manager()
            if manager.is_extension_enabled("omni.graph.ui"):
                import omni.graph.ui

                omni.graph.ui.ComputeNodeWidget.get_instance().add_template_path(__file__)

    def on_startup(self, ext_id):
        log_info("OmniWarpExtension startup")

        wp.init()

        self._is_live = True
        self._ext_name = omni.ext.get_extension_name(ext_id)

        self._register_actions()
        self._menu = WarpMenu()

        try:
            importlib.import_module("omni.kit.browser.sample").register_sample_folder(SCENES_PATH, "Warp")
        except ImportError as e:
            print("Warning: sample browser not enabled.")

        self._update_event_stream = omni.kit.app.get_app_interface().get_update_event_stream()
        self._stage_event_sub = (
            omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._on_stage_event)
        )

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
            KERNEL_NODE_OPT_IN_SETTING,
            on_kernel_opt_in_setting_change,
        )
        assert self._opt_in_setting_sub

    def on_shutdown(self):
        log_info("OmniWarpExtension shutdown")

        self._menu.shutdown()
        self._menu = None
        self._deregister_actions()

        try:
            importlib.import_module("omni.kit.browser.sample").unregister_sample_folder(SCENES_PATH)
        except ImportError as e:
            print(e)

        self._update_event_stream = None
        self._stage_event_sub = None
        self._stage_subscription = None
        self._opt_in_setting_sub = None

    def _register_actions(self):
        action_registry = omni.kit.actions.core.get_action_registry()
        actions_tag = "Warp menu actions"

        # actions
        action_registry.register_action(
            self._ext_name,
            "cloth_scene",
            lambda: self._on_scene_menu_click(menu_common.SCENE_CLOTH),
            display_name="Warp->Cloth Simulation",
            description="",
            tag=actions_tag,
        )

        action_registry.register_action(
            self._ext_name,
            "deformer_scene",
            lambda: self._on_scene_menu_click(menu_common.SCENE_DEFORM),
            display_name="Warp->Simple Deformer",
            description="",
            tag=actions_tag,
        )

        action_registry.register_action(
            self._ext_name,
            "particles_scene",
            lambda: self._on_scene_menu_click(menu_common.SCENE_PARTICLES),
            display_name="Warp->Particle Simulation",
            description="",
            tag=actions_tag,
        )

        action_registry.register_action(
            self._ext_name,
            "wave_scene",
            lambda: self._on_scene_menu_click(menu_common.SCENE_WAVE),
            display_name="Warp->Wave Pool",
            description="",
            tag=actions_tag,
        )

        action_registry.register_action(
            self._ext_name,
            "marching_scene",
            lambda: self._on_scene_menu_click(menu_common.SCENE_MARCHING),
            display_name="Warp->Marching Cubes",
            description="",
            tag=actions_tag,
        )

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

    def _on_update(self, event):
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing() and self._example is not None:
            with wp.ScopedDevice("cuda:0"):
                self._example.update()
                self._example.render(is_live=self._is_live)

    def _on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.CLOSED):
            self._refresh_example()
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._refresh_example()

    def _reset_example(self):
        if self._example is not None:
            stage = omni.usd.get_context().get_stage()
            stage.GetRootLayer().Clear()
            with wp.ScopedDevice("cuda:0"):
                self._example.init(stage)
                self._example.render(is_live=self._is_live)

    def _refresh_example(self):
        self._example = None
        self._update_event_sub = None

    def _on_script_menu_click(self, script_name):
        def new_stage():
            new_stage = omni.usd.get_context().new_stage()
            if new_stage:
                stage = omni.usd.get_context().get_stage()
            else:
                log_error("Could not open new stage")
                return

            import_path = os.path.normpath(os.path.join(SCRIPTS_PATH, script_name))

            module = importlib.load_source(script_name, import_path)
            self._example = module.Example()

            if self._example is None:
                log_error("Problem loading example module")
                return
            if not hasattr(self._example, "init"):
                log_error("Example missing init() function")
                return
            if not hasattr(self._example, "update"):
                log_error("Example missing update() function")
                return
            if not hasattr(self._example, "render"):
                log_error("Example missing render() function")
                return

            with wp.ScopedDevice("cuda:0"):
                self._example.init(stage)
                self._example.render(is_live=self._is_live)

            # focus camera
            omni.usd.get_context().get_selection().set_selected_prim_paths(
                [stage.GetDefaultPrim().GetPath().pathString], False
            )
            viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window()
            if viewport_window:
                viewport_window.focus_on_selected()
            omni.usd.get_context().get_selection().clear_selected_prim_paths()

            self._update_event_sub = self._update_event_stream.create_subscription_to_pop(self._on_update)

        omni.kit.window.file.prompt_if_unsaved_stage(new_stage)

    def _on_scene_menu_click(self, scene_name):
        def new_stage():
            stage_path = os.path.normpath(os.path.join(SCENES_PATH, scene_name))
            omni.usd.get_context().open_stage(stage_path)

        omni.kit.window.file.prompt_if_unsaved_stage(new_stage)

    def _on_browse_scripts_click(self):
        open_file(SCRIPTS_PATH)

    def _on_browse_scenes_click(self):
        open_file(SCENES_PATH)

    def _on_getting_started_click(self, *_):
        webbrowser.open(WARP_GETTING_STARTED_URL)

    def _on_documentation_click(self, *_):
        webbrowser.open(WARP_DOCUMENTATION_URL)
