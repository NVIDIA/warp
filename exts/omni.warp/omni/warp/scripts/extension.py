# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from contextlib import suppress
from .menu import WarpMenu
from .common import log_info
from .common import log_error
from . import menu_common
import warp as wp
import os
import imp
import webbrowser
import importlib
import omni.ext
import omni.kit.actions.core
import omni.timeline

SCRIPTS_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scripts"))
SCENES_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scenes"))

WARP_GETTING_STARTED_URL = "https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_warp.html"
WARP_DOCUMENTATION_URL = "https://nvidia.github.io/warp/"

class OmniWarpExtension(omni.ext.IExt):

    def __init__(self):
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
            importlib.import_module("omni.kit.browser.sample").register_sample_folder(
                SCENES_PATH,
                "Warp"
            )
        except ImportError as e:
            print(e)

        self._update_event_stream = omni.kit.app.get_app_interface().get_update_event_stream()
        self._stage_event_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._on_stage_event)

    def on_shutdown(self):
        log_info("OmniWarpExtension shutdown")

        self._menu.shutdown()
        self._menu = None
        self._deregister_actions()

        try:
            importlib.import_module("omni.kit.browser.sample").unregister_sample_folder(
                SCENES_PATH
            )
        except ImportError as e:
            print(e)

        self._update_event_stream = None
        self._stage_event_sub = None

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
            tag=actions_tag
        )

        action_registry.register_action(
            self._ext_name,
            "deformer_scene",
            lambda: self._on_scene_menu_click(menu_common.SCENE_DEFORM),
            display_name="Warp->Simple Deformer",
            description="",
            tag=actions_tag
        )

        action_registry.register_action(
            self._ext_name,
            "particles_scene",
            lambda: self._on_scene_menu_click(menu_common.SCENE_PARTICLES),
            display_name="Warp->Particle Simulation",
            description="",
            tag=actions_tag
        )        

        action_registry.register_action(
            self._ext_name,
            "wave_scene",
            lambda: self._on_scene_menu_click(menu_common.SCENE_WAVE),
            display_name="Warp->Wave Pool",
            description="",
            tag=actions_tag
        )

        action_registry.register_action(
            self._ext_name,
            "marching_scene",
            lambda: self._on_scene_menu_click(menu_common.SCENE_MARCHING),
            display_name="Warp->Marching Cubes",
            description="",
            tag=actions_tag
        )

        action_registry.register_action(
            self._ext_name,
            "getting_started",
            lambda: self._on_getting_started_click(),
            display_name="Warp->Getting Started",
            description="",
            tag=actions_tag
        )       

        action_registry.register_action(
            self._ext_name,
            "documentation",
            lambda: self._on_documentation_click(),
            display_name="Warp->Documentation",
            description="",
            tag=actions_tag
        )

        action_registry.register_action(
            self._ext_name,
            "browse_scenes",
            lambda: self._on_browse_scenes_click(),
            display_name="Warp->Browse Scenes",
            description="",
            tag=actions_tag
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

            module = imp.load_source(script_name, import_path)
            self._example = module.Example()

            if self._example is None:
                log_error("Problem loading example module")
                return
            if not hasattr(self._example, 'init'):
                log_error("Example missing init() function")
                return
            if not hasattr(self._example, 'update'):
                log_error("Example missing update() function")
                return
            if not hasattr(self._example, 'render'):
                log_error("Example missing render() function")
                return

            with wp.ScopedDevice("cuda:0"):
                self._example.init(stage)
                self._example.render(is_live=self._is_live)

            # focus camera
            omni.usd.get_context().get_selection().set_selected_prim_paths([stage.GetDefaultPrim().GetPath().pathString], False)
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
        os.startfile(SCRIPTS_PATH)

    def _on_browse_scenes_click(self):
        os.startfile(SCENES_PATH)

    def _on_getting_started_click(self, *_):
        webbrowser.open(WARP_GETTING_STARTED_URL)

    def _on_documentation_click(self, *_):
        webbrowser.open(WARP_DOCUMENTATION_URL)