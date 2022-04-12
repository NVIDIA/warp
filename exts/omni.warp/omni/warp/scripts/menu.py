import carb
import carb.settings
import omni.kit.menu.utils
import omni.kit.context_menu
import omni.usd
import omni.timeline
import warp as wp
from omni.kit.menu.utils import MenuItemDescription
from . import menu_common
from .common import log_error
import os
import imp
import webbrowser

SCRIPTS_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scripts"))
SCENES_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scenes"))
WARP_GETTING_STARTED_URL = "https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_warp.html"
WARP_DOCUMENTATION_URL = "https://nvidia.github.io/warp/"

class WarpMenu:
    def __init__(self):

        self._script_menu = [
            {
                "name": menu_common.EXAMPLE_DEM_MENU_ITEM.split("/")[-1],
                "onclick_fn": lambda *_: self._on_script_menu_click(menu_common.EXAMPLE_DEM_SCRIPT)
            },
            {
                "name": menu_common.EXAMPLE_MESH_MENU_ITEM.split("/")[-1],
                "onclick_fn": lambda *_: self._on_script_menu_click(menu_common.EXAMPLE_MESH_SCRIPT)
            },
            {
                "name": menu_common.EXAMPLE_BROWSE_MENU_ITEM.split("/")[-1],
                "onclick_fn": lambda *_: self._on_browse_scripts_click()
            }
        ]

        self._scene_menu = [
            {
                "name": menu_common.SCENE_CLOTH_MENU_ITEM.split("/")[-1],
                "onclick_fn": lambda *_: self._on_scene_menu_click(menu_common.SCENE_CLOTH)
            },
            {
                "name": menu_common.SCENE_DEFORM_MENU_ITEM.split("/")[-1],
                "onclick_fn": lambda *_: self._on_scene_menu_click(menu_common.SCENE_DEFORM)
            },
            {
                "name": menu_common.SCENE_PARTICLES_MENU_ITEM.split("/")[-1],
                "onclick_fn": lambda *_: self._on_scene_menu_click(menu_common.SCENE_PARTICLES)
            },
            {
                "name": menu_common.SCENE_WAVE_MENU_ITEM.split("/")[-1],
                "onclick_fn": lambda *_: self._on_scene_menu_click(menu_common.SCENE_WAVE)
            },
            {
                "name": menu_common.SCENE_BROWSE_MENU_ITEM.split("/")[-1],
                "onclick_fn": lambda *_: self._on_browse_scenes_click()
            }
        ]

        self._help_menu = [
            {
                "name": "Getting Started",
                "onclick_fn": lambda *_: self._on_getting_started_click()
            },
            {
                "name": "Documentation",
                "onclick_fn": lambda *_: self._on_documentation_click()
            }
        ]

        self._script_menu_descriptions = []
        for menu_item in self._script_menu:
            menu_item_description = MenuItemDescription(name=menu_item.get("name"), onclick_fn=menu_item.get("onclick_fn"))
            self._script_menu_descriptions.append(menu_item_description)

        self._scene_menu_descriptions = []
        for menu_item in self._scene_menu:
            menu_item_description = MenuItemDescription(name=menu_item.get("name"), onclick_fn=menu_item.get("onclick_fn"))
            self._scene_menu_descriptions.append(menu_item_description)

        self._help_menu_descriptions = []
        for menu_item in self._help_menu:
            menu_item_description = MenuItemDescription(name=menu_item.get("name"), onclick_fn=menu_item.get("onclick_fn"))
            self._help_menu_descriptions.append(menu_item_description)

        # line breaks
        self._script_menu_descriptions.insert(len(self._script_menu_descriptions) - 1, MenuItemDescription())
        self._scene_menu_descriptions.insert(len(self._scene_menu_descriptions) - 1, MenuItemDescription())

        self._window_menu_descriptions = [
            MenuItemDescription(name="Scripts", sub_menu=self._script_menu_descriptions),
            MenuItemDescription(name="Scenes", sub_menu=self._scene_menu_descriptions),
            MenuItemDescription(name="Help", sub_menu=self._help_menu_descriptions)
        ]

        self._window_menus_warp = [
            MenuItemDescription(
                name="Warp",
                appear_after="Simulation",
                sub_menu=self._window_menu_descriptions
            ),
        ]

        omni.kit.menu.utils.add_menu_items(self._window_menus_warp, "Window")
        self._update_event_stream = omni.kit.app.get_app_interface().get_update_event_stream()
        self._stage_event_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._on_stage_event)

    def shutdown(self):

        omni.kit.menu.utils.remove_menu_items(self._window_menus_warp, "Window")
        self._example = None
        self._update_event_sub = None
        self._update_event_stream = None
        self._stage_event_sub = None

    def _on_update(self, event):

        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing() and self._example is not None:
            with wp.ScopedCudaGuard():
                self._example.update()
                self._example.render(is_live=True)

    def _on_stage_event(self, event):

        if event.type == int(omni.usd.StageEventType.CLOSED):
            self._refresh_example()
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._refresh_example()

    def _refresh_example(self):

        self._example = None
        self._update_event_sub = None

    def _on_script_menu_click(self, script_name):

        def new_stage():
            self._refresh_example()
            
            new_stage = omni.usd.get_context().new_stage()
            if new_stage:
                stage = omni.usd.get_context().get_stage()
            else:
                log_error("Could not open new stage")

            import_path = os.path.normpath(os.path.join(SCRIPTS_PATH, script_name))

            module = imp.load_source(script_name, import_path)
            self._example = module.Example()

            if not hasattr(self._example, 'init'):
                log_error("Example missing init() function")
            if not hasattr(self._example, 'update'):
                log_error("Example missing update() function")
            if not hasattr(self._example, 'render'):
                log_error("Example missing render() function")

            self._example.init(stage)

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