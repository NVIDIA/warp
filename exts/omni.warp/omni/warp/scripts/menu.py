import omni.kit.menu.utils
from omni.kit.menu.utils import MenuItemDescription, MenuItemOrder

class WarpMenu:

    def __init__(self):

        omni.kit.menu.utils.set_default_menu_proirity("Warp", -8)
        self._top_menu = None
        self._build_warp_menu()

    def _build_warp_menu(self):

        # scene menu
        scene_menu_list = []

        scene_menu_list.append(
            MenuItemDescription(name="Browse Scenes", onclick_action=("omni.warp", "browse_scenes"))
        )
        scene_menu_list.append(MenuItemDescription())  # line break
        scene_menu_list.append(
            MenuItemDescription(name="Cloth Simulation", onclick_action=("omni.warp", "cloth_scene"))
        )
        scene_menu_list.append(
            MenuItemDescription(name="Particle Simulation", onclick_action=("omni.warp", "particles_scene"))
        )
        scene_menu_list.append(
            MenuItemDescription(name="Wave Pool", onclick_action=("omni.warp", "wave_scene"))
        )
        scene_menu_list.append(
            MenuItemDescription(name="Marching Cubes", onclick_action=("omni.warp", "marching_scene"))
        )
        scene_menu_list.append(
            MenuItemDescription(name="Simple Deformer", onclick_action=("omni.warp", "deformer_scene"))
        )

        # help menu
        help_menu_list = []

        help_menu_list.append(
            MenuItemDescription(name="Getting Started", onclick_action=("omni.warp", "getting_started"))
        )
        help_menu_list.append(
            MenuItemDescription(name="Documentation", onclick_action=("omni.warp", "documentation"))
        )

        # Warp menu
        warp_menu = []

        warp_menu.append(
            MenuItemDescription(name="Scenes", sub_menu=scene_menu_list)
        )
        warp_menu.append(
            MenuItemDescription(name="Help", sub_menu=help_menu_list)
        )

        self._top_menu = [MenuItemDescription(name="Warp", appear_after="Simulation", sub_menu=warp_menu)]

        omni.kit.menu.utils.add_menu_items(self._top_menu, "Window", -8)

    def shutdown(self):
        omni.kit.menu.utils.remove_menu_items(self._top_menu, "Window")
