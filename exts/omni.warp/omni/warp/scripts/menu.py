# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import omni.kit.menu.utils
from omni.kit.menu.utils import MenuItemDescription


class WarpMenu:
    def __init__(self):
        omni.kit.menu.utils.set_default_menu_proirity("Warp", -8)
        self._top_menu = None
        self._build_warp_menu()

    def _build_warp_menu(self):
        # Warp menu
        warp_menu = []

        warp_menu.append(MenuItemDescription(name="Documentation", onclick_action=("omni.warp", "documentation")))
        warp_menu.append(MenuItemDescription(name="Getting Started", onclick_action=("omni.warp", "getting_started")))
        warp_menu.append(MenuItemDescription(name="Sample Scenes", onclick_action=("omni.warp", "browse_scenes")))

        self._top_menu = [MenuItemDescription(name="Warp", appear_after="Simulation", sub_menu=warp_menu)]

        omni.kit.menu.utils.add_menu_items(self._top_menu, "Window", -8)

    def shutdown(self):
        omni.kit.menu.utils.remove_menu_items(self._top_menu, "Window")
