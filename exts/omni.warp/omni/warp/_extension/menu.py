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

import omni.kit.menu.utils
from omni.kit.menu.utils import MenuItemDescription


class WarpMenu:
    def __init__(self):
        self._top_menu = None
        self._build_warp_menu()

    def _build_warp_menu(self):
        # Warp menu
        warp_menu = []

        warp_menu.append(MenuItemDescription(name="Documentation", onclick_action=("omni.warp", "documentation")))
        warp_menu.append(MenuItemDescription(name="Getting Started", onclick_action=("omni.warp", "getting_started")))
        warp_menu.append(MenuItemDescription(name="Sample Scenes", onclick_action=("omni.warp", "browse_scenes")))

        self._top_menu = [MenuItemDescription(name="Warp", appear_after="Simulation", sub_menu=warp_menu)]

        omni.kit.menu.utils.add_menu_items(self._top_menu, "Window")

    def shutdown(self):
        omni.kit.menu.utils.remove_menu_items(self._top_menu, "Window")
