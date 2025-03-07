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

"""Script to build the node.json OGN file that lists the extension's nodes."""

import json
import os


def gather_nodes_info(
    ext_path: str,
    ext_name: str,
) -> None:
    # fmt: off
    ogn_file_paths = tuple(
        os.path.join(dir_path, file_name)
        for (dir_path, _, file_names) in os.walk(ext_path)
        for file_name in file_names if file_name.endswith(".ogn")
    )
    # fmt: on

    nodes_info = {}
    for file_path in ogn_file_paths:
        with open(file_path) as file:
            data = json.load(file)
            node_key = next(iter(data.keys()))
            node_data = data[node_key]
            nodes_info[node_key] = {
                "description": node_data.get("description", ""),
                "version": node_data.get("version", 1),
                "uiName": node_data.get("uiName", ""),
                "extension": ext_name,
                "language": node_data.get("language", ""),
            }

    return {"nodes": nodes_info}


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    root_path = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ext_path = os.path.join(root_path, "exts", "omni.warp")
    ogn_path = os.path.join(ext_path, "ogn")
    nodes_info_path = os.path.join(ogn_path, "nodes.json")

    nodes_info = gather_nodes_info(ext_path, "omni.warp")

    os.makedirs(ogn_path, exist_ok=True)
    with open(nodes_info_path, "w") as file:
        json.dump(nodes_info, file, indent=4)
