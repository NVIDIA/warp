# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
