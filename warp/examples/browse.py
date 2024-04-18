# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import subprocess
import sys


def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        subprocess.call(["xdg-open", filename])


if __name__ == "__main__":
    import warp.examples

    source_dir = warp.examples.get_source_directory()
    print(f"Example source directory: {source_dir}")

    try:
        open_file(source_dir)
    except Exception:
        pass
