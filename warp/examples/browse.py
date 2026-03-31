# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
