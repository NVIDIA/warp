import glob
import os

python_header = """# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""

cpp_header = """/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

"""

dirs = [
    "warp/*.py",
    "warp/sim/*.py",
    "warp/native/*.h",
    "warp/native/*.cpp",
    "warp/native/*.cu",
    "exts/omni.warp/omni/warp/*.py",
    "exts/omni.warp/omni/warp/nodes/*.py",
    "tests/*.py",
]


for d in dirs:
    paths = glob.glob(d)

    for p in paths:
        ext = os.path.splitext(p)[1]

        if ext == ".py":
            header = python_header
        elif ext == ".cpp" or ext == ".cu" or ext == ".h":
            header = cpp_header

        f = open(p, "rt")
        s = f.read()
        f.close()

        if not s.startswith(header):
            s = header + s

            f = open(p, "wt")
            f.write(s)
            f.close()
