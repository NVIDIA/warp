# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import subprocess
import sys

from warp.context import export_functions_rst, export_stubs

# docs

# disable sphinx color output
os.environ["NO_COLOR"] = "1"

base_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(base_path, "docs", "modules", "functions.rst"), "w") as function_ref:
    export_functions_rst(function_ref)

# run Sphinx build
try:
    docs_folder = os.path.join(base_path, "docs")
    make_html_cmd = ["make.bat" if os.name == "nt" else "make", "html"]

    if os.name == "nt" or len(sys.argv) == 1:
        subprocess.check_output(make_html_cmd, cwd=docs_folder)
    else:
        # Sphinx options were passed via the argument list
        make_html_cmd.append("-e")
        sphinx_options = " ".join(sys.argv[1:])
        subprocess.check_output(make_html_cmd, cwd=docs_folder, env=dict(os.environ, SPHINXOPTS=sphinx_options))
except subprocess.CalledProcessError as e:
    print(e.output.decode())
    raise e

# generate stubs for autocomplete
with open(os.path.join(base_path, "warp", "stubs.py"), "w") as stub_file:
    export_stubs(stub_file)

# code formatting
subprocess.run([sys.executable, "-m", "black", os.path.join(base_path, "warp", "stubs.py")])

print("Finished")
