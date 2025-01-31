# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import shutil
import subprocess

import warp  # ensure all API functions are loaded  # noqa: F401
from warp.context import export_functions_rst, export_stubs

parser = argparse.ArgumentParser(description="Warp Sphinx Documentation Builder")
parser.add_argument("--quick", action="store_true", help="Only build docs, skipping doctest tests of code blocks.")

args = parser.parse_args()

base_path = os.path.dirname(os.path.realpath(__file__))

# generate stubs for autocomplete
with open(os.path.join(base_path, "warp", "stubs.py"), "w") as stub_file:
    export_stubs(stub_file)

# code formatting of stubs.py
subprocess.run(["ruff", "format", "--verbose", os.path.join(base_path, "warp", "stubs.py")], check=True)

with open(os.path.join(base_path, "docs", "modules", "functions.rst"), "w") as function_ref:
    export_functions_rst(function_ref)

source_dir = os.path.join(base_path, "docs")
output_dir = os.path.join(base_path, "docs", "_build", "html")

# Clean previous HTML output
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

command = ["sphinx-build", "-W", "-b", "html", source_dir, output_dir]

subprocess.run(command, check=True)

if not args.quick:
    print("Running doctest... (skip with --no_doctest)")

    output_dir = os.path.join(base_path, "docs", "_build", "doctest")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    command = ["sphinx-build", "-W", "-b", "doctest", source_dir, output_dir]

    subprocess.run(command, check=True)

print("Finished")
