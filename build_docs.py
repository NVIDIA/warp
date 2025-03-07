# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
