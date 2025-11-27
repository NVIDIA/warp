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
import logging
import os
import shutil

import warp  # ensure all API functions are loaded  # noqa: F401
from warp._src.context import export_functions_rst, export_stubs

parser = argparse.ArgumentParser(description="Warp Sphinx Documentation Builder")
parser.add_argument("--quick", action="store_true", help="Only build docs, skipping doctest tests of code blocks.")
parser.add_argument("--doctest-only", action="store_true", help="Only run doctest, skipping HTML build.")
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

args = parser.parse_args()

# Validate argument combinations
if args.quick and args.doctest_only:
    parser.error("--quick and --doctest-only are mutually exclusive")

# Configure logging
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(
    level=log_level,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def format_file_with_ruff(file_path):
    """Format file with ruff using pre-commit for version consistency."""
    logger.debug(f"Formatting {file_path} with ruff via pre-commit")
    try:
        import pre_commit.main  # noqa: PLC0415

        result = pre_commit.main.main(["run", "ruff-format", "--files", file_path])
        if result == 0:
            logger.info(f"Formatted {file_path} using pre-commit (no changes needed)")
        elif result == 1:
            # Exit code 1 typically means files were modified (which is what we want)
            logger.info(f"Formatted {file_path} using pre-commit (files modified)")
        else:
            raise RuntimeError(f"Pre-commit formatting failed for {file_path} with exit code {result}")
    except ImportError as err:
        raise ImportError(
            f"Could not format {file_path}: pre-commit not available. "
            "Install with 'pip install warp-lang[dev]' or equivalent."
        ) from err


def build_sphinx_docs(source_dir, output_dir, builder="html"):
    """Build Sphinx documentation programmatically."""
    logger.info(f"Building {builder} documentation: {source_dir} -> {output_dir}")
    try:
        from sphinx.cmd.build import build_main  # noqa: PLC0415

        # Clean previous output
        if os.path.exists(output_dir):
            logger.debug(f"Cleaning previous output directory: {output_dir}")
            shutil.rmtree(output_dir)

        # sphinx-build -W -b html source_dir output_dir
        logger.debug(f"Running sphinx-build -W -b {builder} {source_dir} {output_dir}")
        result = build_main(["-W", "-b", builder, source_dir, output_dir])
        if result != 0:
            raise RuntimeError(f"Sphinx build failed with exit code {result}")

        logger.info(f"Successfully built {builder} documentation")

    except ImportError as err:
        raise ImportError(
            "Could not build docs: sphinx not available. Install with 'pip install warp-lang[docs]' or equivalent."
        ) from err


base_path = os.path.dirname(os.path.realpath(__file__))

logger.info("Starting Warp documentation build")

# generate stubs for autocomplete
logger.info("Generating API stubs for autocomplete")
with open(os.path.join(base_path, "warp", "__init__.pyi"), "w") as stub_file:
    export_stubs(stub_file)

# code formatting of __init__.pyi
format_file_with_ruff(os.path.join(base_path, "warp", "__init__.pyi"))

logger.info("Generating function reference documentation")
with open(os.path.join(base_path, "docs", "modules", "functions.rst"), "w") as function_ref:
    export_functions_rst(function_ref)

source_dir = os.path.join(base_path, "docs")

if args.doctest_only:
    # Only run doctest
    logger.info("Running doctest only (skipping HTML build)")
    doctest_output_dir = os.path.join(base_path, "docs", "_build", "doctest")
    build_sphinx_docs(source_dir, doctest_output_dir, "doctest")
else:
    # Build HTML docs
    html_output_dir = os.path.join(base_path, "docs", "_build", "html")
    build_sphinx_docs(source_dir, html_output_dir, "html")

    # Run doctest unless --quick is specified
    if not args.quick:
        logger.info("Running doctest... (skip with --quick)")
        doctest_output_dir = os.path.join(base_path, "docs", "_build", "doctest")
        build_sphinx_docs(source_dir, doctest_output_dir, "doctest")

logger.info("Documentation build completed successfully")
