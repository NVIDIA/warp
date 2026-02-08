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
from warp._src.context import export_stubs

parser = argparse.ArgumentParser(
    description="Warp Sphinx Documentation Builder",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--html",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Build HTML documentation",
)
parser.add_argument(
    "--doctest",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run doctest tests of code blocks",
)
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

args = parser.parse_args()

# Validate argument combinations
if not args.html and not args.doctest:
    parser.error("At least one of --html or --doctest must be enabled")

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
    try:
        import pre_commit.main  # noqa: PLC0415

        result = pre_commit.main.main(["run", "ruff-format", "--files", file_path])
        logger.debug(f"pre-commit returned exit code {result} (first run)")

        if result == 0:
            # Success - file was already formatted or no changes needed
            logger.info(f"File {file_path} is already formatted")
        elif result == 1:
            # Exit code 1 typically means files were modified
            # Run again to verify the file is now properly formatted
            logger.info("Running pre-commit again to verify formatting (a 'Passed' message below is expected)")
            result = pre_commit.main.main(["run", "ruff-format", "--files", file_path])
            logger.debug(f"pre-commit returned exit code {result} (second run)")

            if result == 0:
                # Success - file is now properly formatted
                logger.info(f"Formatted {file_path}")
            else:
                # Still failing after formatting - this is a real error
                raise RuntimeError(
                    f"pre-commit formatting failed for {file_path}. "
                    f"File was modified but still has issues (exit code {result})"
                )
        else:
            # Exit code > 1: Unexpected error
            raise RuntimeError(f"pre-commit formatting failed for {file_path} with exit code {result}")
    except ImportError as err:
        raise ImportError(
            f"Could not format {file_path}: pre-commit not available. "
            "Install with 'pip install warp-lang[docs]' or equivalent."
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
        logger.debug(f"Running sphinx-build -W -j auto -b {builder} {source_dir} {output_dir}")
        result = build_main(["-W", "-j", "auto", "-b", builder, source_dir, output_dir])
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
with open(os.path.join(base_path, "warp", "__init__.pyi"), "w", encoding="utf-8") as stub_file:
    export_stubs(stub_file)

# code formatting of __init__.pyi
logger.info("Formatting __init__.pyi (a 'Failed' message in the output below is expected)")
format_file_with_ruff(os.path.join(base_path, "warp", "__init__.pyi"))

source_dir = os.path.join(base_path, "docs")

if args.html:
    # Build HTML docs
    html_output_dir = os.path.join(base_path, "docs", "_build", "html")
    build_sphinx_docs(source_dir, html_output_dir, "html")

if args.doctest:
    # Run doctest
    logger.info("Running doctest...")
    doctest_output_dir = os.path.join(base_path, "docs", "_build", "doctest")
    build_sphinx_docs(source_dir, doctest_output_dir, "doctest")

logger.info("Documentation build completed successfully")
