# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pre-commit hook that regenerates exports.h, version.h, and __init__.pyi.

These files are derived from warp/_src/builtins.py, warp/config.py, VERSION.md,
and related source files. When those sources change, the generated files must be
updated to match. This hook regenerates them in place; the pre-commit framework
detects any modifications and blocks the commit so the developer can stage them.

Usage (pre-commit hook):
    Configured in .pre-commit-config.yaml with language: python
    and additional_dependencies: [numpy, ruff]
"""

from __future__ import annotations

import os
import subprocess
import sys

# Ensure the repo root is on sys.path so warp is importable in the
# pre-commit isolated virtualenv (which only has additional_dependencies).
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

import warp  # noqa: E402 — populates builtin_functions via builtins.py
import warp.config  # noqa: E402
from warp._src.generated_files import export_stubs, generate_exports_header_file, generate_version_header  # noqa: E402


def main() -> None:
    # Regenerate warp/native/exports.h
    generate_exports_header_file(base_path)

    # Regenerate warp/native/version.h
    generate_version_header(base_path, warp.config.version)

    # Regenerate warp/__init__.pyi
    stub_path = os.path.join(base_path, "warp", "__init__.pyi")
    with open(stub_path, "w", encoding="utf-8") as f:
        export_stubs(f)

    # Format with ruff so the generated file is stable across hook runs.
    # export_stubs() emits trailing whitespace and non-canonical formatting;
    # without this step, ruff-format would modify the file on every commit.
    # ruff is provided via additional_dependencies in .pre-commit-config.yaml.
    from ruff.__main__ import find_ruff_bin  # noqa: PLC0415

    ruff = find_ruff_bin()
    subprocess.run([ruff, "format", stub_path], check=True)
    subprocess.run([ruff, "check", "--fix", "--unsafe-fixes", stub_path], check=False)


if __name__ == "__main__":
    main()
