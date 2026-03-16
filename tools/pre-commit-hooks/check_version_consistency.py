# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pre-commit hook that checks version consistency and regenerates version.h.

VERSION.md is the single source of truth for the Warp version string.
This hook regenerates warp/native/version.h from VERSION.md (fixer) and
verifies that warp/config.py agrees (checker). If config.py disagrees the
commit is blocked with an actionable error message.

This hook has zero dependencies beyond the standard library, so it runs
everywhere — including pre-commit.ci.

Usage (pre-commit hook):
    Configured in .pre-commit-config.yaml with language: python
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path


def _c_copyright_header(year: int | str) -> str:
    """Return a C-style copyright/license block for generated headers.

    NOTE: This duplicates warp._src.generated_files._c_copyright_header so
    that this hook can run without importing the warp package.
    """
    return f"""/*
 * SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

"""


def _regenerate_version_header(base_path: Path, version: str) -> None:
    """Regenerate warp/native/version.h from the given version string.

    Only writes the file when the content has actually changed, so that file
    modification timestamps are preserved and downstream build systems (make,
    ninja, etc.) don't trigger unnecessary rebuilds.

    NOTE: Output must stay byte-identical to
    warp._src.generated_files.generate_version_header.
    """
    version_header_path = base_path / "warp" / "native" / "version.h"

    new_content = _c_copyright_header(datetime.date.today().year)
    new_content += "#ifndef WP_VERSION_H\n"
    new_content += "#define WP_VERSION_H\n\n"
    new_content += f'#define WP_VERSION_STRING "{version}"\n\n'
    new_content += "#endif  // WP_VERSION_H\n"

    try:
        with open(version_header_path) as f:
            if f.read() == new_content:
                return
    except FileNotFoundError:
        pass

    with open(version_header_path, "w") as f:
        f.write(new_content)


def read_version_md(path: Path) -> str:
    """Read and validate version from VERSION.md."""
    with open(path) as f:
        version = f.readline().strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+(\.\w+)?", version):
        raise ValueError(f"VERSION.md contains invalid version string: {version!r}")
    return version


def read_config_py_version(path: Path) -> str:
    """Read version from warp/config.py."""
    with open(path) as f:
        content = f.read()

    match = re.search(r'^version:\s*str\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {path}")

    return match.group(1)


def check_version_consistency(base_path: Path, verbose: bool = False) -> bool:
    """Regenerate version.h and check that config.py matches VERSION.md.

    Args:
        base_path: Root directory of the Warp repository.
        verbose: Print detailed information.

    Returns:
        True if config.py matches VERSION.md, False otherwise.
    """
    version_md_path = base_path / "VERSION.md"
    config_py_path = base_path / "warp" / "config.py"

    # Check that source files exist
    for path in [version_md_path, config_py_path]:
        if not path.exists():
            print(f"ERROR: Missing file: {path}", file=sys.stderr)
            return False

    # Read versions from source files
    try:
        version_md = read_version_md(version_md_path)
        config_py_version = read_config_py_version(config_py_path)
    except (OSError, ValueError) as e:
        print(f"ERROR: Failed to read version files: {e}", file=sys.stderr)
        return False

    # Regenerate version.h from VERSION.md (the source of truth).
    # pre-commit detects the file modification and asks the developer to stage it.
    _regenerate_version_header(base_path, version_md)

    if verbose:
        print(f"VERSION.md:             {version_md}")
        print(f"config.py:              {config_py_version}")
        print(f"version.h:              regenerated from VERSION.md")

    # Check that config.py (a source file) matches VERSION.md
    if version_md != config_py_version:
        print(
            f"ERROR: Version mismatch between VERSION.md ({version_md}) and config.py ({config_py_version})",
            file=sys.stderr,
        )
        print(
            "\nPlease ensure VERSION.md and warp/config.py contain the same version.",
            file=sys.stderr,
        )
        return False

    if verbose:
        print(f"All versions consistent: {version_md}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check version consistency across Warp source files")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(__file__).parent.parent.parent,
        help="Base path to the Warp repository (default: script location's repo root)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed version information",
    )

    args = parser.parse_args()

    success = check_version_consistency(args.base_path, verbose=args.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
