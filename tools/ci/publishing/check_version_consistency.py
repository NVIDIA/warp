#!/usr/bin/env python3
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

"""Check version consistency across VERSION.md, config.py, and version.h"""

import argparse
import re
import sys
from pathlib import Path


def read_version_md(path: Path) -> str:
    """Read version from VERSION.md."""
    with open(path) as f:
        version = f.readline().strip()
    return version


def read_config_py_version(path: Path) -> str:
    """Read version from warp/config.py."""
    with open(path) as f:
        content = f.read()
    
    match = re.search(r'^version:\s*str\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {path}")
    
    return match.group(1)


def read_version_h(path: Path) -> str:
    """Read WP_VERSION_STRING from warp/native/version.h."""
    with open(path) as f:
        content = f.read()
    
    match = re.search(r'#define\s+WP_VERSION_STRING\s+"([^"]+)"', content)
    if not match:
        raise ValueError(f"Could not find WP_VERSION_STRING in {path}")
    
    return match.group(1)


def check_version_consistency(base_path: Path, verbose: bool = False) -> bool:
    """Check that versions are consistent across all files.
    
    Args:
        base_path: Root directory of the Warp repository.
        verbose: Print detailed information.
    
    Returns:
        True if all versions match, False otherwise.
    """
    version_md_path = base_path / "VERSION.md"
    config_py_path = base_path / "warp" / "config.py"
    version_h_path = base_path / "warp" / "native" / "version.h"
    
    # Check that all files exist
    missing_files: list[str] = []
    for path in [version_md_path, config_py_path, version_h_path]:
        if not path.exists():
            missing_files.append(str(path))
    
    if missing_files:
        print("ERROR: Missing version files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    # Read versions
    try:
        version_md = read_version_md(version_md_path)
        config_py_version = read_config_py_version(config_py_path)
        version_h_version = read_version_h(version_h_path)
    except (OSError, ValueError) as e:
        print(f"ERROR: Failed to read version files: {e}")
        return False
    
    if verbose:
        print(f"VERSION.md:             {version_md}")
        print(f"config.py:              {config_py_version}")
        print(f"version.h:              {version_h_version}")
    
    # Check consistency
    all_match = True
    
    if version_md != config_py_version:
        print(f"ERROR: Version mismatch between VERSION.md ({version_md}) and config.py ({config_py_version})")
        all_match = False
    
    if version_md != version_h_version:
        print(f"ERROR: Version mismatch between VERSION.md ({version_md}) and version.h ({version_h_version})")
        all_match = False
    
    if config_py_version != version_h_version:
        print(f"ERROR: Version mismatch between config.py ({config_py_version}) and version.h ({version_h_version})")
        all_match = False
    
    if all_match:
        if verbose:
            print(f"✓ All versions consistent: {version_md}")
        return True
    else:
        print("\nPlease ensure VERSION.md, warp/config.py, and warp/native/version.h all contain the same version.")
        print("After updating VERSION.md, run 'uv run build_lib.py' to regenerate version.h and update config.py.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check version consistency across Warp source files")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent,
        help="Base path to the Warp repository (default: script location's repo root)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed version information",
    )
    
    args = parser.parse_args()
    
    success = check_version_consistency(args.base_path, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
