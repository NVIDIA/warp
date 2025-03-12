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

"""Script to update VERSION.md and config.py with a development version number."""

import os
import datetime
import re
import argparse
import sys
from typing import Optional, Tuple

def increment_minor(version: str) -> str:
    """Return an incremented version based on the input semantic version.
    
    Args:
        version: Input version string in format <major>.<minor>.<patch>.
        
    Returns:
        New version with incremented minor version and patch set to 0.
        
    Raises:
        ValueError: If version string is not in expected format.
    """
    # Split the version string into components
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError("Version string must be in <major>.<minor>.<patch> format")
    
    # Increment the minor version and reset patch to 0
    major, minor, patch = map(int, parts)
    minor += 1
    patch = 0
    
    # Return the new version string
    return f"{major}.{minor}.{patch}"

def write_new_version_to_config(config_file_path: str, new_version: str, dry_run: bool = False) -> bool:
    """Write new version into config.py.
    
    Args:
        config_file_path: Path to the config.py file.
        new_version: New version string to write.
        dry_run: If ``True``, don't actually write changes to file.
        
    Returns:
        ``True`` if successful, ``False`` otherwise.
    """
    try:
        with open(config_file_path, "r") as file:
            content = file.read()

        # Define the regex to match the version assignment
        pattern = r'^version\s*:\s*str\s*=\s*"(\d+\.\d+\.\d+(?:\.\w+\d+)?)"$'

        # Replace the old version with the new version
        updated_content = re.sub(pattern, f'version: str = "{new_version}"', content, flags=re.MULTILINE)
        
        if dry_run:
            print(f"Dry run: Would update version in {config_file_path} to {new_version}")
            return True
            
        with open(config_file_path, 'w') as file:
            file.write(updated_content)
            
        print(f"Successfully updated version in {config_file_path} to {new_version}")
        return True
    except Exception as e:
        print(f"Error updating version in config file: {e}")
        return False

def write_new_version_to_version_file(version_file_path: str, new_version: str, dry_run: bool = False) -> bool:
    """Write new version into VERSION.md file.
    
    Args:
        version_file_path: Path to the VERSION.md file.
        new_version: New version string to write.
        dry_run: If ``True``, don't actually write changes to file.
        
    Returns:
        ``True`` if successful, ``False`` otherwise.
    """
    try:
        if dry_run:
            print(f"Dry run: Would update version in {version_file_path} to {new_version}")
            return True
            
        with open(version_file_path, 'w') as file:
            file.write(new_version + '\n')
            
        print(f"Successfully updated version in {version_file_path} to {new_version}")
        return True
    except Exception as e:
        print(f"Error updating version file: {e}")
        return False

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update VERSION.md and config.py with a development version number."
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print what would be done without actually modifying files"
    )
    parser.add_argument(
        "--version-file", 
        type=str, 
        help="Path to VERSION.md file (optional, defaults to VERSION.md in repository root)"
    )
    parser.add_argument(
        "--config-file", 
        type=str, 
        help="Path to config.py file (optional, defaults to warp/config.py in repository root)"
    )
    parser.add_argument(
        "--date", 
        type=str, 
        help="Date to use in version (YYYYMMDD format, defaults to today)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    here = os.path.dirname(__file__)
    root_path = os.path.abspath(os.path.join(here, "..", "..", ".."))

    version_file = args.version_file if args.version_file else os.path.join(root_path, "VERSION.md")
    config_file = args.config_file if args.config_file else os.path.join(root_path, "warp", "config.py")

    try:
        with open(version_file, "r") as file:
            base_version = file.readline().strip()
            
        # Increment the minor version so the nightly build is considered newer than the latest release
        base_version_incremented = increment_minor(base_version)

        # See https://peps.python.org/pep-0440/#developmental-releases
        dateint = args.date if args.date else datetime.date.today().strftime("%Y%m%d")

        dev_version_string = f"{base_version_incremented}.dev{dateint}"

        print(f"Preparing to update version from {base_version} to {dev_version_string}...")

        # Update VERSION.md and /warp/config.py with the new version
        version_file_success = write_new_version_to_version_file(version_file, dev_version_string, dry_run=args.dry_run)
        config_file_success = write_new_version_to_config(config_file, dev_version_string, dry_run=args.dry_run)
        
        if not (version_file_success and config_file_success):
            sys.exit(1)
            
    except Exception as e:
        print(f"Error updating version: {e}")
        sys.exit(1)
