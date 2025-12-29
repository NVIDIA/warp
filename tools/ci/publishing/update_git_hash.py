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

"""Script to update the _git_commit_hash in config.py with the current git commit hash."""

import os
import re
import subprocess
import sys
import argparse
from typing import Optional

def get_git_hash() -> Optional[str]:
    """Get the current git commit hash.
    
    Checks environment variables used by CI systems first:
    1. GitLab CI: CI_COMMIT_SHA
    2. GitHub Actions: GITHUB_SHA
    
    Falls back to running 'git rev-parse HEAD' if no CI environment variables are set.
    
    Returns:
        The git commit hash if available, or ``None`` if it cannot be determined.
    """
    # First check CI environment variables
    # GitLab CI
    git_hash = os.environ.get('CI_COMMIT_SHA')
    if git_hash:
        return git_hash
        
    # GitHub Actions
    git_hash = os.environ.get('GITHUB_SHA')
    if git_hash:
        return git_hash
    
    # Fallback to git command if environment variable is not set
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                          stderr=subprocess.STDOUT).decode('utf-8').strip()
        return git_hash
    except (subprocess.SubprocessError, FileNotFoundError):
        print(
            "Warning: Could not determine git commit hash. "
            "No CI environment variables (CI_COMMIT_SHA, GITHUB_SHA) are set and git command failed."
        )
        return None

def update_git_hash_in_config(config_file_path: str, git_hash: str, dry_run: bool = False) -> bool:
    """Update the _git_commit_hash in config.py with the current git commit hash.
    
    Args:
        config_file_path: Path to the config.py file.
        git_hash: The git commit hash to use.
        dry_run: If ``True``, don't actually write changes to file.
        
    Returns:
        ``True`` if successful, ``False`` otherwise.
    """
    if not git_hash:
        return False
        
    try:
        with open(config_file_path, "r") as file:
            content = file.read()

        # Define the regex to match the _git_commit_hash assignment
        pattern = r'^(_git_commit_hash\s*:\s*Optional\[str\]\s*=\s*)(None|"[^"]*")(.*)$'
        
        # Replace existing value with the git hash
        updated_content = re.sub(pattern, rf'\g<1>"{git_hash}"\g<3>', content, flags=re.MULTILINE)
        
        if dry_run:
            print(f"Dry run: Would update _git_commit_hash in {config_file_path} to {git_hash}")
            return True
        
        with open(config_file_path, 'w') as file:
            file.write(updated_content)
            
        print(f"Successfully updated _git_commit_hash in {config_file_path} to {git_hash}")
        return True
    except Exception as e:
        print(f"Error updating git hash in config file: {e}")
        return False

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update the _git_commit_hash in config.py with the current git commit hash."
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print what would be done without actually modifying files"
    )
    parser.add_argument(
        "--config-file", 
        type=str, 
        help="Path to config.py file (optional, defaults to warp/config.py in repository root)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    here = os.path.dirname(__file__)
    root_path = os.path.abspath(os.path.join(here, "..", "..", ".."))
    
    config_file = args.config_file if args.config_file else os.path.join(root_path, "warp", "config.py")
    
    git_hash = get_git_hash()
    if git_hash:
        success = update_git_hash_in_config(config_file, git_hash, dry_run=args.dry_run)
        if not success:
            sys.exit(1)
    else:
        sys.exit(1) 
