# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script to update the _git_commit_hash in config.py with the current git commit hash."""

import ast
import os
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
    if not git_hash or not all(character in "0123456789abcdefABCDEF" for character in git_hash):
        return False
        
    try:
        with open(config_file_path, "r", encoding="utf-8", newline="") as file:
            content = file.read()

        assignment_values = []
        for statement in ast.parse(content, filename=config_file_path).body:
            if isinstance(statement, ast.AnnAssign):
                if isinstance(statement.target, ast.Name) and statement.target.id == "_git_commit_hash":
                    assignment_values.append(statement.value)
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id == "_git_commit_hash":
                    assignment_values.append(statement.value)

        if len(assignment_values) != 1:
            print(f"Error: _git_commit_hash assignment not found or unsupported in {config_file_path}")
            return False

        value = assignment_values[0]
        is_supported_value = isinstance(value, ast.Constant) and (value.value is None or isinstance(value.value, str))
        if not is_supported_value or value.lineno != value.end_lineno:
            print(f"Error: _git_commit_hash assignment not found or unsupported in {config_file_path}")
            return False

        lines = content.splitlines(keepends=True)
        line_index = value.lineno - 1
        line = lines[line_index].encode("utf-8")
        lines[line_index] = (
            line[: value.col_offset] + f'"{git_hash}"'.encode() + line[value.end_col_offset :]
        ).decode("utf-8")
        updated_content = "".join(lines)

        if dry_run:
            print(f"Dry run: Would update _git_commit_hash in {config_file_path} to {git_hash}")
            return True

        with open(config_file_path, 'w', encoding="utf-8", newline="") as file:
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
