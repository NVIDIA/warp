# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Script to update VERSION.md and config.py with a development version number."""

import os
import datetime
import re

def increment_minor(version):
    """Return a incremented version based on the input semantic version."""

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

def write_new_version_to_config(config_file_path, new_version):
    """Writes version into config.py."""

    with open(config_file_path, "r") as file:
        content = file.read()

    # Define the regex to match the version assignment
    pattern = r'^(version\s*:\s*str\s*=\s*")(\d+\.\d+\.\d+)(")$'

    # Replace the old version with the new version
    updated_content = re.sub(pattern, rf"\g<1>{new_version}\g<3>", content, flags=re.MULTILINE)

    with open(config_file_path, 'w') as file:
        file.write(updated_content)

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    root_path = os.path.abspath(os.path.join(here, "..", "..", ".."))

    version_file = os.path.join(root_path, "VERSION.md")
    config_file = os.path.join(root_path, "warp", "config.py")

    with open(version_file, "r") as file:
        base_version = file.readline().strip()

    # Increment the minor version so the nightly build is considered newer than the latest release
    base_version_incremented = increment_minor(base_version)

    # See https://peps.python.org/pep-0440/#developmental-releases
    dateint = datetime.date.today().strftime("%Y%m%d")

    dev_version_string = f"{base_version_incremented}.dev{dateint}"

    print(f"Updating version to {dev_version_string}...")

    # Update VERSION.md and /warp/config.py with the new version
    with open(version_file, 'w') as file:
        file.write(dev_version_string + '\n')

    write_new_version_to_config(config_file, dev_version_string)
