# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import stat
import subprocess
import sys
from typing import Optional

import omni.ext

HERE = os.path.dirname(__file__)

# Path to the `exts/omni.warp.core` folder.
EXT_PATH = os.path.realpath(os.path.join(HERE, "..", "..", "..", ".."))

# Path to the `warp` link/folder located in `exts/omni.warp.core`.
LOCAL_LIB_PATH = os.path.join(EXT_PATH, "warp")

# Warp's core library path, relative to the `exts/omni.warp.core` folder.
LIB_PATH_REL = os.path.join("..", "..", "warp")

# Warp's core library path.
LIB_PATH_ABS = os.path.realpath(os.path.join(EXT_PATH, LIB_PATH_REL))


def _read_link(path: str) -> Optional[str]:
    try:
        dst_path = os.readlink(path)
    except Exception:
        # The given path doesn't exist or the link is invalid.
        return None

    if os.name == "nt" and dst_path.startswith("\\\\?\\"):
        return dst_path[4:]

    return dst_path


class OmniWarpCoreExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        # We want to make Warp's library available through this `omni.warp.core`
        # extension. Due to `extension.toml` expecting the Python package to be
        # located at `exts/omni.warp.core/warp`, one way to do achieve that is
        # to create a symlink to the actual `warp` folder.
        # However, symlinks tend to be annoying to deploy to Windows users since
        # they require special privileges, so we don't commit it as part of
        # our code repository and, instead, we try to create a link in one way
        # or another while the extension is being loaded.
        # Eventually, this link is only used/created when developers are loading
        # the extension directly from Warp's code repository.
        # End-users loading the published extension from Omniverse won't see or
        # need any link since the process that publishes the extension will
        # include a copy of Warp's library.

        # Try reading the content of the link.
        link_dst_path = _read_link(LOCAL_LIB_PATH)

        try:
            # We're using `os.stat()` instead of `os.path.is*()` since
            # we don't want to follow links, otherwise calling `os.path.isdir()`
            # would return `True` for links pointing to directories.
            file_stat = os.stat(LOCAL_LIB_PATH, follow_symlinks=False)
            file_stat_mode = file_stat.st_mode
        except FileNotFoundError:
            file_stat_mode = 0

        # Check if we have a directory.
        # Windows' junctions are a bit special and return `True` when
        # checked against `stat.S_ISDIR()`, so we also need to check that
        # the link is invalid, in which case we can safely assume that we have
        # an actual directory instead of a junction.
        is_dir = stat.S_ISDIR(file_stat_mode) and link_dst_path is None

        if not is_dir and link_dst_path not in (LIB_PATH_REL, LIB_PATH_ABS):
            # No valid folder/link corresponding to the library were found in
            # the extension, so let's try creating a new link.

            if stat.S_ISREG(file_stat_mode) or stat.S_ISLNK(file_stat_mode):
                # We have an invalid file/link, remove it.
                os.remove(LOCAL_LIB_PATH)

            try:
                os.symlink(LIB_PATH_REL, LOCAL_LIB_PATH, target_is_directory=True)
            except OSError as e:
                # On Windows, creating symbolic links might fail due to lack of
                # privileges. In that case, try creating a junction instead.
                if os.name == "nt":
                    # Junctions don't support relative paths so we need
                    # to use an absolute path for the destination.
                    cmd = ("mklink", "/j", LOCAL_LIB_PATH, LIB_PATH_ABS)
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True, check=True)
                else:
                    raise RuntimeError(f"Failed to create the symlink `{LOCAL_LIB_PATH}`") from e

            # Due to Kit's fast importer mechanism having already cached our
            # extension's Python path before we had a chance to create the link,
            # we need to update Python's standard `sys.path` for `import warp`
            # to work as expected.
            sys.path.insert(0, EXT_PATH)
