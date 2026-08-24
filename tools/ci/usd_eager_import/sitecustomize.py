# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import USD Tf before the Warp test runner and its workers initialize."""

from __future__ import annotations

import json
import os
import sys
import time

try:
    import pxr  # noqa: F401

    variable_name = "PXR_USD_WINDOWS_DLL_PATH"
    search_paths = [path for path in os.environ.get(variable_name, "").split(os.pathsep) if path]
    normalized_paths = {os.path.normcase(os.path.abspath(path)) for path in search_paths}

    for relative_path in ("bin", os.path.join("Library", "bin")):
        runtime_directory = os.path.join(sys.prefix, relative_path)
        normalized_path = os.path.normcase(os.path.abspath(runtime_directory))
        if os.path.isdir(runtime_directory) and normalized_path not in normalized_paths:
            search_paths.append(runtime_directory)
            normalized_paths.add(normalized_path)

    os.environ[variable_name] = os.pathsep.join(search_paths)

    from pxr import Tf
except BaseException as error:
    print(
        json.dumps(
            {
                "kind": "usd-eager-import-error",
                "pid": os.getpid(),
                "time": time.time(),
                "error_type": type(error).__name__,
                "error": repr(error),
                "winerror": getattr(error, "winerror", None),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    # Python's site module otherwise reports and ignores sitecustomize errors.
    os._exit(86)

print(
    json.dumps(
        {
            "kind": "usd-eager-import",
            "pid": os.getpid(),
            "time": time.time(),
            "tf_module": getattr(Tf, "__file__", None),
        },
        sort_keys=True,
    ),
    flush=True,
)
