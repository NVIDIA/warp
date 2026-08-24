# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import USD Tf before the Warp test runner and its workers initialize."""

from __future__ import annotations

import json
import os
import sys
import time

try:
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
