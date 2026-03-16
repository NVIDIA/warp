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

"""Functions that generate derived source files (headers, stubs).

These functions have no build-toolchain dependencies (no CUDA, LLVM, etc.)
but do require the ``warp`` package (and transitively ``numpy``) to be
importable. They are used by ``build_lib.py``, ``build_docs.py``, and the
pre-commit hooks in ``tools/pre-commit-hooks/``.

Re-exports ``export_builtins`` and ``export_stubs`` from
``warp._src.context`` for convenience.
"""

from __future__ import annotations

import datetime
import os

from warp._src.context import export_builtins, export_stubs

__all__ = [
    "export_builtins",
    "export_stubs",
    "generate_exports_header_file",
    "generate_version_header",
]


def _c_copyright_header(year: int | str) -> str:
    """Return a C-style copyright/license block for generated headers."""
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


def generate_version_header(base_path: str, version: str) -> None:
    """Generate version.h with WP_VERSION_STRING macro.

    Only writes the file when the content has actually changed, so that file
    modification timestamps are preserved and downstream build systems don't
    trigger unnecessary rebuilds.

    NOTE: The pre-commit hook ``check_version_consistency.py`` has an inlined
    copy of this function (``_regenerate_version_header``) so it can run
    without importing ``warp``.  Keep the two in sync.
    """
    version_header_path = os.path.join(base_path, "warp", "native", "version.h")

    new_content = _c_copyright_header(datetime.date.today().year)
    new_content += "#ifndef WP_VERSION_H\n"
    new_content += "#define WP_VERSION_H\n\n"
    new_content += f'#define WP_VERSION_STRING "{version}"\n\n'
    new_content += "#endif  // WP_VERSION_H\n"

    try:
        with open(version_header_path) as f:
            if f.read() == new_content:
                print(f"{version_header_path} is up to date (version {version})")
                return
    except FileNotFoundError:
        pass

    with open(version_header_path, "w") as f:
        f.write(new_content)

    print(f"Generated {version_header_path} with version {version}")


def generate_exports_header_file(base_path: str) -> None:
    """Generate warp/native/exports.h with host-side wrappers for built-in functions."""
    export_path = os.path.join(base_path, "warp", "native", "exports.h")

    with open(export_path, "w") as f:
        f.write(_c_copyright_header(2022))
        export_builtins(f)

    print(f"Finished writing {export_path}")
