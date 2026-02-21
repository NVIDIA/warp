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

"""Pure-Python functions that generate derived source files (headers, stubs).

These functions have no build-toolchain dependencies (no CUDA, LLVM, etc.) and
can be safely imported by lightweight scripts such as pre-commit hooks and CI
checks without pulling in the rest of the build machinery.

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


def generate_version_header(base_path: str, version: str) -> None:
    """Generate version.h with WP_VERSION_STRING macro."""
    version_header_path = os.path.join(base_path, "warp", "native", "version.h")
    current_year = datetime.date.today().year

    copyright_notice = f"""/*
 * SPDX-FileCopyrightText: Copyright (c) {current_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    with open(version_header_path, "w") as f:
        f.write(copyright_notice)
        f.write("#ifndef WP_VERSION_H\n")
        f.write("#define WP_VERSION_H\n\n")
        f.write(f'#define WP_VERSION_STRING "{version}"\n\n')
        f.write("#endif  // WP_VERSION_H\n")

    print(f"Generated {version_header_path} with version {version}")


def generate_exports_header_file(base_path: str) -> None:
    """Generates warp/native/exports.h, which lets built-in functions be callable from outside kernels."""
    export_path = os.path.join(base_path, "warp", "native", "exports.h")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    try:
        with open(export_path, "w") as f:
            copyright_notice = """/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            f.write(copyright_notice)
            export_builtins(f)

        print(f"Finished writing {export_path}")
    except FileNotFoundError:
        print(f"Error: The file '{export_path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied. Unable to write to '{export_path}'.")
    except OSError as e:
        print(f"Error: An OS-related error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
