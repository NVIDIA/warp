# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
import io
import os
from collections.abc import Callable

from warp._src.context import export_builtins, export_stubs

__all__ = [
    "export_builtins",
    "export_stubs",
    "generate_exports_header_file",
    "generate_stubs_file",
    "generate_version_header",
]


def _c_copyright_header(year: int | str) -> str:
    """Return a C-style copyright/license block for generated headers."""
    return f"""// SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"""


def _write_generated_file(path: str, render: Callable[[io.TextIOBase], None]) -> bool:
    """Render a generated text file and write it only when its content changed.

    The file is always written with LF line endings, and an existing file with CRLF
    line endings counts as changed. CI build jobs on every platform save these files
    as artifacts, so they must be byte-identical on Windows and Linux.

    Returns:
        True if the file was updated, or False if it was already current.
    """
    output = io.StringIO()
    render(output)
    new_content = output.getvalue()

    try:
        with open(path, encoding="utf-8", newline="") as f:
            if f.read() == new_content:
                return False
    except FileNotFoundError:
        pass

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)
    return True


def generate_version_header(base_path: str, version: str) -> bool:
    """Generate version.h with WP_VERSION_STRING macro.

    Only writes the file when the content has actually changed, so that file
    modification timestamps are preserved and downstream build systems don't
    trigger unnecessary rebuilds.

    NOTE: The pre-commit hook ``check_version_consistency.py`` has an inlined
    copy of this function (``_regenerate_version_header``) so it can run
    without importing ``warp``.  Keep the two in sync.

    Returns:
        True if the file was updated, or False if it was already current.
    """
    version_header_path = os.path.join(base_path, "warp", "native", "version.h")

    def render(file: io.TextIOBase) -> None:
        file.write(_c_copyright_header(datetime.date.today().year))
        file.write("#ifndef WP_VERSION_H\n")
        file.write("#define WP_VERSION_H\n\n")
        file.write(f'#define WP_VERSION_STRING "{version}"\n\n')
        file.write("#endif  // WP_VERSION_H\n")

    return _write_generated_file(version_header_path, render)


def generate_exports_header_file(base_path: str) -> bool:
    """Generate warp/native/exports.h with host-side wrappers for built-in functions.

    Returns:
        True if the file was updated, or False if it was already current.
    """
    export_path = os.path.join(base_path, "warp", "native", "exports.h")

    def render(file: io.TextIOBase) -> None:
        file.write(_c_copyright_header(2022))
        export_builtins(file)

    return _write_generated_file(export_path, render)


def generate_stubs_file(base_path: str) -> bool:
    """Generate ``warp/__init__.pyi`` with type stubs for the public API.

    Returns:
        True if the file was updated, or False if it was already current.
    """
    stub_path = os.path.join(base_path, "warp", "__init__.pyi")
    return _write_generated_file(stub_path, export_stubs)
