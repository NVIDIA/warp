# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the version files and native libraries collected for wheel packaging agree.

Every CI build job saves its own copy of ``VERSION.md``, ``warp/config.py``,
``warp/native/version.h``, and ``warp/native/exports.h`` as artifacts. A packaging job
that needs several build jobs gets all of those copies extracted over the same paths,
and GitLab does not define which one is extracted last. The result is only correct when
every copy is identical, so this check fails when:

* any of those files has CRLF line endings, as a Windows checkout can produce, or
* the versions in ``VERSION.md``, ``warp/config.py``, and ``warp/native/version.h``
  differ, or a platform's core native library was built with a different version. The
  native library refuses to load when its compiled-in version differs from
  ``warp.config.version``.

Run it from the repository root after the build artifacts and binaries are in place and
before any step that rewrites ``VERSION.md``.
"""

import argparse
import mmap
import re
import sys
from pathlib import Path

VERSION_FILES = ("VERSION.md", "warp/config.py", "warp/native/version.h", "warp/native/exports.h")

# The library whose compiled-in WP_VERSION_STRING is checked against warp.config.version
# when Warp loads, one name per platform.
CORE_LIBRARY_NAMES = ("warp.so", "warp.dll", "libwarp.dylib")


def read_version(path: Path, pattern: str) -> str | None:
    match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return match.group(1) if match else None


def library_embeds_version(path: Path, version: str) -> bool:
    # WP_VERSION_STRING is stored as a NUL-terminated string. Requiring a non-printable
    # byte before it stops a shorter version such as 1.20.0 from matching inside a longer
    # one such as 1.20.0.dev20261104.
    pattern = re.compile(rb"(?<![\x20-\x7e])" + re.escape(version.encode("ascii")) + rb"\x00")
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return pattern.search(data) is not None


def check(root: Path) -> list[str]:
    errors = []

    for name in VERSION_FILES:
        if b"\r" in (root / name).read_bytes():
            errors.append(f"{name} has CRLF line endings")

    version = (root / "VERSION.md").read_text(encoding="utf-8").splitlines()[0].strip()
    other_versions = {
        "warp/config.py": read_version(root / "warp/config.py", r'^version\s*:\s*str\s*=\s*"([^"]*)"'),
        "warp/native/version.h": read_version(root / "warp/native/version.h", r'^#define WP_VERSION_STRING "([^"]*)"'),
    }
    for name, other in other_versions.items():
        if other != version:
            errors.append(f"{name} has version {other!r}, but VERSION.md has {version!r}")

    libraries = sorted(p for p in (root / "warp" / "bin").rglob("*") if p.name in CORE_LIBRARY_NAMES)
    if not libraries:
        errors.append("no core native libraries found under warp/bin")
    for library in libraries:
        relative = library.relative_to(root).as_posix()
        if library_embeds_version(library, version):
            print(f"{relative}: built as version {version}")
        else:
            errors.append(f"{relative} was not built as version {version}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root (default: current directory)")
    args = parser.parse_args(argv)

    errors = check(args.root)
    sys.stdout.flush()
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
