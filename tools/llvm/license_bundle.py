# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble the LLVM SDK license and attribution bundle."""

import shutil
from pathlib import Path


class LicenseBundleError(Exception):
    """A required license source is missing or ambiguous."""


_COPIED_LICENSES = {
    "llvm/LICENSE.TXT": "llvm/LICENSE.TXT",
    "clang/LICENSE.TXT": "clang/LICENSE.TXT",
    "llvm/lib/Support/BLAKE3/LICENSE": "third-party/BLAKE3/LICENSE",
    "llvm/lib/Support/COPYRIGHT.regex": "third-party/regex/COPYRIGHT",
}

# Markers locate a notice by the block comment it sits in, so each must be
# unique within its file. Prefer strings that identify the upstream project or
# name its license: copyright years and maintainer emails get refreshed
# whenever LLVM re-syncs a vendored source, which would fail the build for a
# cosmetic edit rather than a real change in licensing.
_EXTRACTED_NOTICES = {
    "third-party/MD5/NOTICE.txt": (
        "llvm/lib/Support/MD5.cpp",
        "MD5 Message-Digest Algorithm (RFC 1321)",
    ),
    "third-party/xxhash/NOTICE.txt": (
        "llvm/lib/Support/xxhash.cpp",
        "xxHash - Extremely Fast Hash algorithm",
    ),
    "third-party/strlcpy/NOTICE.txt": (
        "llvm/lib/Support/regstrlcpy.c",
        "This code is derived from OpenBSD's libc",
    ),
    "third-party/Unicode/ConvertUTF-NOTICE.txt": (
        "llvm/lib/Support/ConvertUTF.cpp",
        "Unicode, Inc. All rights reserved.",
    ),
    "third-party/Unicode/UnicodeData-NOTICE.txt": (
        "llvm/lib/Support/UnicodeNameToCodepointGenerated.cpp",
        "UNICODE, INC. LICENSE AGREEMENT - DATA FILES AND SOFTWARE",
    ),
}

LICENSE_BUNDLE_PATHS = (
    "LLVM-ATTRIBUTION.txt",
    *_COPIED_LICENSES.values(),
    *_EXTRACTED_NOTICES,
)

_ATTRIBUTION = """LLVM and Clang Attribution

This SDK contains LLVM and Clang from LLVM Project release {llvm_version}.
Source: https://github.com/llvm/llvm-project/releases/tag/llvmorg-{llvm_version}

Copyright in LLVM Project code is held by the respective contributors.
LLVM and Clang are licensed under Apache-2.0 WITH LLVM-exception.
See llvm/LICENSE.TXT and clang/LICENSE.TXT for the complete license terms.

LLVM Project copyright, license, and patent policy:
https://llvm.org/docs/DeveloperPolicy.html#copyright-license-and-patents
"""


def extract_c_comment(path, marker):
    """Extract the unique C block comment containing ``marker``."""
    if not isinstance(marker, str) or not marker:
        raise LicenseBundleError(f"{path}: extraction marker must be a non-empty string")
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except FileNotFoundError as exc:
        raise LicenseBundleError(f"missing license source: {path}") from exc
    occurrences = text.count(marker)
    if occurrences == 0:
        raise LicenseBundleError(f"{path}: marker {marker!r} not found; re-audit the license manifest")
    if occurrences > 1:
        raise LicenseBundleError(
            f"{path}: marker {marker!r} is ambiguous ({occurrences} occurrences); pick a more specific one"
        )
    marker_offset = text.index(marker)
    start = text.rfind("/*", 0, marker_offset + 1)
    preceding_end = text.rfind("*/", 0, marker_offset)
    end = text.find("*/", marker_offset)
    if start < 0 or end < 0 or preceding_end > start:
        raise LicenseBundleError(f"{path}: marker {marker!r} is not inside a C block comment")
    return text[start : end + 2].rstrip() + "\n"


def assemble_license_bundle(source_root, license_root, llvm_version) -> None:
    """Assemble the fixed LLVM SDK license manifest from its source tree."""
    source_root = Path(source_root)
    license_root = Path(license_root)

    for source_relative, output_relative in _COPIED_LICENSES.items():
        source = source_root / source_relative
        if not source.is_file():
            raise LicenseBundleError(f"missing license source: {source}")
        output = license_root / output_relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)

    for output_relative, (source_relative, marker) in _EXTRACTED_NOTICES.items():
        notice = extract_c_comment(source_root / source_relative, marker)
        output = license_root / output_relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(notice.encode("utf-8"))

    attribution = license_root / "LLVM-ATTRIBUTION.txt"
    attribution.parent.mkdir(parents=True, exist_ok=True)
    attribution.write_text(_ATTRIBUTION.format(llvm_version=llvm_version), encoding="utf-8")
