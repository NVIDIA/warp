#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package a deployed LLVM SDK tree into a deterministic tar.xz plus build-info JSON.

Determinism: entries are added in sorted order with uid/gid 0, empty
user/group names, a fixed mtime, and normalized file modes, so rebuilding
the same tree yields a byte-identical archive. This makes the release
policy (never clobber a published asset; bump the bundle revision instead)
auditable by hash comparison.
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile

# 2000-01-01T00:00:00Z; any fixed value works, zero confuses some tools.
_FIXED_MTIME = 946684800

# Conan MSVC compiler.version -> Visual Studio toolset tag. 193 and 194
# (VS 2022 and its 17.10+ update) both ship the v143 toolset.
_VS_TOOLSETS = {"190": "vs140", "191": "vs141", "192": "vs142", "193": "vs143", "194": "vs143"}
# Conan arch setting -> arch tag used in platform tags and asset names.
_ARCH_TAGS = {"x86_64": "x86_64", "armv8": "aarch64"}


def parse_profile_settings(profile_path):
    """Parse the [settings] section of a Conan profile into a dict."""
    settings = {}
    section = None
    with open(profile_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                section = line.strip("[]")
            elif section == "settings" and "=" in line:
                key, _, value = line.partition("=")
                settings[key.strip()] = value.strip()
    return settings


def platform_tag(settings):
    """Derive the platform tag string from resolved profile settings."""
    os_name = settings["os"]
    arch = _ARCH_TAGS[settings["arch"]]
    if os_name == "Linux":
        return f"linux-{arch}-gcc{settings['compiler.version']}-cxxabi0"
    if os_name == "Windows":
        toolset = _VS_TOOLSETS.get(settings["compiler.version"])
        if toolset is None:
            raise ValueError(
                f"MSVC compiler.version {settings['compiler.version']!r} has no toolset tag; "
                "add it to _VS_TOOLSETS in tools/llvm/package_sdk.py"
            )
        win_arch = "arm64" if arch == "aarch64" else arch
        return f"windows-{win_arch}-{toolset}-static-mt"
    if os_name == "Macos":
        if settings["arch"] != "armv8":
            raise ValueError(
                f"macOS SDKs are arm64-only by design; got arch {settings['arch']!r} "
                "(add a profile, CI leg, and tag mapping before packaging other arches)"
            )
        deployment = settings["os.version"].split(".")[0]
        return f"macos-arm64-macos{deployment}"
    raise ValueError(f"Unsupported profile os: {os_name}")


def _tar_filter(info):
    """Normalize a tar entry's ownership, mtime, and mode for determinism."""
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = _FIXED_MTIME
    if info.isfile():
        info.mode = 0o755 if info.mode & 0o100 else 0o644
    elif info.isdir():
        info.mode = 0o755
    return info


def create_archive(sdk_dir, archive_path):
    """Write the SDK tree to a deterministic tar.xz archive."""
    entries = []
    for root, dirs, files in os.walk(sdk_dir):
        dirs.sort()
        for name in sorted(dirs) + sorted(files):
            entries.append(os.path.join(root, name))
    # Write-only: this script never extracts archives, so decompression-bomb
    # concerns do not apply. Consumers extract only after SHA-256 verification
    # against digests pinned at creation time (see the packman project file).
    with tarfile.open(archive_path, "w:xz") as tar:
        for path in sorted(entries):
            arcname = os.path.relpath(path, sdk_dir)
            tar.add(path, arcname=arcname, recursive=False, filter=_tar_filter)


def sha256_of(path):
    """Return the SHA-256 hex digest of the file at path."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    """Package the SDK tree and emit the archive and build-info JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-dir", required=True, help="Deployed llvm-sdk tree")
    parser.add_argument("--profile", required=True, help="Conan host profile used for the build")
    parser.add_argument(
        "--llvm-version", required=True, help="LLVM version string embedded in the asset name (e.g. 18.1.8)"
    )
    parser.add_argument(
        "--bundle-revision",
        required=True,
        help="Warp bundle revision; bump to republish without clobbering a prior asset",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write the archive and build-info JSON")
    parser.add_argument("--recipe-sha", default="", help="Git SHA of the recipe checkout")
    parser.add_argument("--image-digest", default="", help="Container image digest, if any")
    parser.add_argument("--toolchain-info", default="", help="Free-form toolchain description")
    parser.add_argument("--conan-version", default="", help="Conan version used for the build")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.sdk_dir):
        parser.error(f"--sdk-dir does not exist or is not a directory: {args.sdk_dir}")

    settings = parse_profile_settings(args.profile)
    tag = platform_tag(settings)
    asset_name = f"clang+llvm-warp@{args.llvm_version}-{tag}-warp.{args.bundle_revision}.tar.xz"

    os.makedirs(args.output_dir, exist_ok=True)
    archive_path = os.path.join(args.output_dir, asset_name)
    create_archive(args.sdk_dir, archive_path)
    digest = sha256_of(archive_path)

    with open(args.profile, encoding="utf-8") as f:
        profile_text = f.read()
    build_info = {
        "schema_version": 1,
        "package": "clang-warp",
        "llvm_version": args.llvm_version,
        "bundle_revision": args.bundle_revision,
        "conan_version": args.conan_version,
        "platform_tag": tag,
        "asset_name": asset_name,
        "sha256": digest,
        "size_bytes": os.path.getsize(archive_path),
        "profile": profile_text,
        "recipe_sha": args.recipe_sha,
        "image_digest": args.image_digest,
        "toolchain_info": args.toolchain_info,
    }
    info_path = archive_path.replace(".tar.xz", ".buildinfo.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=2)
        f.write("\n")

    print(asset_name)
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
