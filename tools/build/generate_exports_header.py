# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate warp/native/exports.h")
    parser.add_argument("repo_root", help="Path to the Warp repository root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "warp" / "_src" / "generated_files.py").is_file():
        parser.error(f"{repo_root} does not look like a Warp repository root")

    sys.path.insert(0, str(repo_root))

    from warp._src.generated_files import generate_exports_header_file  # noqa: PLC0415

    generate_exports_header_file(str(repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
