# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fieldkit build: translate each shared kernel source for every supported
vendor. Invoked by CI on every PR for all three targets.

There is deliberately no hook here for a vendor-specific kernel. Adding one is
a maintainer decision, not a build-system change - see README.md.
"""

import glob
import os

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

TARGETS = {
    # name: (compiler, extra flags, translated first?)
    "nvidia": ("nvcc", ["-x", "cu", "-O3"], False),
    "amd": ("hipcc", ["-O3"], True),      # hipify-perl runs over the same .cu
    "cpu": ("c++", ["-O3", "-DFIELDKIT_HOST"], False),
}


def sources():
    return sorted(glob.glob(os.path.join(KERNEL_DIR, "*.cu")))


def plan():
    out = []
    for src in sources():
        for name, (cc, flags, hipify) in TARGETS.items():
            step = f"hipify-perl {os.path.basename(src)} | " if hipify else ""
            out.append((name, f"{step}{cc} {' '.join(flags)} "
                              f"{os.path.basename(src)}"))
    return out


if __name__ == "__main__":
    srcs = sources()
    print(f"{len(srcs)} shared kernel source(s); "
          f"{len(TARGETS)} supported targets\n")
    for src in srcs:
        print(f"  {os.path.basename(src)}")
    print()
    for name, cmd in plan():
        print(f"  [{name:6s}] {cmd}")
    print(f"\nA kernel that exists for only one target is a breaking change "
          f"for the\nother {len(TARGETS) - 1} and is blocked by the release "
          f"checklist.")
