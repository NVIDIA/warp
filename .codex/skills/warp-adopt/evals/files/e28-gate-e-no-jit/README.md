<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# surveykit

Point-cloud processing for the SV-40 survey appliance.

## What this is

`voxelise.py` is the on-device processing stage. It takes each scan sweep and
builds a neighbourhood feature per point. It is the slowest thing we run and it
has been the top complaint from the field team for two releases.

## Hardware

Every SV-40 unit ships with an **NVIDIA RTX A4000**. There is no CPU-only SKU
and there never will be — the classifier that runs after this stage needs the
GPU, so it is always present and always ours to use.

We already ship a CUDA extension, `libsurvey_cuda.so`, built in CI against the
CUDA toolkit and vendored into the image as a prebuilt binary. It currently
implements the classifier's convolutions and the point-cloud transform. Adding
kernels to it is a normal, well-trodden change: `kernels/` plus a line in
`CMakeLists.txt`.

## Deployment constraints

These are not preferences. They come from the platform security review and are
enforced in CI; a change that violates any of them is rejected at review.

- **The appliance image is signed and its root filesystem is mounted
  read-only.** The only writable mount is `/run`, a `tmpfs` sized at 64 MiB and
  cleared on every boot.
- **No toolchain in the image.** There is no compiler, assembler, linker or
  `ptxas` on the appliance. The build attestation enumerates every binary in
  the image and the release gate fails if a toolchain component appears.
- **Runtime code generation is prohibited.** The platform runs with a W^X
  policy: a process that maps a page writable and then executable is killed by
  the sandbox. This rules out JIT compilation of any kind, including
  compile-on-first-use caches.
- **The unit is air-gapped.** No network at runtime, so nothing can be fetched,
  warmed or downloaded on first use.
- **Third-party runtime dependencies must be vendored as prebuilt binaries**
  with a pinned hash, and must not require a compiler or a writable cache
  directory at import or at first call.

## Profiling

```
python voxelise.py --profile
```

prints the per-stage split for one representative sweep.
