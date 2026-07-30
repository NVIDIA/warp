<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# fieldkit

Volumetric field resampling for the acquisition pipeline.

## Supported accelerators

`fieldkit` ships **one** kernel source per operation, in `kernels/`. The build
translates that single source for every supported vendor:

| Vendor | Toolchain | Status | Covered by CI |
|---|---|---|---|
| NVIDIA | CUDA (`nvcc`) | supported | yes, every PR |
| AMD | HIP (`hipcc`, auto-translated from the same `.cu`) | supported | yes, every PR |
| CPU | portable C++ fallback | supported | yes, every PR |

Roughly a third of deployments are on AMD parts (see `docs/deployments.md`);
the AMD path is not aspirational and is not a tier-2 target. A change that
works on only one vendor is a **breaking change** for the other two, and the
release checklist blocks it.

## Adding a kernel

New kernels go in `kernels/` as a single `.cu` that the build translates. We do
not accept vendor-specific kernel sources: the cost of a second source is not
the writing, it is that every future change has to be made, reviewed, tested
and released three times instead of once.

If a proposal cannot be expressed in the shared source, it needs an explicit
decision from the maintainers about who owns the extra implementation, how it
is tested on hardware we do not have in CI, and what happens on the vendors it
does not cover.

## Performance

`resample_field` dominates a batch. Run `python pipeline.py --profile` for the
current split.
