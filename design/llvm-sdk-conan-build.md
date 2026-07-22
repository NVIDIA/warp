# LLVM SDK Conan Build Pipeline

**Status**: Implemented

**Issue**: none (companion to the
[LLVM SDK distribution plan](https://gist.github.com/shi-eric/4eeb78d42309eab47c6c3c79a01ed6a6))

## Motivation

Warp embeds a custom LLVM/Clang flavor (clang-only static libraries with the NVPTX backend,
self-contained, size-optimized) as its CPU-JIT compiler.
The distribution plan linked above moves these prebuilt SDKs to GitHub Releases on `NVIDIA/warp`,
but its build phase left open how the SDKs get built: a public Conan recipe, or container-based
Python/CMake scripts like `docker/warp-builder` and `build_llvm.py`.

Three overlapping build definitions exist today:

1. `build_llvm.py`: plain CMake driver, used by `build_lib.py --build-llvm` as a user-facing fallback.
2. `docker/warp-builder/Dockerfile`: mirrors those flags inside manylinux containers to bake `/opt/llvm`
   into builder images.
3. An NVIDIA-internal Conan recipe (`clang-warp`): the most complete of the three, with per-platform
   profiles, a two-pass native-tablegen cross build for windows-aarch64, size optimization,
   and library pruning.

This document specifies the build side of the distribution plan: a public, standalone Conan recipe
under `tools/llvm/` executed entirely on GitHub Actions runners.
The release model (tag namespace, asset naming, Packman consumption) originates in the distribution
plan; the operational release process is documented in the Release process section below,
since the workflow in this repository now implements most of it.

## Requirements

| ID  | Requirement                                                                                | Priority | Notes                                          |
| --- | ------------------------------------------------------------------------------------------ | -------- | ---------------------------------------------- |
| R1  | One build definition covers linux-x86_64/aarch64, windows-x86_64/arm64, macos-arm64        | Must     | Same recipe, different profiles                |
| R2  | Reproducible by external contributors without NVIDIA-internal remotes or tooling           | Must     | pip-installed Conan, public sources only       |
| R3  | Linux SDKs are safe to link into manylinux_2_28 (x86_64) / manylinux_2_34 (aarch64) wheels | Must     | Built inside the matching manylinux containers |
| R4  | windows-x86_64 SDK remains linkable with VS2019 (v142) and newer                           | Must     | Preserves Warp's documented VS2019+ floor      |
| R5  | Consumers never interact with Conan; the deployed tree is the product                      | Must     | Deployer emits a normalized install tree       |
| R6  | The exact CI build is runnable locally with the same commands                              | Must     | `pip install conan` + `conan create`           |
| R7  | Artifacts and metadata plug into the distribution plan's release pipeline unchanged        | Must     | Archive naming, build-info JSON, SBOM          |
| R8  | windows-arm64 is built but cannot block the platform matrix while experimental             | Should   | `continue-on-error`, outside required matrix   |

**Non-goals**:
the release/publish machinery beyond draft-release assembly
(publishing, promotion, SBOM generation, and the Packman consumer manifest stay with the distribution plan);
changes to `build_llvm.py` or any mechanism to keep it in sync with the recipe (drift is accepted;
the recipe is designed as if `build_llvm.py` did not exist);
consolidating the `warp-builder` Dockerfile onto published SDKs (future work in the distribution plan);
publishing Conan packages to any registry;
deciding the fate of the internal `clang-warp` pipeline.

## Design

### Approach

A single stripped-down Conan recipe under `tools/llvm/` is the sole definition of the SDK build.
Every platform runs the same job shape on GitHub Actions:
install a pinned Conan with pip, run `conan create` with a checked-in profile,
deploy a normalized Conan-free tree, archive it deterministically, and upload.

Context that drove the decision:

- Rebuilds are rare (roughly 1-2 times a year, on LLVM version bumps), so Conan-registry build caching
  is worthless here and its absence costs nothing.
- The recipe is deliberately dependency-free (no zlib, zstd, terminfo, or xml2), so the Conan graph
  contains only the LLVM sources. The recipe declares no tool_requires and checks the
  system-provided cmake/ninja at build time, so Conan downloads nothing from a registry.
- The hardest platform, windows-aarch64 cross-compilation with a native tablegen pre-pass,
  is already solved in the internal recipe and carries over.
- The internal pipeline's future is undecided, so the public build system must stand alone.

### Repository layout

```
tools/llvm/
├── README.md              # build + local-reproduction instructions per platform
├── conanfile.py           # the recipe (see below)
├── conandata.yml          # source URL + SHA-256 per LLVM version, patch list
├── profiles/
│   ├── linux-x86_64       # gcc, libstdc++, pre-C++11 ABI, -fabi-version=13
│   ├── linux-aarch64
│   ├── macos-arm64        # apple-clang, libc++, deployment target 11.0
│   ├── windows-x86_64     # MSVC v142 (compiler.version=192), static CRT
│   └── windows-arm64      # MSVC v143, static CRT; also the -pr:b for native builds
├── deployers/
│   └── llvm_sdk.py        # conan deployer -> normalized, Conan-free install tree
├── package_sdk.py         # deterministic tar.xz + build-info JSON
├── check_sdk.py           # tree/ABI/CRT sanity checks
├── ci/                    # scripts the workflow runs (build-linux.sh, smoke-linux.sh)
└── test_package/          # compile + link + JIT-execute smoke test
```

Native windows-arm64 builds pass `windows-arm64` as both host and build profile.
The cross fallback from an x64 machine passes `-pr:h windows-arm64 -pr:b windows-x86_64`;
the recipe detects `cross_building()` and runs the tablegen pre-pass automatically.

### The recipe, designed standalone

The recipe hardcodes Warp's flavor rather than exposing the upstream option matrix:

- **Options: nearly none.** Clang-only projects, static libraries, no external dependencies, no tools.
  These are code, not options.
  The kept knobs have real use cases: `build_type` (a RelWithDebInfo/Debug SDK for debugging JIT issues)
  and `targets` (defaulting to the host backend plus NVPTX).
  Every deleted option is a configuration that cannot silently produce a wrong SDK.
- **No component-graph machinery.** The internal recipe spends roughly 250 lines parsing CMake graphviz
  output into per-component Conan metadata for Conan consumers.
  This SDK has no Conan consumers: the deployer tree is the product, and Warp links the full set with
  `--start-group`.
  `package_info()` reduces to `collect_libs()`, which is sufficient for `test_package`.
- **`conandata.yml`** pins the source release URL and SHA-256 per LLVM version.
- **`validate()`** enforces platform and toolset floors (for example MSVC `compiler.version=192`
  on windows-x86_64, see R4).
- **`generate()`** owns settings not expressible in profiles: the MinSizeRel-style flag overrides
  (`/O1` on MSVC, `-Oz`/`-Os` elsewhere; these libraries ship inside PyPI wheels, so size wins over
  compiler throughput) and ABI flags such as `-fabi-version=13` on Linux/gcc.
- **`build()`** keeps the two-pass cross build: pass 1 compiles native x64 tablegen tools,
  pass 2 cross-compiles LLVM with `LLVM_NATIVE_TOOL_DIR` pointing at them.
- **`package()`** keeps the pruning of binaries and verified-unused subsystem libraries
  (for example the clang static analyzer archives, which the JIT never links),
  preserving the relink-verification rationale in comments.

### Profiles and toolchains

The environment provides the toolchain; the profile describes it and pins the ABI.

- The recipe declares no tool_requires at all: the container, runner, or developer machine provides
  cmake and ninja, and the recipe checks their presence and minimum versions at the start of `build()`.
  Nothing is fetched from ConanCenter at build time.
  (`[platform_tool_requires]` was considered but rejected: it forces each checked-in profile to claim
  exact tool versions that legitimately vary between environments.)
- Linux profiles encode the manylinux-wheel ABI contract:
  `compiler.libcxx=libstdc++` (pre-C++11 ABI) plus `-fabi-version=13`,
  with the gcc version matching what the chosen manylinux image ships.
  The glibc floor itself comes from building inside `manylinux_2_28_x86_64` / `manylinux_2_34_aarch64`
  containers; the profile and the recorded image digest jointly define the environment.
- The macos-arm64 profile pins deployment target 11.0 and libc++,
  matching the existing `arm64-apple-macos11` SDK.
- windows-x86_64 pins the v142 toolset (`compiler.version=192`).
  MSVC binary compatibility is directional: a final link must use a toolset at least as new as the
  newest toolset that compiled any object in it.
  A v143-built SDK would therefore be unlinkable from VS2019, breaking Warp's documented VS2019+ floor.
  LLVM 21 and 22 still support VS2019 as a host compiler, so v142 remains upstream-supported.
  The v142 component is not preinstalled on `windows-2022` runners and is added by a VS installer step.
  When Warp's Windows floor moves to VS2022 (anticipated with a future CUDA toolkit support-policy change),
  flip the profile to the v143 toolset and drop the CI v142 installer step and the vs_version conf;
  `validate()` makes this a one-line change.
- windows-arm64 uses v143. The SDK is new, has no existing consumers, and v142 ARM64 support is weak;
  its documented floor is VS2022.
- Conan itself is pinned to one exact version, installed everywhere (CI and local) with
  `pip install conan==<pinned>`.
  The official `conan-io/setup-conan` action was considered and rejected: it only helps the native
  runner jobs (the Linux builds run inside containers), its cache features are worthless at this
  cadence, and it is not a GitHub-verified-creator action, so plain pip keeps all environments identical
  with one less third-party action to pin and audit.

### CI architecture

All jobs run on GitHub Actions runners:

| Platform       | Runner                          | Environment                        |
| -------------- | ------------------------------- | ---------------------------------- |
| linux-x86_64   | `ubuntu-latest`                 | `manylinux_2_28_x86_64` container  |
| linux-aarch64  | `ubuntu-24.04-arm`              | `manylinux_2_34_aarch64` container |
| windows-x86_64 | `windows-2022`                  | Native, VS 2022 + v142 toolset     |
| macos-arm64    | `macos-15`                      | Native, Xcode                      |
| windows-arm64  | `windows-11-arm` (experimental) | Native, VS 2022                    |

Running everything on GitHub Actions means one workflow file defines the whole matrix,
every platform build is publicly reproducible (the distribution plan's "public recipe execution"
validation is the build itself), GitHub artifact attestations cover all platforms uniformly,
and nothing depends on internal infrastructure.
Attestation and SBOM generation are owned by the distribution plan's release pipeline,
not these build jobs.

Every job has the same shape:

1. `pip install conan==<pinned>` (plus cmake/ninja via pip where the environment lacks them).
2. `conan create tools/llvm --version <llvm-version> -pr:h <profile> -pr:b <profile>`;
   `test_package` compiles, links, and (except when cross-compiling) runs automatically.
3. Deployer produces the normalized SDK tree; deterministic `tar.xz` with the distribution plan's
   asset naming.
4. Emit a build-info JSON fragment (recipe git SHA, resolved profile, container image digest,
   toolchain and Conan versions) for the promotion job to merge.
5. Upload as a job artifact for the release pipeline.
   On `platforms=all` dispatches, a fully green build+smoke matrix automatically assembles the
   draft release: the archives, a `SHA256SUMS` file, and the merged build-info document,
   under the derived tag `llvm-sdk-<version>-warp.<revision>`.
   SBOM generation and the post-publish packman verification remain owned by the
   distribution plan's promotion step.

Triggering is manual only (`workflow_dispatch`), matching the rebuild cadence;
SDK builds never run in PR pipelines.
The windows-arm64 job is `continue-on-error` (R8).

Feasibility notes: the `build-warp-builder-images` workflow already compiles LLVM 21 on
`ubuntu-latest` and `ubuntu-24.04-arm`, so Linux capacity is proven.
Known risks from the distribution plan carry over: roughly 14 GB of free disk on hosted runners and
4-core build times on the Windows jobs (the 6-hour job limit is the ceiling).
Linux jobs report disk usage; the escape hatch if a build outgrows hosted runners is
larger GitHub-hosted runners, not a return to internal CI.

### Deployer and packaging

The deployer (`deployers/llvm_sdk.py`, a standard Conan `deploy()` hook) copies the package into a
normalized tree: `include/llvm`, `include/clang`, flat `lib/` with static libraries only,
and `licenses/` with LLVM's LICENSE.TXT and third-party notices.
Conan artifacts are stripped.
The tree shape matches what `build_llvm.py:fetch_prebuilt_libraries` and `WarpDependencies.cmake`
already expect, so consumer code needs no changes when the download source moves.

Archives are created deterministically (sorted entries, fixed mtimes, numeric owners, `tar.xz`),
so rebuilding the same recipe revision in the same environment yields a byte-comparable archive.
This makes the distribution plan's never-clobber, bump-`warp.N` policy auditable.
The toolchain/ABI segment of the asset name is derived from the profile rather than hand-written.

### Release process

The build workflow owns everything up to a draft release; publishing is a human promotion decision.
Expected cadence is one or two SDK releases a year, on LLVM version bumps.

**Roles and permissions.**
Anyone with repository write access can dispatch builds and publish drafts.
The workflow itself runs read-only (`permissions: contents: read`);
only the `publish-draft` job elevates to a job-scoped `contents: write` token.

**1. Dispatch.**
A maintainer runs the `Build LLVM SDK` workflow (`workflow_dispatch` only; never on push or PR) with:
`llvm_version` (empty selects the newest version in `conandata.yml`; explicit values are validated against it),
`bundle_revision` (see corrections below),
and `platforms`.
The `validate-inputs` job rejects unknown platform tokens and malformed revisions.
There is no release input: every green `platforms=all` run assembles a draft release under the
derived tag `llvm-sdk-<version>-warp.<revision>` (re-runs replace their own draft;
published releases are never touched), so drafting requires no human choice and cannot be
mistyped. Partial-platform dispatches build and validate without drafting,
which is the mode for iterating on a single platform.

**2. Build and validate.**
Five platform builds (windows-arm64 non-blocking while experimental), each running the recipe's
JIT `test_package`, the deployer, `check_sdk.py`, and deterministic packaging;
then per-platform consumer smoke jobs build warp-clang from the artifacts and JIT a CPU kernel,
with the Linux legs running inside the manylinux containers at the real glibc floor.

**3. Draft assembly (automatic).**
When every required stage is green, the `publish-draft` job downloads the artifacts and creates a
draft release containing the platform archives, `SHA256SUMS`,
and the merged `llvm-source-and-build-info.json`.
Before drafting it asserts the four required platform archives are present by name;
windows-arm64 is optional (warning only) while experimental, so a missing required archive
fails the job rather than producing an incomplete draft.
Draft semantics: visible only to users with repo access, assets are not anonymously downloadable,
and the git tag materializes only when the draft is published.
The `llvm-sdk-*` tag namespace deliberately avoids `v*` so product-release workflows never trigger.

**4. Promotion (manual).**
Review the draft and its build-info document, generate the SBOM (not yet automated),
publish with `make_latest: false` so SDK releases never displace Warp product releases,
then run the cold-cache Packman pull against the published assets.
That check cannot run pre-publish because draft assets require authentication;
the full consumption path (redirect chain, checksum rejection of corrupted archives, cache reuse,
`@` surviving in asset names) was validated end-to-end against a temporary fork release.

**5. Consumer migration (distribution plan Phase 2).**
A checked-in Packman manifest pins the release tag, per-platform package versions,
and SHA-256 digests, and `build_llvm.py` / CMake route through it.
The manifest must use the `packman pull` project-file path:
checksum pinning is not enforced by `packman install NAME VERSION`,
which is what `fetch_prebuilt_libraries` calls today.

**6. Corrections.**
Published assets are never overwritten.
Rebuild with an incremented `bundle_revision`, which produces new `warp.N` asset names
(deterministic packaging makes unchanged content auditable by hash), and update the manifest.

The [distribution plan](https://gist.github.com/shi-eric/4eeb78d42309eab47c6c3c79a01ed6a6)
remains the planning record for the phases beyond this repository
(consumer migration, CloudFront retirement).

### Alternatives considered

- **Python/CMake scripts in containers** (extend `build_llvm.py` and the warp-builder pattern).
  Strongest argument: the CI builder and the user-facing fallback become the same code with zero new
  tooling.
  Rejected because it reimplements what the recipe already solves
  (the windows-aarch64 two-pass cross build, size-optimization overrides, library pruning,
  per-platform ABI capture, source pinning, packaging, smoke test),
  and containers only exist for the Linux half of the matrix anyway.
- **Per-platform split** (scripts + containers for Linux, Conan for windows-arm64/macOS).
  Rejected: two build definitions for one artifact family maximizes drift,
  and every cross-platform discrepancy investigation starts by reconciling two systems.
- **Keeping the recipe and `build_llvm.py` in sync** via a shared CMake-variable module.
  Considered and deliberately dropped: `build_llvm.py` stays untouched as an independent fallback,
  drift is accepted, and the recipe is designed as if it did not exist.
  This keeps the recipe free to diverge (newer LLVM, different pruning) without coupling.

## Testing Strategy

Validation is layered, cheapest first:

1. **`test_package`** inside every `conan create`:
   compile, link, and JIT-execute a trivial function against the built libraries;
   link-only when cross-compiling.
2. **Artifact sanity checks** in-job after deployment:
   expected tree shape, machine type and CRT assertions
   (`readelf` symbol-version ceiling on Linux, `lipo -archs` on macOS,
   `dumpbin` machine type and `LIBCMT`-not-`MSVCRT` on Windows).
3. **Consumer smoke job** per platform, consuming the artifact the way a real consumer would:
   extract the archive, `build_lib.py --llvm-path <sdk>`, then JIT-compile and execute a CPU
   kernel (`tools/llvm/ci/smoke_test.py`: deliberately a single fast kernel with no test-suite
   dependencies, so the gate runs identically in bare manylinux containers and attributes failures
   unambiguously to the SDK; suite-level breadth belongs to wheel validation).
   This is the gate that answers whether Warp works with the SDK.
   On linux-aarch64 the check allow-lists `_dl_find_object@GLIBC_2.35`,
   which gcc-toolset-14's static-libgcc unwinder injects and AlmaLinux 9's ld.so provides via a
   version-tagged backport; it is toolchain-injected, not an SDK reference.
4. **Promotion gates** owned by the distribution plan:
   full required matrix present, checksums, cold-cache Packman pull.

Local reproduction uses the identical commands as CI
(`pip install conan==<pinned>`, `conan create` with a checked-in profile),
with a documented `docker run` one-liner for bit-exact manylinux reproduction on Linux
and Conan's local flow (`conan build tools/llvm`) for recipe iteration without cache round-trips.
