# Warp LLVM/Clang SDK builds

This directory is the single build definition for the prebuilt LLVM/Clang
SDKs Warp links its CPU-JIT compiler (`warp-clang`) against. The Conan
recipe hardcodes Warp's flavor: clang-only static libraries with the NVPTX
backend, self-contained, size-optimized. Consumers never use Conan. The
deployed tree is the product. Design: `design/llvm-sdk-conan-build.md`.

## Building locally

Requires Python, a host C++ toolchain, and ~50 GB of disk. cmake and ninja
come from your environment (pip works); nothing is fetched from a Conan
registry.

    pip install conan==2.30.0 cmake ninja
    conan profile detect --force
    conan create tools/llvm --version 22.1.8 \
      -pr:h tools/llvm/profiles/<platform> -pr:b default

Profiles: `linux-x86_64`, `linux-aarch64`, `macos-arm64`, `windows-x86_64`,
`windows-arm64`. On Windows and macOS pass the platform profile for `-pr:b`
too. The `test_package` JIT smoke test runs automatically.

Bit-exact reproduction of the Linux CI build (the CI environment is the
manylinux image, pinned by digest in `.github/workflows/build-llvm-sdk.yml`):

    docker run --rm -v "$PWD:/warp" -w /warp \
      -e LLVM_VERSION=22.1.8 -e BUNDLE_REVISION=1 \
      -e PROFILE=linux-x86_64 -e IMAGE_DIGEST=local -e OUTPUT_DIR=/warp/_sdk_assets \
      -e CONAN_VERSION=2.30.0 -e CMAKE_PIN=3.31.6 -e NINJA_PIN=1.11.1.4 \
      quay.io/pypa/manylinux_2_28_x86_64@sha256:a61875a2f84cab7df8de222ff12cabc08ff86eb4ad402ac90ba7bdaed9600cca \
      bash tools/llvm/ci/build-linux.sh

Keep the pin values (CONAN_VERSION, CMAKE_PIN, NINJA_PIN) in sync with the workflow's env block.

### windows-arm64

CI builds natively on `windows-11-arm` runners. To cross-compile from an
x64 machine instead (fallback), pair the profiles; the recipe detects the
cross build and compiles native x64 tablegen tools first:

    conan create tools/llvm --version 22.1.8 \
      -pr:h tools/llvm/profiles/windows-arm64 \
      -pr:b tools/llvm/profiles/windows-x86_64

### Toolset floors

- windows-x86_64 builds require the MSVC v142 toolset (VS2019 floor: MSVC
  link compatibility is directional, so a v143-built SDK could not be
  linked by VS2019 users). CI installs the v142 component on the
  `windows-2022` runner. When building locally on a machine whose only
  Visual Studio is newer than the profile's IDE mapping, add
  `-c:a "tools.microsoft.msbuild:vs_version=<installed major, e.g. 17>"`
  to the `conan create` command.
- windows-arm64 uses v143; its floor is VS2022.

## Extracting the SDK from a build

    conan install --requires clang-warp/22.1.8 \
      -pr:h tools/llvm/profiles/<platform> -pr:b default \
      --deployer tools/llvm/deployers/llvm_sdk.py --deployer-folder out

Then `out/llvm-sdk` is a plain install tree usable directly with
`build_lib.py --llvm-path` or CMake's `WARP_LLVM_PATH`.
`package_sdk.py` turns it into the deterministic release archive;
`check_sdk.py` verifies tree shape and ABI markers.

## CI

`.github/workflows/build-llvm-sdk.yml` (manual `workflow_dispatch`) builds
all five platforms, checks and packages each SDK, and runs a consumer smoke
job that builds `warp-clang` from the artifact and JIT-executes a CPU
kernel. Inputs: `llvm_version` (empty selects the newest version in
`conandata.yml`; explicit versions are validated against `conandata.yml`
before any build starts), `bundle_revision` (bump for corrected
rebuilds; published assets are never overwritten), and `platforms`.

Every fully green `platforms=all` run automatically creates a draft
GitHub release named `llvm-sdk-<version>-warp.<revision>` with the
archives, a `SHA256SUMS` file, and the merged build-info document.
Re-runs replace their own draft; published releases are never touched.
The draft is never marked latest, and the git tag only materializes
when the draft is published. Publishing remains a manual promotion step
per the LLVM SDK distribution plan. Partial-platform dispatches build
and validate without drafting.

## Updating the LLVM version

1. Add the new version's source URL and sha256 to `conandata.yml`
   (verify the digest against the official release independently).
2. Dispatch the workflow with the new `llvm_version`.
3. Follow the release steps in the LLVM SDK distribution plan.
