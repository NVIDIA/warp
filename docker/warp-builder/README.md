# Warp Builder Images

Architecture- and CUDA-specific manylinux images used to build NVIDIA Warp in CI. Each image provides a
pinned uv release, the uv-managed Python versions used by the Warp build jobs, and one repository-locked CUDA
Toolkit. LLVM remains repository-selected at build time.

## Availability

The published images are internal NVIDIA CI artifacts. GitHub Actions builds and tests the canonical images in
GitHub Container Registry (GHCR). CUDA is downloaded only when these deliberately published images are built;
jobs that consume a promoted image use its embedded Toolkit instead of downloading CUDA from an external
package host.

After all four variants are validated, a maintainer deliberately promotes them into the GitLab registry. A
separate follow-up change must map each Linux architecture and CUDA release to its promoted digest, replace
the Linux build jobs' `/usr/local/cuda` arguments with `/opt/cuda` (or rely on `WARP_CUDA_PATH`), and remove the
redundant free-threaded Python install. GitLab CI continues to use the legacy images until that promotion and
follow-up change are complete.

The GHCR package is not publicly accessible. Anyone without NVIDIA organization access can build the
Dockerfile from source.

## Image Contract

The image contains:

- An architecture-specific, digest-pinned manylinux base.
- One CUDA Toolkit assembled from `tools/ci/cuda_toolkit_lock.json` and installed in `/opt/cuda`.
- uv 0.12.13, copied from a digest-pinned official uv image.
- uv-managed CPython 3.10.21, 3.11.16, 3.12.14, 3.13.15, 3.14.7, and 3.14.7t.
- The compiler and utilities already supplied by the manylinux base.

The image sets:

```text
UV_PYTHON_INSTALL_DIR=/opt/uv/python
UV_PYTHON_PREFERENCE=only-managed
CUDA_HOME=/opt/cuda
CUDA_PATH=/opt/cuda
WARP_CUDA_PATH=/opt/cuda
WARP_CUDA_VERSION=<CUDA Toolkit release>
```

Automatic Python downloads remain enabled. A CI experiment can therefore request a newer unbundled Python,
while the normal build matrix uses the preinstalled versions.

The image does not add LLVM, CMake, or Ninja beyond anything supplied by its manylinux base. A tool present in
the selected base is not part of this image's supported contract unless listed above. Redistribution of the
embedded Toolkit is subject to the NVIDIA CUDA Toolkit EULA.

## Tags

Tags describe the compatibility boundary, CUDA and uv releases, and architecture:

```text
ghcr.io/nvidia/warp-builder:manylinux_2_28-cuda12.9.2-uv0.12.13-x86_64
ghcr.io/nvidia/warp-builder:manylinux_2_28-cuda13.4.1-uv0.12.13-x86_64
ghcr.io/nvidia/warp-builder:manylinux_2_34-cuda12.9.2-uv0.12.13-aarch64
ghcr.io/nvidia/warp-builder:manylinux_2_34-cuda13.4.1-uv0.12.13-aarch64
```

The tags can move when an image is rebuilt with the same contract. Consumers must pin the image digest for
immutable identity. OCI labels and the GitHub Actions workflow history record the precise build inputs and
source revision.

The workflow does not publish multi-architecture manifests. Each tag identifies one native architecture and
one CUDA release.

## Building and Testing

The [Build Warp Builder Images](../../.github/workflows/build-warp-builder-images.yml) workflow builds x86_64
and aarch64 images independently on native runners. For each CUDA release it assembles the repository-locked
Toolkit, passes that directory to BuildKit as the `cuda_toolkit` build context, verifies CUDA and every bundled
Python without network access, and then builds Warp with the embedded Toolkit. The image is published only
after these checks pass.

To build the x86_64 image locally:

```bash
docker build \
  --platform linux/amd64 \
  --build-context cuda_toolkit="$WARP_CUDA_PATH" \
  --build-arg MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64@sha256:53390351aeb4688114b02c36a23b3e6ce1166ee9b7afc5df1a4f776354fc764c \
  --build-arg MANYLINUX_POLICY=manylinux_2_28 \
  --build-arg CUDA_VERSION=13.4.1 \
  --build-arg TARGETARCH=x86_64 \
  --tag warp-builder:manylinux_2_28-cuda13.4.1-uv0.12.13-x86_64 \
  docker/warp-builder
```

`WARP_CUDA_PATH` must identify a Toolkit matching `CUDA_VERSION`. The GitHub workflow obtains this directory
from the repository's CUDA setup action rather than relying on a developer installation.

Verify the embedded Toolkit and preinstalled Python runtimes without network access:

```bash
docker run --rm --network=none \
  -e EXPECTED_CUDA_PLATFORM=linux-x86_64 \
  -e EXPECTED_CUDA_VERSION=13.4.1 \
  -v "$(pwd):/workspace:ro" \
  warp-builder:manylinux_2_28-cuda13.4.1-uv0.12.13-x86_64 \
  bash /workspace/docker/warp-builder/verify-image.sh
```

To build Warp with the embedded Toolkit:

```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  -e WARP_CACHE_PATH=/workspace/.cache/warp-builder \
  warp-builder:manylinux_2_28-cuda13.4.1-uv0.12.13-x86_64 \
  uv run --no-python-downloads --python 3.12 build_lib.py --cuda-path=/opt/cuda
```

`build_lib.py` fetches the LLVM SDK and libmathdx package selected by their manifests in `deps/` unless an
explicit dependency path is provided or libmathdx is disabled.

## Updating the Images

Rebuild and promote the affected images when the CUDA release, manylinux policy, supported Python versions,
or uv release changes. Routine LLVM updates do not require a new image.
