# Warp C++ Test Environment

Pre-configured Docker image for testing Warp C++ examples in CI/CD pipelines.

**Contents**: Ubuntu 24.04 + CUDA 13.4.2 (selective components via `parse_redist.py`) + CMake +
`build-essential` + uv 0.12.13

**Architecture**: x86_64/amd64 only (ARM64 support can be added later if needed)

## Availability

The [Build Warp C++ Test Environment Image](../../.github/workflows/build-warp-cpp-test-env-image.yml)
workflow builds and verifies the canonical `linux/amd64` image in GitHub Container Registry (GHCR). The
workflow passes CUDA 13.4.2 and Ubuntu 24.04 explicitly, independently of the Dockerfile defaults, and
publishes revision, date, CUDA-major, CUDA-version, and `latest` tags.

After validation, a maintainer promotes the runnable image manifest to the internal GitLab registry. GitLab
CI consumers pin the promoted manifest digest so a later tag update cannot silently change their environment.

The GitHub workflow publishes an OCI index containing the runnable `linux/amd64` image and a provenance
attestation. Promotion must select the runnable platform manifest rather than copying the index or its
`unknown/unknown` attestation. Follow the digest-addressed procedure in
[Promoting Images](../warp-builder/README.md#promoting-images) with `TARGET_ARCH=amd64`.

## Quick Start

```bash
cd docker/warp-cpp-test-env
./build.sh
```

The default command builds `warp-cpp-test-env:cuda13.4.2-ubuntu24.04` locally.

## What's Inside

- **Base**: Ubuntu 24.04
- **CUDA**: Minimal components via `parse_redist.py` (`nvcc`, headers, and runtime only)
- **CMake**: Installed from the Kitware package repository
- **Build tools**: GCC, G++, and Make from `build-essential`
- **uv**: Version 0.12.13 from a digest-pinned official image

The image excludes NVRTC and `libnvjitlink`, which are needed when building Warp itself but not when testing
against pre-built Warp libraries. It also avoids a Packman download in the C++ examples job.

## Building

### Default (CUDA 13.4.2, Ubuntu 24.04)

```bash
./build.sh
```

### Custom versions

```bash
./build.sh --cuda 12.9.2 --ubuntu 22.04
```

### Build and push to a registry

```bash
./build.sh --registry registry.example.com/project --push
```

See `./build.sh --help` for all options.

### Direct Docker build

```bash
docker build \
  --build-arg CUDA_VERSION=13.4.2 \
  --build-arg UBUNTU_VERSION=24.04 \
  --tag warp-cpp-test-env:cuda13.4.2-ubuntu24.04 \
  .
```

## Using in CI

Pin the promoted runnable manifest, retaining the revision tag for readability:

```yaml
linux-x86_64 cpp examples test:
  stage: test
  image: ${CI_REGISTRY_IMAGE}/warp-cpp-test-env:cuda13.4.2-ubuntu24.04-<source-revision>@sha256:<promoted-manifest-digest>
  needs: [linux-x86_64 build]
  before_script:
    - mv warp/bin/linux-x86_64/*.so warp/bin/
  script:
    - cd warp/examples/cpp
    - bash test_examples.sh
```

`${CI_REGISTRY_IMAGE}` expands to the current project's registry path. The digest, rather than the tag,
determines the image pulled by the runner.

## Customization

Available build arguments:

- `CUDA_VERSION` (default: `13.4.2`) - CUDA version from NVIDIA redistrib
- `UBUNTU_VERSION` (default: `24.04`) - Ubuntu base version

The image is built for x86_64 only. See `warp-builder` for architecture-specific x86_64 and ARM64 images.
Available CUDA versions are listed in the
[CUDA redistrib manifest](https://developer.download.nvidia.com/compute/cuda/redist/).

## Comparison with `warp-builder`

| | `warp-cpp-test-env` | `warp-builder` |
|-|---------------------|----------------|
| **Purpose** | Test C++ examples | Build Python wheels |
| **Architectures** | x86_64 | x86_64 and ARM64 |
| **LLVM** | Not included | Not included; selected by the repository |
| **CUDA** | Selective redistrib download | Repository-locked Toolkit build context |
| **Python** | System Python plus uv | Pinned uv-managed CPython installations |

## License

This Dockerfile and build script are licensed under Apache 2.0 (see [LICENSE.md](../../LICENSE.md)).

Building and redistributing the image must comply with component licenses, including the
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/).
