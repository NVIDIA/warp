# Warp C++ Test Environment

Pre-configured Docker image for testing Warp C++ examples in CI/CD pipelines.

**Contents**: Ubuntu 24.04 + CUDA 12.9.1 (selective components via parse_redist.py) + CMake (latest) + build-essential + uv

**Architecture**: x86_64/amd64 only (ARM64 support can be added later if needed)

## Quick Start

```bash
# Build the image
cd docker/warp-cpp-test-env
./build.sh

# Push to registry (replace with your registry URL)
docker tag warp-cpp-test-env:12.9.1-ubuntu24.04 your-registry.com/project/warp-cpp-test-env:12.9.1-ubuntu24.04
docker push your-registry.com/project/warp-cpp-test-env:12.9.1-ubuntu24.04
```

## What's Inside

- **Base**: Ubuntu 24.04
- **CUDA**: Minimal components via parse_redist.py (nvcc, headers, runtime only)
- **CMake**: Latest from Kitware PPA
- **Build tools**: gcc, g++, make (build-essential)
- **uv**: Python package manager

**Benefits**: Saves 5-10 min per CI run. Smaller image (~2.8GB) vs full CUDA devel (~10GB). No packman needed.

**Components excluded**: nvrtc, libnvjitlink (only needed for building Warp, not testing)

## Building

### Default (CUDA 12.9.1, Ubuntu 24.04)
```bash
./build.sh
```

### Custom versions
```bash
./build.sh --cuda 13.0.0 --ubuntu 22.04
```

### Build and push to registry
```bash
./build.sh --registry registry.example.com --push
```

See `./build.sh --help` for all options.

## Using in CI

### GitLab CI Example

```yaml
linux-x86_64 cpp examples test:
  stage: test
  image: ${CI_REGISTRY_IMAGE}/warp-cpp-test-env:12.9.1-ubuntu24.04
  needs: [linux-x86_64 build]
  before_script:
    - mv warp/bin/linux-x86_64/*.so warp/bin/
  script:
    - cd warp/examples/cpp
    - bash test_examples.sh
```

**Note**: `${CI_REGISTRY_IMAGE}` expands to your project's registry path automatically.

### Direct Docker build

```bash
docker build \
  --build-arg CUDA_VERSION=12.9.1 \
  --build-arg UBUNTU_VERSION=24.04 \
  -t warp-cpp-test-env:12.9.1-ubuntu24.04 .
```

## Customization

Available build arguments:

- `CUDA_VERSION` (default: `12.9.1`) - CUDA version from NVIDIA redistrib
- `UBUNTU_VERSION` (default: `24.04`) - Ubuntu base version

**Note**: This image is built for x86_64 only. For ARM64/aarch64 support, see `warp-builder` for multi-arch example.

Available CUDA versions: Check NVIDIA's redistrib JSON at https://developer.download.nvidia.com/compute/cuda/redist/

## Comparison with `warp-builder`

| | `warp-cpp-test-env` | `warp-builder` |
|-|---------------------|----------------|
| **Purpose** | C++ testing | Wheel building |
| **LLVM** | No | Yes |
| **CUDA** | parse_redist.py (selective) | parse_redist.py (selective) |
| **Build time** | ~2.5 min | ~60 min |
| **Size** | ~2.8 GB | ~8 GB |

## License

This Dockerfile and build scripts are licensed under Apache 2.0 (see [LICENSE.md](../../LICENSE.md)).

When building the image, ensure compliance with component licenses (notably the [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/)).
