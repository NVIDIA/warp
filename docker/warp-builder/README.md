# Warp Builder Docker Images

Multi-architecture Docker images for building NVIDIA Warp with CUDA and LLVM compiled from source.

> **Note:** These images are published to `ghcr.io/nvidia/warp-builder`. Depending on organization
> settings, they may require authentication even if marked as public. See [Authentication](#authentication) below.

## Quick Start

### Use in GitHub Actions

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/nvidia/warp-builder:cuda13  # Or use :latest for newest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Warp
        run: uv run build_lib.py --llvm_path /opt/llvm
      
      - name: Build wheel
        run: uv build --wheel
```

> **Note:** GitHub Actions automatically authenticates with GHCR when using containers in workflows.

### Use Locally to Build Warp

**Prerequisites:** Docker installed, Warp repository cloned locally

```bash
# 1. Navigate to your Warp checkout
cd /path/to/your/warp

# 2. Pull the image (Docker auto-selects x86_64 or aarch64)
docker pull ghcr.io/nvidia/warp-builder:cuda13

# 3. Build Warp native libraries
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/nvidia/warp-builder:cuda13 \
  bash -c "uv run build_lib.py --llvm_path /opt/llvm"

# 4. Build Python wheel (optional)
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/nvidia/warp-builder:cuda13 \
  bash -c "uv build --wheel"

# Result: Compiled binaries in warp/bin/, wheel in dist/
```

**Interactive development with GPU access:**

```bash
# Prerequisites: Install NVIDIA Container Toolkit on your host
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Start a shell with GPU access
docker run --rm -it \
  --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/nvidia/warp-builder:cuda13 \
  bash

# Inside container:
# uv run build_lib.py --llvm_path /opt/llvm       # Build
# uv run --extra dev -m warp.tests -s autodetect  # Test (GPU tests will be skipped without --gpus)
# uv build --wheel                                 # Package
```

> **Note:** The `--gpus all` flag requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host. You can run tests without a GPU, but GPU-specific tests will be skipped.

## Available Tags

**Short aliases (recommended for most users):**
- `latest` - Latest build with newest CUDA version
- `cuda13` - Latest CUDA 13.x build (currently 13.0.2 with LLVM 21)
- `cuda12` - Latest CUDA 12.x build (currently 12.9.1 with LLVM 21)

**Full version tags (for reproducibility):**
- `cuda13.0.2-llvm21-latest` - Multi-arch, always current
- `cuda13.0.2-llvm21-20241129` - Multi-arch, date-pinned
- `cuda13.0.2-llvm21-x86_64-latest` - Architecture-specific
- `cuda13.0.2-llvm21-aarch64-latest` - Architecture-specific

**Examples:**
```bash
# Short and memorable
docker pull ghcr.io/nvidia/warp-builder:cuda13

# Pinned to specific CUDA version
docker pull ghcr.io/nvidia/warp-builder:cuda13.0.2-llvm21-latest

# Pinned to exact build date
docker pull ghcr.io/nvidia/warp-builder:cuda13.0.2-llvm21-20241129
```

All multi-arch tags work on both x86_64 and aarch64.

## Authentication

If the images are not publicly accessible or you encounter authentication errors:

### For Personal Use

```bash
# Authenticate with GitHub (requires personal access token with read:packages scope)
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Then pull normally
docker pull ghcr.io/nvidia/warp-builder:cuda13
```

### For CI/CD
GitHub Actions automatically authenticates when using `container:` in workflows. For other CI systems, use a GitHub Personal Access Token (PAT) or GitHub App token with `read:packages` permission.

### Alternative: Build Locally
If you cannot access published images, you can build them locally:
```bash
cd docker/warp-builder
docker buildx build --platform linux/amd64 -t warp-builder:cuda13 -f Dockerfile \
  --build-arg CUDA_VERSION=13.0.2 --load .
```

## Image Contents

- **Base:** manylinux_2_28 (x86_64) / manylinux_2_34 (aarch64)
- **CUDA:** Configurable (supports 12.x and 13.x, default 13.0.2)
- **LLVM:** Compiled from source at `/opt/llvm` (default 21.1.0)
- **Python:** Managed by uv
- **Tools:** GCC toolchain (from manylinux base)

> **Note:** The Dockerfile automatically handles component differences between CUDA 12.x and 13.x.

## Rebuilding Images

Images are automatically built by the workflow at `.github/workflows/build-warp-builder-images.yml`.

**Each workflow run builds:**

- CUDA 12.9.1 (x86_64 + aarch64)
- CUDA 13.0.2 (x86_64 + aarch64)
- All 4 builds run in parallel (~60 minutes total)

**To trigger a rebuild:**

1. Go to Actions → Build Warp Builder Images
2. Click "Run workflow"
3. Both CUDA versions will be built and published

To change CUDA versions, edit the matrix in the workflow file.

## Architecture Support

Both x86_64 and aarch64 are fully supported with native compilation (no emulation).

**ARM64 variants:**

- **CUDA 12:** Uses Tegra/Jetson packages (`linux-aarch64`) - optimized for Jetson devices but works on most ARM64 systems
- **CUDA 13:** Uses unified server packages (`linux-sbsa`) - works on all ARM64 systems including Jetson

## Licenses

These images contain software components under different licenses:

- **CUDA Toolkit:** Subject to the [NVIDIA CUDA End User License Agreement (EULA)](https://docs.nvidia.com/cuda/eula/)
  - License files are included in the image at `/usr/local/cuda/licenses/`
  - By using these images, you agree to the CUDA EULA terms
  
- **LLVM/Clang:** Licensed under [Apache License v2.0 with LLVM Exceptions](https://llvm.org/LICENSE.txt)
  - License files are included in the image at `/opt/llvm/licenses/`
  - Source code available at https://github.com/llvm/llvm-project
  
- **Container Image:** Components built and distributed by NVIDIA for use with NVIDIA Warp

To view licenses in a running container:

```bash
docker run --rm ghcr.io/nvidia/warp-builder:cuda13 cat /usr/local/cuda/licenses/README.txt
docker run --rm ghcr.io/nvidia/warp-builder:cuda13 cat /opt/llvm/licenses/README.txt
```

## Notes

- Always pass `--llvm_path /opt/llvm` to `build_lib.py` to use the built-in LLVM
- Images are self-contained with no external dependencies (no Packman required)
- LLVM is compiled with targets for X86/AArch64 + NVPTX
