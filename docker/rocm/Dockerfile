# ROCm 7.2 / gfx942 image for Warp

# Both targets copy the Warp repo into /workspace/warp, build the native
# library and install it in editable mode, so the build context must be the
# repo root. Build locally (from the repo root) with:
#   docker build --network=host -f docker/rocm/Dockerfile --target warp_gfx942 -t <tag> .

ARG ROCM_VERSION=7.2
ARG UBUNTU_VERSION=24.04
ARG BASE_IMAGE=rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0

FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-complete AS warp_gfx942

ARG ROCM_VERSION=7.2
ARG TORCH_VERSION=2.9.1
ARG TORCH_VISION_VERSION=0.24.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libstdc++-14-dev \
    wget \
    git-lfs \
    libegl1 \
    libegl-mesa0 \
    libgles2 \
    libglvnd0 \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

WORKDIR /root/

# python3 (3.12), pip and the Python headers come with the base image.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

RUN pip install --break-system-packages \
    cmake pybind11 build ninja scikit-build-core setuptools-scm numpy pytest pytest-xdist matplotlib "coverage[toml]" && \
    pip install --break-system-packages \
    torch==${TORCH_VERSION} torchvision==${TORCH_VISION_VERSION} -f https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/

ARG HIP_ARCH=gfx942

COPY . /workspace/warp
WORKDIR /workspace/warp

RUN ROCM_PATH=/opt/rocm python build_lib.py --hip-arch=${HIP_ARCH} --hipcc-options="-Xarch_device -fno-inline" && \
    pip install -e . --break-system-packages



# ROCm 10.1.0 / gfx950 image for the Mujoco Warp workload.
#
# Build locally with:
#   docker build --network=host -f docker/rocm/Dockerfile --target warp_gfx950 -t <tag> .
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS warp_gfx950

WORKDIR /root/
ENV MAX_JOBS=128
ENV PYTORCH_ROCM_ARCH="gfx950"
ENV ROCM_AMDGPU_TARGETS="gfx950"

# Install Ubuntu dependencies. python3/pip come with the base image.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y \
    git \
    git-lfs \
    libtool \
    libegl1-mesa-dev \
    g++-14 \
    wget \
    sudo \
    curl \
    libstdc++-14-dev \
    libdw1 \
    libdrm-dev \
    ccache \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100

# Python build tooling for the warp wheel, plus coverage for the unit-test run.
# None of these depend on torch, so the base image's ROCm torch stays in place.
RUN pip install --break-system-packages \
    cmake pybind11 build ninja scikit-build-core setuptools-scm numpy pytest pytest-xdist Pillow matplotlib "coverage[toml]"

# The base image's ROCm lives in the rocm-sdk-devel wheel, not /opt/rocm.
# Expose it at the conventional path so the warp build (and the env vars below)
# find it.
RUN ln -s "$(rocm-sdk path --root)" /opt/rocm \
    && test -e /opt/rocm/lib/libamdhip64.so

# Setup ROCm stack related envs.
# Note: /opt/rocm/lib is intentionally NOT in PATH (it is a library dir, not a bin dir).
ENV PATH="/opt/rocm/bin:$PATH"
ENV LD_LIBRARY_PATH=/opt/rocm/lib
ENV ROCM_PATH=/opt/rocm
ENV HIP_PLATFORM=amd
ENV HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
# Workaround for the HAS_STATUS_ERROR_OUT_OF_RESOURCES issue (runtime)
ENV HAS_ENABLE_SCRATCH_ASYNC_RECLAIM=0
ENV HAS_NO_SCRATCH_RECLAIM=1

# Create a symlink to python3 in /usr/local/bin/python
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

ARG HIP_ARCH=gfx950

COPY . /workspace/warp
WORKDIR /workspace/warp

RUN ROCM_PATH=/opt/rocm python build_lib.py --hip-arch=${HIP_ARCH} --hipcc-options="-Xarch_device -fno-inline" && \
    pip install -e . --break-system-packages
