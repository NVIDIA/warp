# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dockerfile for Warp Builder - Multi-architecture
# Supports both x86_64 and aarch64
#
# Provides the toolchain (CUDA, CMake, Ninja, uv). Clang/LLVM is deliberately NOT
# baked in: build_lib.py fetches the SDK through Packman using the digests pinned in
# deps/llvm-deps.packman.xml, so the LLVM version is chosen by the repository being
# built rather than by whichever image happens to be in use.

ARG CUDA_VERSION=13.0.2
ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64:latest
FROM ${MANYLINUX_IMAGE} AS base

# Set build arguments
ARG CUDA_VERSION
ARG TARGETARCH=x86_64

# Install minimal dependencies for downloading and extracting
RUN yum install -y \
    wget \
    ca-certificates \
    && yum clean all

# Download and install CUDA components using parse_redist.py
FROM base AS cuda-installer
WORKDIR /tmp/cuda-install

ARG CUDA_VERSION
ARG TARGETARCH

# Download parse_redist.py from NVIDIA
# Using specific commit hash for security (instead of branch name)
RUN wget --max-redirect=0 https://raw.githubusercontent.com/NVIDIA/build-system-archive-import-examples/20983d06bcaf0ba85933f8c6e46782cd85f64cf6/parse_redist.py \
    && chmod +x parse_redist.py

# Install CUDA components using redistrib system
# Note: parse_redist.py requires Python 3 and processes one component at a time
# Component list differs between CUDA 12.x and 13.x:
#   CUDA 12.x: cuda_cccl cuda_cudart cuda_nvcc cuda_nvrtc libnvjitlink
#              (libnvptxcompiler_static.a and libnvvm.so are bundled with nvrtc)
#   CUDA 13.x: cuda_cccl cuda_crt cuda_cudart cuda_nvcc cuda_nvrtc libnvvm libnvptxcompiler libnvjitlink
#              (cuda_crt, libnvvm, libnvptxcompiler are new standalone components)
RUN CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) && \
    if [ "$CUDA_MAJOR" = "12" ]; then \
    COMPONENTS="cuda_cccl cuda_cudart cuda_nvcc cuda_nvrtc libnvjitlink"; \
    else \
    COMPONENTS="cuda_cccl cuda_crt cuda_cudart cuda_nvcc cuda_nvrtc libnvvm libnvptxcompiler libnvjitlink"; \
    fi && \
    for component in $COMPONENTS; do \
    echo "Installing $component..."; \
    python3 parse_redist.py \
    --label ${CUDA_VERSION} \
    --product cuda \
    --os linux \
    --arch ${TARGETARCH} \
    --output /opt/cuda \
    --component $component || exit 1; \
    done

# Install pinned CMake from official Kitware binary tarballs.
FROM base AS cmake-installer

ARG CMAKE_VERSION=4.3.3
ARG TARGETARCH

RUN if [ "$TARGETARCH" = "sbsa" ]; then \
    CMAKE_ARCH="aarch64"; \
    elif [ "$TARGETARCH" = "x86_64" ]; then \
    CMAKE_ARCH="x86_64"; \
    elif [ "$TARGETARCH" = "aarch64" ]; then \
    CMAKE_ARCH="aarch64"; \
    else \
    echo "Unsupported architecture: $TARGETARCH" && exit 1; \
    fi && \
    if [ "$CMAKE_ARCH" = "x86_64" ]; then \
    CMAKE_SHA256="927b2368a946c37269c3a66225ab00544e756459cdd0b5d0da438694fb9ff802"; \
    elif [ "$CMAKE_ARCH" = "aarch64" ]; then \
    CMAKE_SHA256="9ea38356dbd3e32e51029a3e09a0f2f8e117ef4fbcaad7a21ffb36409bbd5cb4"; \
    fi && \
    CMAKE_FILE="cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz" && \
    wget -nv --https-only --max-redirect=1 "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_FILE}" && \
    echo "${CMAKE_SHA256}  ${CMAKE_FILE}" | sha256sum -c - && \
    mkdir -p /opt/cmake && \
    tar -xzf "${CMAKE_FILE}" --strip-components=1 -C /opt/cmake && \
    rm "${CMAKE_FILE}" && \
    /opt/cmake/bin/cmake --version

# Modern Ninja for the final image (yum's ninja-build is too old for current CMake projects).
# Latest release: https://github.com/ninja-build/ninja/releases
FROM base AS ninja-installer

RUN yum install -y curl unzip \
    && yum clean all

ARG NINJA_VERSION=1.13.2
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
    NINJA_FILE="ninja-linux.zip"; \
    NINJA_SHA256="5749cbc4e668273514150a80e387a957f933c6ed3f5f11e03fb30955e2bbead6"; \
    elif [ "$ARCH" = "aarch64" ]; then \
    NINJA_FILE="ninja-linux-aarch64.zip"; \
    NINJA_SHA256="fd2cacc8050a7f12a16a2e48f9e06fca5c14fc4c2bee2babb67b58be17a607fc"; \
    else \
    echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    curl -fL --proto '=https' --proto-redir '=https' --max-redirs 1 \
    -o "${NINJA_FILE}" \
    "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/${NINJA_FILE}" && \
    echo "${NINJA_SHA256}  ${NINJA_FILE}" | sha256sum -c - && \
    unzip "${NINJA_FILE}" && \
    mv ninja /usr/local/bin/ && \
    chmod +x /usr/local/bin/ninja && \
    rm "${NINJA_FILE}" && \
    ninja --version

# Get uv from official image
FROM ghcr.io/astral-sh/uv:latest AS uv

# Final stage
FROM base AS final

ARG CUDA_VERSION
ARG TARGETARCH

# Copy CUDA installation from builder stage
# parse_redist.py creates /opt/cuda/linux-${TARGETARCH}/ structure (e.g., linux-x86_64, linux-aarch64)
COPY --from=cuda-installer /opt/cuda/linux-${TARGETARCH} /usr/local/cuda

# Copy the pinned CMake binary distribution from Kitware.
COPY --from=cmake-installer /opt/cmake /opt/cmake

# Copy the pinned modern Ninja.
COPY --from=ninja-installer /usr/local/bin/ninja /usr/local/bin/ninja

# Copy uv from official image
COPY --from=uv /uv /uvx /bin/

# Create lib64 symlink if it doesn't exist (parse_redist.py uses lib, not lib64)
RUN if [ ! -d /usr/local/cuda/lib64 ] && [ -d /usr/local/cuda/lib ]; then \
    ln -s lib /usr/local/cuda/lib64; \
    fi

# Collect CUDA license files for compliance
RUN mkdir -p /usr/local/cuda/licenses && \
    find /usr/local/cuda -type f \( -name "LICENSE.txt" -o -name "EULA.txt" -o -name "LICENSE" \) \
    -exec cp {} /usr/local/cuda/licenses/ \; 2>/dev/null || true && \
    echo "CUDA Toolkit components are subject to the NVIDIA CUDA EULA" > /usr/local/cuda/licenses/README.txt && \
    echo "See: https://docs.nvidia.com/cuda/eula/" >> /usr/local/cuda/licenses/README.txt

# Set environment variables
ENV PATH=/opt/cmake/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

# Verify CUDA, CMake, Ninja, and uv installation
RUN nvcc --version && \
    cmake --version && \
    ninja --version && \
    uv --version && \
    echo "Build environment ready"

# Set working directory
WORKDIR /workspace

# Metadata labels
LABEL org.opencontainers.image.title="Warp Builder"
LABEL org.opencontainers.image.description="Build environment for NVIDIA Warp with CUDA ${CUDA_VERSION}"
LABEL org.opencontainers.image.vendor="NVIDIA"
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/warp"
LABEL cuda.version="${CUDA_VERSION}"

CMD ["/bin/bash"]
