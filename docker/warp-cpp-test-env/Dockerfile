# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile for Warp C++ Testing Environment
# Pre-configured environment for testing Warp C++ examples and integrations
# Uses parse_redist.py to install only required CUDA components

ARG CUDA_VERSION=12.9.1
ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION} AS base

ARG CUDA_VERSION
ARG UBUNTU_VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    gnupg \
    software-properties-common \
    wget \
    curl \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Install CMake from Kitware PPA
RUN wget -qO /etc/apt/trusted.gpg.d/kitware-key.asc https://apt.kitware.com/keys/kitware-archive-latest.asc \
    && echo "deb https://apt.kitware.com/ubuntu/ $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install -y cmake \
    && rm -rf /var/lib/apt/lists/*

# Download and install CUDA components using parse_redist.py
FROM base AS cuda-installer
WORKDIR /tmp/cuda-install

ARG CUDA_VERSION

# Install Python 3 (needed for parse_redist.py)
RUN apt-get update && apt-get install -y --no-install-recommends python3 wget \
    && rm -rf /var/lib/apt/lists/*

# Download parse_redist.py from NVIDIA
# Using specific commit hash for security (instead of branch name)
RUN wget --max-redirect=0 https://raw.githubusercontent.com/NVIDIA/build-system-archive-import-examples/20983d06bcaf0ba85933f8c6e46782cd85f64cf6/parse_redist.py \
    && chmod +x parse_redist.py

# Install CUDA components using redistrib system
# Note: parse_redist.py requires Python 3 and processes one component at a time
#
# Architecture: x86_64 only (ARM64/sbsa support can be added later if needed)
#
# Minimal component list for C++ testing (pre-built Warp):
#   - cuda_cccl: CUDA headers (cuda.h, cuda_runtime.h, thrust, cub)
#   - cuda_cudart: CUDA runtime library (libcudart.so)
#   - cuda_nvcc: CUDA compiler for compiling C++ examples
#
# Components NOT needed for testing (only for building Warp):
#   - cuda_nvrtc: Runtime compilation (used by Warp library internally)
#   - libnvjitlink: JIT linking (used by Warp library internally)
#
# CUDA 13.x differences:
#   - cuda_crt: C runtime (new standalone in CUDA 13, was bundled in 12)
RUN CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) && \
    if [ "$CUDA_MAJOR" = "12" ]; then \
        COMPONENTS="cuda_cccl cuda_cudart cuda_nvcc"; \
    else \
        COMPONENTS="cuda_cccl cuda_crt cuda_cudart cuda_nvcc"; \
    fi && \
    for component in $COMPONENTS; do \
        echo "Installing $component for x86_64..."; \
        python3 parse_redist.py \
            --label ${CUDA_VERSION} \
            --product cuda \
            --os linux \
            --arch x86_64 \
            --output /opt/cuda \
            --component $component || exit 1; \
    done

# Get uv from official distroless image
FROM ghcr.io/astral-sh/uv:latest AS uv

# Final stage
FROM base AS final

ARG CUDA_VERSION
ARG UBUNTU_VERSION

# Copy CUDA installation from builder stage
# parse_redist.py creates /opt/cuda/linux-x86_64/ structure
COPY --from=cuda-installer /opt/cuda/linux-x86_64 /usr/local/cuda

# Copy uv from official image
COPY --from=uv /uv /uvx /bin/

# Create lib64 symlink if it doesn't exist (parse_redist.py uses lib, not lib64)
RUN if [ ! -d /usr/local/cuda/lib64 ] && [ -d /usr/local/cuda/lib ]; then \
        ln -s lib /usr/local/cuda/lib64; \
    fi

# Set environment variables
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda

# Verify installations
RUN nvcc --version && uv --version && cmake --version

WORKDIR /workspace

# Metadata labels
LABEL org.opencontainers.image.title="Warp C++ Test Environment"
LABEL org.opencontainers.image.description="Testing environment for NVIDIA Warp C++ examples with CUDA ${CUDA_VERSION}, CMake, and build tools"
LABEL org.opencontainers.image.vendor="NVIDIA"
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/warp"
LABEL cuda.version="${CUDA_VERSION}"
LABEL ubuntu.version="${UBUNTU_VERSION}"

CMD ["/bin/bash"]
