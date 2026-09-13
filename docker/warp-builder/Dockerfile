# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Architecture- and CUDA-specific manylinux build environment for Warp. LLVM is
# deliberately supplied by the repository at build time instead of being tied to
# the image lifecycle.

ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.13@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6
ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64@sha256:53390351aeb4688114b02c36a23b3e6ce1166ee9b7afc5df1a4f776354fc764c
ARG PYTHON_VERSIONS="3.10.21 3.11.16 3.12.14 3.13.15 3.14.7 3.14.7t"

FROM ${UV_IMAGE} AS uv
FROM ${MANYLINUX_IMAGE} AS python

ARG PYTHON_VERSIONS

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed

RUN uv --version \
    && uv python install ${PYTHON_VERSIONS} \
    && uv python list --only-installed \
    && uv cache clean

FROM python AS final

ARG CUDA_VERSION=13.4.1
ARG MANYLINUX_POLICY=manylinux_2_28
ARG PYTHON_VERSIONS
ARG TARGETARCH=x86_64
ARG UV_VERSION=0.12.13

COPY --from=cuda_toolkit / /opt/cuda/

ENV CUDA_HOME=/opt/cuda \
    CUDA_PATH=/opt/cuda \
    WARP_CUDA_PATH=/opt/cuda \
    WARP_CUDA_VERSION=${CUDA_VERSION} \
    PATH=/opt/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/cuda/lib64:/opt/cuda/lib

RUN test -x /opt/cuda/bin/nvcc \
    && /opt/cuda/bin/nvcc --version

WORKDIR /workspace

LABEL org.opencontainers.image.title="Warp Builder" \
    org.opencontainers.image.description="manylinux build environment for NVIDIA Warp" \
    org.opencontainers.image.vendor="NVIDIA" \
    org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-NVIDIA-CUDA-EULA" \
    org.opencontainers.image.source="https://github.com/NVIDIA/warp" \
    com.nvidia.warp.manylinux-policy="${MANYLINUX_POLICY}" \
    com.nvidia.warp.cuda-version="${CUDA_VERSION}" \
    com.nvidia.warp.python-versions="${PYTHON_VERSIONS}" \
    com.nvidia.warp.target-architecture="${TARGETARCH}" \
    com.nvidia.warp.uv-version="${UV_VERSION}"

CMD ["/bin/bash"]
