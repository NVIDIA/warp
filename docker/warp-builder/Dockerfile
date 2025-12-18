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

# Dockerfile for Warp Builder with LLVM - Multi-architecture
# Supports both x86_64 and aarch64
# This version builds LLVM from source, making it fully self-contained

ARG CUDA_VERSION=13.1.0
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

# Build LLVM from source
FROM base AS llvm-builder
WORKDIR /tmp/llvm-build

ARG TARGETARCH

# Install build tools needed for LLVM (not needed in final image)
RUN yum install -y \
    git \
    unzip \
    cmake \
    && yum clean all

# Install modern Ninja from GitHub releases (yum's ninja-build is too old for LLVM 21.1.0)
# Latest release: https://github.com/ninja-build/ninja/releases
ARG NINJA_VERSION=1.13.2
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
    NINJA_FILE="ninja-linux.zip"; \
    elif [ "$ARCH" = "aarch64" ]; then \
    NINJA_FILE="ninja-linux-aarch64.zip"; \
    else \
    echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget --max-redirect=1 https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/${NINJA_FILE} && \
    unzip ${NINJA_FILE} && \
    mv ninja /usr/local/bin/ && \
    chmod +x /usr/local/bin/ninja && \
    rm ${NINJA_FILE} && \
    ninja --version

# Clone LLVM 21.1.0 (same version as build_llvm.py)
# Using specific commit hash for security: 3623fe661ae35c6c80ac221f14d85be76aa870f1 (llvmorg-21.1.0)
RUN git clone https://github.com/llvm/llvm-project.git llvm-project && \
    cd llvm-project && \
    git checkout 3623fe661ae35c6c80ac221f14d85be76aa870f1 && \
    cd ..

# Determine target backend and host triple based on architecture
RUN if [ "$TARGETARCH" = "x86_64" ]; then \
    echo "X86" > /tmp/target_backend; \
    echo "x86_64-pc-linux" > /tmp/host_triple; \
    else \
    echo "AArch64" > /tmp/target_backend; \
    echo "aarch64-pc-linux" > /tmp/host_triple; \
    fi

# Configure and build LLVM with Clang
# This mirrors the configuration in build_llvm.py for Linux x86_64/aarch64
# Note: Using modern Ninja installed from GitHub releases for faster parallel builds
RUN cmake \
    -S llvm-project/llvm \
    -B build \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=MinSizeRel \
    -D "LLVM_TARGETS_TO_BUILD=$(cat /tmp/target_backend);NVPTX" \
    -D LLVM_ENABLE_PROJECTS=clang \
    -D LLVM_ENABLE_ZLIB=FALSE \
    -D LLVM_ENABLE_ZSTD=FALSE \
    -D LLVM_ENABLE_TERMINFO=FALSE \
    -D LLVM_BUILD_LLVM_C_DYLIB=FALSE \
    -D LLVM_BUILD_RUNTIME=FALSE \
    -D LLVM_BUILD_RUNTIMES=FALSE \
    -D LLVM_BUILD_TOOLS=FALSE \
    -D LLVM_BUILD_UTILS=FALSE \
    -D LLVM_INCLUDE_BENCHMARKS=FALSE \
    -D LLVM_INCLUDE_DOCS=FALSE \
    -D LLVM_INCLUDE_EXAMPLES=FALSE \
    -D LLVM_INCLUDE_RUNTIMES=FALSE \
    -D LLVM_INCLUDE_TESTS=FALSE \
    -D LLVM_INCLUDE_TOOLS=TRUE \
    -D LLVM_INCLUDE_UTILS=FALSE \
    -D "CMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=13" \
    -D CMAKE_INSTALL_PREFIX=/opt/llvm \
    -D "LLVM_HOST_TRIPLE=$(cat /tmp/host_triple)" \
    -D CLANG_BUILD_TOOLS=FALSE \
    -D LLVM_ENABLE_PLUGINS=FALSE \
    -D CLANG_PLUGIN_SUPPORT=FALSE \
    -D CLANG_ENABLE_ARCMT=FALSE \
    -D CLANG_ENABLE_STATIC_ANALYZER=FALSE \
    -D CLANG_TOOLING_BUILD_AST_INTROSPECTION=FALSE \
    -D CLANG_TOOL_AMDGPU_ARCH_BUILD=FALSE \
    -D CLANG_TOOL_APINOTES_TEST_BUILD=FALSE \
    -D CLANG_TOOL_ARCMT_TEST_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_CHECK_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_DIFF_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_EXTDEF_MAPPING_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_FORMAT_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_FORMAT_VS_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_FUZZER_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_IMPORT_TEST_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_LINKER_WRAPPER_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_NVLINK_WRAPPER_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_OFFLOAD_BUNDLER_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_OFFLOAD_PACKAGER_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_OFFLOAD_WRAPPER_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_REFACTOR_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_RENAME_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_REPL_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_SCAN_DEPS_BUILD=FALSE \
    -D CLANG_TOOL_CLANG_SHLIB_BUILD=FALSE \
    -D CLANG_TOOL_C_ARCMT_TEST_BUILD=FALSE \
    -D CLANG_TOOL_C_INDEX_TEST_BUILD=FALSE \
    -D CLANG_TOOL_DIAGTOOL_BUILD=FALSE \
    -D CLANG_TOOL_DRIVER_BUILD=FALSE \
    -D CLANG_TOOL_LIBCLANG_BUILD=FALSE \
    -D CLANG_TOOL_SCAN_BUILD_BUILD=FALSE \
    -D CLANG_TOOL_SCAN_BUILD_PY_BUILD=FALSE \
    -D CLANG_TOOL_SCAN_VIEW_BUILD=FALSE \
    -D LLVM_ENABLE_BINDINGS=FALSE \
    -D LLVM_ENABLE_OCAMLDOC=FALSE \
    -D LLVM_TOOL_BUGPOINT_BUILD=FALSE \
    -D LLVM_TOOL_BUGPOINT_PASSES_BUILD=FALSE \
    -D LLVM_TOOL_CLANG_BUILD=FALSE \
    -D LLVM_TOOL_DSYMUTIL_BUILD=FALSE \
    -D LLVM_TOOL_DXIL_DIS_BUILD=FALSE \
    -D LLVM_TOOL_GOLD_BUILD=FALSE \
    -D LLVM_TOOL_LLC_BUILD=FALSE \
    -D LLVM_TOOL_LLDB_BUILD=FALSE \
    -D LLVM_TOOL_LLI_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_AR_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_AS_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_AS_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_BCANALYZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_CAT_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_CFI_VERIFY_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_CONFIG_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_COV_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_CVTRES_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_CXXDUMP_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_CXXFILT_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_CXXMAP_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_C_TEST_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_DEBUGINFOD_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_DEBUGINFOD_FIND_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_DIFF_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_DIS_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_DIS_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_DLANG_DEMANGLE_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_DWARFDUMP_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_DWARFUTIL_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_DWP_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_EXEGESIS_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_EXTRACT_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_GO_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_GSYMUTIL_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_IFS_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_ISEL_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_ITANIUM_DEMANGLE_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_JITLINK_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_JITLISTENER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_LIBTOOL_DARWIN_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_LINK_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_LIPO_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_LTO2_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_LTO_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_MCA_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_MC_ASSEMBLE_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_MC_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_MC_DISASSEMBLE_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_MICROSOFT_DEMANGLE_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_ML_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_MODEXTRACT_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_MT_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_NM_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_OBJCOPY_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_OBJDUMP_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_OPT_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_OPT_REPORT_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_PDBUTIL_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_PROFDATA_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_PROFGEN_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_RC_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_READOBJ_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_REDUCE_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_REMARK_SIZE_DIFF_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_RTDYLD_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_RUST_DEMANGLE_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_SHLIB_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_SIM_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_SIZE_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_SPECIAL_CASE_LIST_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_SPLIT_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_STRESS_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_STRINGS_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_SYMBOLIZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_TAPI_DIFF_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_TLI_CHECKER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_UNDNAME_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_XRAY_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_YAML_NUMERIC_PARSER_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LLVM_YAML_PARSER_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_LTO_BUILD=FALSE \
    -D LLVM_TOOL_OBJ2YAML_BUILD=FALSE \
    -D LLVM_TOOL_OPT_BUILD=FALSE \
    -D LLVM_TOOL_OPT_VIEWER_BUILD=FALSE \
    -D LLVM_TOOL_REMARKS_SHLIB_BUILD=FALSE \
    -D LLVM_TOOL_SANCOV_BUILD=FALSE \
    -D LLVM_TOOL_SANSTATS_BUILD=FALSE \
    -D LLVM_TOOL_SPLIT_FILE_BUILD=FALSE \
    -D LLVM_TOOL_VERIFY_USELISTORDER_BUILD=FALSE \
    -D LLVM_TOOL_VFABI_DEMANGLE_FUZZER_BUILD=FALSE \
    -D LLVM_TOOL_XCODE_TOOLCHAIN_BUILD=FALSE \
    -D LLVM_TOOL_YAML2OBJ_BUILD=FALSE

# Build LLVM (this takes 30-60 minutes)
# Use all available CPU cores for parallel build
RUN cmake --build build --parallel $(nproc)

# Install LLVM to /opt/llvm
RUN cmake --install build

# Copy LLVM license files
RUN mkdir -p /opt/llvm/licenses && \
    cp /tmp/llvm-build/llvm-project/llvm/LICENSE.TXT /opt/llvm/licenses/ && \
    echo "LLVM is licensed under the Apache License v2.0 with LLVM Exceptions" > /opt/llvm/licenses/README.txt && \
    echo "See: https://llvm.org/LICENSE.txt" >> /opt/llvm/licenses/README.txt

# Get uv from official image
FROM ghcr.io/astral-sh/uv:latest AS uv

# Final stage
FROM base AS final

ARG CUDA_VERSION
ARG TARGETARCH

# Copy CUDA installation from builder stage
# parse_redist.py creates /opt/cuda/linux-${TARGETARCH}/ structure (e.g., linux-x86_64, linux-aarch64)
COPY --from=cuda-installer /opt/cuda/linux-${TARGETARCH} /usr/local/cuda

# Copy LLVM installation from builder stage
COPY --from=llvm-builder /opt/llvm /opt/llvm

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
ENV PATH=/usr/local/cuda/bin:/opt/llvm/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/opt/llvm/lib:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

# Verify CUDA, LLVM, and uv installation
RUN nvcc --version && \
    uv --version && \
    echo "Build environment ready with LLVM!" && \
    echo "LLVM libraries:" && \
    ls -lh /opt/llvm/lib/*.a | head -10

# Set working directory
WORKDIR /workspace

# Metadata labels
LABEL org.opencontainers.image.title="Warp Builder with LLVM"
LABEL org.opencontainers.image.description="Self-contained build environment for NVIDIA Warp with CUDA ${CUDA_VERSION} and LLVM 21.1.0"
LABEL org.opencontainers.image.vendor="NVIDIA"
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/warp"
LABEL cuda.version="${CUDA_VERSION}"
LABEL llvm.version="21.1.0"

CMD ["/bin/bash"]
