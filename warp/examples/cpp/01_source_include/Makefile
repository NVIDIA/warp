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

# Makefile for Warp C++ Source Inclusion Example
#
# This builds a C++ program that statically includes Warp-generated CUDA source code.

# Compiler
NVCC := nvcc

# Target executable (named after example directory)
TARGET := 01_source_include

# Source files
SOURCES := main.cu

# Generated Warp source (dependency)
GENERATED_SOURCE := generated/wp___main__.cu

# Warp native directory (relative path from example to repo root)
# Override by setting WARP_NATIVE_DIR environment variable
WARP_NATIVE_DIR ?= ../../../native
WARP_INCLUDE := $(WARP_NATIVE_DIR)

# Target CUDA architectures for broad compatibility
# Generate binaries for current architectures, PTX for future GPUs
GENCODE_FLAGS := -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_80,code=sm_80 \
                 -gencode arch=compute_86,code=sm_86 \
                 -gencode arch=compute_87,code=sm_87 \
                 -gencode arch=compute_89,code=sm_89 \
                 -gencode arch=compute_90,code=sm_90 \
                 -gencode arch=compute_100,code=sm_100 \
                 -gencode arch=compute_120,code=sm_120 \
                 -gencode arch=compute_120,code=compute_120

# Compiler flags
# Use -isystem for Warp headers to suppress library warnings
# Suppress warnings from generated code (550: set but never used, 177: declared but never referenced)
NVCC_FLAGS := $(GENCODE_FLAGS) \
              -std=c++17 \
              -O3 \
              -isystem $(WARP_INCLUDE) \
              -I. \
              -Xcompiler -Wall \
              -diag-suppress 550 \
              -diag-suppress 177

# Default target
.PHONY: all
all: $(TARGET)

# Build the executable (depends on generated source)
$(TARGET): $(GENERATED_SOURCE) $(SOURCES)
	@if [ ! -d "$(WARP_INCLUDE)" ]; then \
		echo "Error: Warp native directory not found at: $(WARP_INCLUDE)"; \
		echo "Make sure you're running from warp/examples/cpp/01_source_include/"; \
		echo "Or set WARP_NATIVE_DIR to point to warp/native/"; \
		exit 1; \
	fi
	@echo "Building C++ program..."
	@echo "  Target architectures: sm_75, sm_80, sm_86, sm_87, sm_89, sm_90, sm_100, sm_120"
	@echo "  PTX fallback: compute_120 (forward compatibility)"
	@echo "  Warp include: $(WARP_INCLUDE)"
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET)
	@echo ""
	@echo "Build successful! Run with: ./01_source_include"

# Generate Warp source code
$(GENERATED_SOURCE): compile_kernel.py
	@echo "Compiling Warp kernel to source..."
	@uv run compile_kernel.py 2>/dev/null || python3 compile_kernel.py
	@echo ""

# Build only the C++ program (assumes kernel already exists)
.PHONY: cpp
cpp: $(SOURCES)
	@if [ ! -f "$(GENERATED_SOURCE)" ]; then \
		echo "Error: Generated kernel not found. Run 'python compile_kernel.py' first."; \
		exit 1; \
	fi
	@echo "Building C++ program (skipping kernel compilation)..."
	@echo "  Target architectures: sm_75, sm_80, sm_86, sm_87, sm_89, sm_90, sm_100, sm_120"
	@echo "  PTX fallback: compute_120 (forward compatibility)"
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET)
	@echo ""
	@echo "Build successful! Run with: ./01_source_include"

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(TARGET)
	@echo "Cleaned build artifacts"

# Clean everything including generated files
.PHONY: distclean
distclean: clean
	rm -rf generated/
	@echo "Cleaned all generated files"

# Help target
.PHONY: help
help:
	@echo "Warp C++ Source Inclusion Example - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)   - Build everything (auto-compiles kernel if needed)"
	@echo "  cpp             - Build only C++ program (for iterating on C++ code)"
	@echo "  clean           - Remove build artifacts"
	@echo "  distclean       - Remove build artifacts and generated/ directory"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Note: To recompile the kernel, run: python compile_kernel.py"
