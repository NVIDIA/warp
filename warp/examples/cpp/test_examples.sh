#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test script for Warp C++ examples
# This script builds and runs all C++ examples using CMake and CTest

set -e  # Exit on error

SEPARATOR="========================================"

# Parse command-line arguments
CLEANUP=false
for arg in "$@"; do
    case "$arg" in
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  --cleanup    Remove build/ and generated/ directories after tests pass"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

echo "$SEPARATOR"
echo "Warp C++ Examples Test Runner"
echo "$SEPARATOR"
echo ""

# Check for required commands
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: Required command 'nvcc' not found in PATH" >&2
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "ERROR: Required command 'cmake' not found in PATH" >&2
    exit 1
fi

# Check for Python - either uv or python3 is required
if command -v uv &> /dev/null; then
    PYTHON_VERSION="uv ($(uv run python --version))"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION="$(python3 --version)"
else
    echo "ERROR: Neither 'uv' nor 'python3' found in PATH" >&2
    exit 1
fi

echo "✓ Found required dependencies:"
echo "  - nvcc: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
echo "  - Python: ${PYTHON_VERSION}"
echo "  - cmake: $(cmake --version | head -1)"
echo ""

# Set up environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WARP_NATIVE_DIR="${SCRIPT_DIR}/../../native"

echo "Configuration:"
echo "  - Example directory: ${SCRIPT_DIR}"
echo "  - Warp native directory: ${WARP_NATIVE_DIR}"
echo ""

# Compile Warp kernels for all examples (must happen before CMake configuration)
echo "$SEPARATOR"
echo "Compiling Warp kernels..."
echo "$SEPARATOR"
for example_dir in "${SCRIPT_DIR}"/*/; do
    # Skip if not a directory or if it's the build directory
    if [[ ! -d "${example_dir}" ]] || [[ "$(basename "${example_dir}")" = "build" ]]; then
        continue
    fi

    example_name=$(basename "${example_dir}")

    # APIC examples ship capture_wave.py instead of compile_kernel.py;
    # both are pre-build Python scripts that produce inputs the C++ build needs.
    if [[ -f "${example_dir}/compile_kernel.py" ]]; then
        prebuild_script="compile_kernel.py"
    elif [[ -f "${example_dir}/capture_wave.py" ]]; then
        prebuild_script="capture_wave.py"
    else
        echo "- Skipping ${example_name} (no compile_kernel.py or capture_wave.py)"
        continue
    fi

    echo "- Running ${prebuild_script} for ${example_name}..."
    cd "${example_dir}"
    if command -v uv &> /dev/null; then
        uv run "${prebuild_script}" || { echo "ERROR: ${prebuild_script} failed for ${example_name}" >&2; exit 1; }
    else
        python3 "${prebuild_script}" || { echo "ERROR: ${prebuild_script} failed for ${example_name}" >&2; exit 1; }
    fi
    cd "${SCRIPT_DIR}"
done
echo ""

# Create build directory
BUILD_DIR="${SCRIPT_DIR}/build"
echo "Creating build directory: ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
echo ""

# Configure with CMake
echo "$SEPARATOR"
echo "Running CMake configuration..."
echo "$SEPARATOR"
cd "${BUILD_DIR}"
cmake -DWARP_NATIVE_DIR="${WARP_NATIVE_DIR}" ..
echo ""

# Build all targets
echo "$SEPARATOR"
echo "Building all example targets..."
echo "$SEPARATOR"
cmake --build .
echo ""

# Run tests with CTest
echo "$SEPARATOR"
echo "Running tests with CTest..."
echo "$SEPARATOR"
ctest --output-on-failure --verbose
TEST_RESULT=$?
echo ""

if [[ $TEST_RESULT -eq 0 ]]; then
    echo "$SEPARATOR"
    echo "✓ All C++ example tests passed!"
    echo "$SEPARATOR"
    
    # Clean up if requested
    if [[ "$CLEANUP" = true ]]; then
        echo ""
        echo "Cleaning up generated files..."
        
        # Remove build directory
        if [[ -d "${BUILD_DIR}" ]]; then
            rm -rf "${BUILD_DIR}"
            echo "  ✓ Removed ${BUILD_DIR}"
        fi
        
        # Remove generated directories from each example
        for example_dir in "${SCRIPT_DIR}"/*/; do
            if [[ ! -d "${example_dir}" ]]; then
                continue
            fi
            
            generated_dir="${example_dir}generated"
            if [[ -d "${generated_dir}" ]]; then
                rm -rf "${generated_dir}"
                echo "  ✓ Removed $(basename "${example_dir}")/generated/"
            fi
        done
        
        echo "✓ Cleanup complete"
    fi
else
    echo "$SEPARATOR"
    echo "✗ Some C++ example tests failed"
    echo "$SEPARATOR"
fi

exit $TEST_RESULT
