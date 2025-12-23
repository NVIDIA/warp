#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Default values
CUDA_VERSION="12.9.1"
UBUNTU_VERSION="24.04"
REGISTRY=""
PUSH=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to display usage
usage() {
    local exit_code=${1:-0}
    cat << EOF
Usage: $0 [OPTIONS]

Build and optionally push Warp C++ Test Environment Docker image.

OPTIONS:
    -c, --cuda VERSION       CUDA version (default: 12.9.1)
    -u, --ubuntu VERSION     Ubuntu version (default: 24.04)
    -r, --registry URL       Docker registry URL (e.g., registry.example.com)
    -p, --push               Push image to registry after building
    -h, --help               Show this help message

NOTE:
    This image is built for x86_64/amd64 only.
    ARM64 support can be added later if needed.

EXAMPLES:
    # Build locally (default)
    $0

    # Build with different CUDA version
    $0 --cuda 13.0.0

    # Build and push to registry
    $0 --registry registry.example.com --push

EOF
    exit "$exit_code"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -u|--ubuntu)
            UBUNTU_VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage 1
            ;;
    esac
done

# Build image name
IMAGE_NAME="warp-cpp-test-env"
TAG="${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}"

if [ -n "$REGISTRY" ]; then
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
else
    FULL_IMAGE="${IMAGE_NAME}:${TAG}"
fi

echo "======================================"
echo "Building Warp C++ Test Environment"
echo "======================================"
echo "CUDA Version:    $CUDA_VERSION"
echo "Ubuntu Version:  $UBUNTU_VERSION"
echo "Architecture:    x86_64 only"
echo "Image:           $FULL_IMAGE"
echo "Push:            $PUSH"
echo "======================================"
echo

# Build for x86_64
docker build \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
    -t "$FULL_IMAGE" \
    "$SCRIPT_DIR"

# Push if requested
if [ "$PUSH" = true ]; then
    echo
    echo "Pushing image to registry..."
    docker push "$FULL_IMAGE"
fi

echo
echo "======================================"
echo "Build complete!"
echo "======================================"
echo "Image: $FULL_IMAGE"
echo

if [ "$PUSH" = false ] && [ -n "$REGISTRY" ]; then
    echo "To push this image, run:"
    echo "  docker push $FULL_IMAGE"
    echo
fi

echo "To test the image, run:"
echo "  docker run --rm -it $FULL_IMAGE"
echo
