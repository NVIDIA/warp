#!/bin/bash
set -e

if [ "$GITLAB_CI" = "true" ]; then
    echo -e "\\e[0Ksection_start:`date +%s`:install_dependencies[collapsed=true]\\r\\e[0KInstalling dependencies"

    # Print out disk space info to diagnose runner issues
    df -h
fi

USE_LINBUILD=1
BUILD_MODE="release"
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
CUDA_MAJOR_VER="12"

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help            Show this help message and exit."
    echo "  -d, --debug           Enable debug mode."
    echo "  --no-docker           Don't use Linbuild (Docker build)."
    echo "  --cuda MAJOR_VER      Build Warp with a specific major version of the CUDA toolkit."
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -d|--debug)
            BUILD_MODE="debug"
            shift
            ;;
        --no-docker)
            USE_LINBUILD=0
            shift
            ;;
        --cuda)
            if [[ -n "$2" ]]; then
                CUDA_MAJOR_VER="$2"
                shift 2
            else
                echo "Error: --cuda requires a value"
                usage
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

os="$(uname -s)"
os=$(echo "$os" | tr '[:upper:]' '[:lower:]') # to lowercase
if [[ "$os" == "darwin" ]]; then
    os=macos
fi
arch=$(uname -m)
platform="$os-$arch"

source "${SCRIPT_DIR}/../../../packman/packman" pull --platform "${platform}" "${SCRIPT_DIR}/../../../../deps/target-deps.packman.xml" --verbose
source "${SCRIPT_DIR}/../../../packman/packman" pull --platform "${platform}" "${SCRIPT_DIR}/../../../../deps/cuda-toolkit-deps.packman.xml" --verbose --include-tag "cuda-${CUDA_MAJOR_VER}"

PYTHON="$SCRIPT_DIR/../../../../_build/target-deps/python/python"
CUDA="$SCRIPT_DIR/../../../../_build/target-deps/cuda"

# pip deps
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install --upgrade numpy gitpython cmake ninja

if [ "$GITLAB_CI" = "true" ]; then
    echo -e "\\e[0Ksection_end:`date +%s`:install_dependencies\\r\\e[0K"
fi

if [[ "$os" = "macos" ]]; then
    $PYTHON "$SCRIPT_DIR/../../../../build_lib.py"
else
    if [ ${USE_LINBUILD} -ne 0 ]; then
        source "${SCRIPT_DIR}/../../../packman/packman" pull --platform "${platform}" "${SCRIPT_DIR}/../../../../deps/host-deps.packman.xml" --verbose
        LINBUILD="$SCRIPT_DIR/../../../../_build/host-deps/linbuild/linbuild.sh"
        # build with docker for increased compatibility
        $LINBUILD --profile=centos7-gcc10-builder -- $PYTHON "$SCRIPT_DIR/../../../../build_lib.py" --cuda_path=$CUDA --mode=$BUILD_MODE
    else
        # build without docker
        $PYTHON "$SCRIPT_DIR/../../../../build_lib.py" --cuda_path=$CUDA --mode=$BUILD_MODE
    fi
fi
