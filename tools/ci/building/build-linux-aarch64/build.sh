#!/bin/bash
set -e

if [ "$CI" = "true" ]; then
    echo -e "\\e[0Ksection_start:`date +%s`:install_dependencies[collapsed=true]\\r\\e[0KInstalling dependencies"

    # Print out disk space info to diagnose runner issues
    df -h
fi

USE_LINBUILD=1
BUILD_MODE="release"

# scan command line for options
for arg; do
    shift
    case $arg in
    --no-docker)
        USE_LINBUILD=0
        ;;
    --debug)
        BUILD_MODE="debug"
        ;;
    esac

    # keep all options (including --no-docker) to pass to repo.sh
    if [[ $arg != "--debug" ]]; then
        set -- "$@" "$arg"
    fi 
done

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

"$SCRIPT_DIR/../../../repo.sh" build --fetch-only --config release $@ 

PYTHON="$SCRIPT_DIR/../../../../_build/target-deps/python/python"
LINBUILD="$SCRIPT_DIR/../../../../_build/host-deps/linbuild/linbuild.sh"
CUDA="$SCRIPT_DIR/../../../../_build/target-deps/cuda"

# pip deps
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install numpy
$PYTHON -m pip install gitpython
$PYTHON -m pip install cmake
$PYTHON -m pip install ninja

if [ "$CI" = "true" ]; then
    echo -e "\\e[0Ksection_end:`date +%s`:install_dependencies\\r\\e[0K"
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    $PYTHON "$SCRIPT_DIR/../../../../build_lib.py"
else
    if [ ${USE_LINBUILD} -ne 0 ]; then
        # build with docker for increased compatibility
        $LINBUILD -- $PYTHON "$SCRIPT_DIR/../../../../build_lib.py" --cuda_path=$CUDA --mode=$BUILD_MODE
    else
        # build without docker
        $PYTHON "$SCRIPT_DIR/../../../../build_lib.py" --cuda_path=$CUDA --mode=$BUILD_MODE
    fi
fi
