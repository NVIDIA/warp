#!/bin/bash
set -e

USE_LINBUILD=1

# scan command line for options
for arg; do
    shift
    case $arg in
    --no-docker)
        USE_LINBUILD=0
        ;;
    esac
    # keep all options (including --no-docker) to pass to repo.sh
    set -- "$@" "$arg"
done

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

"$SCRIPT_DIR/../../../../repo.sh" build --fetch-only $@ 

PYTHON="$SCRIPT_DIR/../../../../_build/target-deps/python/python"
LINBUILD="$SCRIPT_DIR/../../../../_build/host-deps/linbuild/linbuild.sh"
CUDA="$SCRIPT_DIR/../../../../_build/target-deps/cuda"

# pip deps
$PYTHON -m pip install numpy
$PYTHON -m pip install gitpython
$PYTHON -m pip install cmake
$PYTHON -m pip install ninja

if [[ "$OSTYPE" == "darwin"* ]]; then
    $PYTHON "$SCRIPT_DIR/../../../../build_lib.py"
else
    if [ ${USE_LINBUILD} -ne 0 ]; then
        # build with docker for increased compatibility
        $LINBUILD -- $PYTHON "$SCRIPT_DIR/../../../../build_lib.py" --cuda_path=$CUDA
    else
        # build without docker
        $PYTHON "$SCRIPT_DIR/../../../../build_lib.py" --cuda_path=$CUDA
    fi
fi
