#!/bin/bash
set -e

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
    # build with docker for increased compatibility
    $LINBUILD -- $PYTHON "$SCRIPT_DIR/../../../../build_lib.py" --cuda_path=$CUDA   
fi
