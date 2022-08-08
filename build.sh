#!/bin/bash
set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

# host deps
# tools/packman/packman pull -p linux-x86_64 deps/host-deps.packman.xml
# tools/packman/packman pull -p linux-x86_64 deps/target-deps.packman.xml

# pip deps
./_build/target-deps/python/python -m pip install numpy
./_build/target-deps/python/python build_lib.py --cuda_path="_build/target-deps/cuda"

# copy linux dependencies to bin dir
cp _build/target-deps/cuda/lib64/libnvrtc.so.11.2 warp/bin
cp _build/target-deps/cuda/lib64/libnvrtc-builtins.so.11.3 warp/bin

# set rpath on libnvrtc so we can distribute without the CUDA SDK
# this allows libnvrtc to find libnvrtc-builtins without a CUDA install
# requires the patchelf package
if ! dpkg -l patchelf > /dev/null 2>&1; then
    echo "patchelf missing on system. Attempting to install"
    sudo apt-get install patchelf
fi

patchelf --set-rpath '$ORIGIN' warp/bin/libnvrtc.so.11.2
