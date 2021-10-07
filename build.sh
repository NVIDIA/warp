#!/bin/bash
#set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

# pip deps
./_build/target-deps/python/python -m pip install numpy
./_build/target-deps/python/python build_lib.py

# copy linux dependencies to bin dir
cp _build/target-deps/cuda/lib64/libnvrtc* warp.zip!warp/bin
cp _build/target-deps/cuda/lib64/libcudart.* warp.zip!warp/bin

# set rpath on libnvrtc so we can distribute without the CUDA SDK
readelf -d warp/bin/warp.so | grep ORIGIN
readelf -d warp/bin/libnvrtc.so | grep runpath

sudo apt-get install patchelf

patchelf --set-rpath '$ORIGIN' warp/bin/libnvrtc.so
patchelf --set-rpath '$ORIGIN' warp/bin/libnvrtc.so.10.1