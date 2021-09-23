#!/bin/bash
#set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

cd _build/packages

echo "Installing Warp to Python"
../target-deps/python/python -m pip install -e .

readelf -d warp/bin/warp.so | grep ORIGIN
readelf -d warp/bin/libnvrtc.so | grep runpath

sudo apt-get install chrpath

chrpath -r '$ORIGIN' warp/bin/libnvrtc.so
chrpath -r '$ORIGIN' warp/bin/libnvrtc.so.10.1

echo "Running tests"
../target-deps/python/python tests/test_ctypes.py