#!/bin/bash
#set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
#source "$SCRIPT_DIR/repo.sh" build --fetch-only $@ || exit $?

"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

cd _build/packages

echo "Installing Warp to Python"
../_build/target-deps/python/python -m pip install -e .

echo "Running tests"
../_build/target-deps/python/python tests/test_ctypes.py