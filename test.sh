#!/bin/bash
set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

echo "Installing test dependencies"
./_build/target-deps/python/python -m pip install matplotlib
./_build/target-deps/python/python -m pip install usd-core

echo "Installing Warp to Python"
../target-deps/python/python -m pip install -e .

echo "Running tests"
../target-deps/python/python tests/test_all.py